# ---------------------------------------------------------------
# FSCD 交互式推理 Demo
# 用法: python DemoFscd.py [--checkpoint <path>] [--model_size base|large]
#
# 操作流程:
#   1. 启动后自动弹出文件选择对话框, 选择一张图片
#   2. 在图片上用鼠标拖拽绘制 exemplar 框 (标注目标样本)
#   3. 按 Enter 键开始推理, 可视化检测结果
#   4. 按 R 键重新选择 exemplar, C 键清除最后一个框
#   5. 按 N 键选择新图片, Q/Esc 退出
# ---------------------------------------------------------------

"""
FSCD 交互式推理 Demo。

支持:
- Base (DINOv2 ViT-S) 和 Large (DINOv2 ViT-B) 模型
- 鼠标绘制 exemplar bounding boxes
- 可视化检测结果 (bbox + count + confidence)
- 实时阈值调整 (T+/T- 键)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfdetr.config import RFDETRFSCDConfig, RFDETRFSCDLargeConfig
from rfdetr.FscdMain import FscdModel
from rfdetr.main import download_pretrain_weights, populate_args
from rfdetr.models.FscdDetr import BuildFscdCriterionAndPostprocessors
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------
COLOR_EXEMPLAR = (0, 255, 255)      # 黄色 (BGR): exemplar 框
COLOR_EXEMPLAR_DRAW = (0, 200, 200) # 绘制中的 exemplar
COLOR_DETECTION = (0, 255, 100)     # 绿色: 检测框
COLOR_HIGH_CONF = (0, 255, 0)       # 高置信度
COLOR_MED_CONF = (0, 200, 255)      # 中置信度 (橙色)
COLOR_LOW_CONF = (0, 100, 255)      # 低置信度 (红色)
COLOR_TEXT_BG = (30, 30, 30)        # 文字背景
COLOR_TEXT = (255, 255, 255)        # 白色文字
COLOR_PANEL_BG = (40, 40, 40)       # 面板背景
COLOR_ACCENT = (255, 180, 0)        # 强调色 (蓝色)


def CvImread(FilePath: str) -> np.ndarray:
    """支持中文/非 ASCII 路径的 imread。

    cv2.imread 不支持中文路径, 用 numpy.fromfile + imdecode 替代。
    """
    Data = np.fromfile(FilePath, dtype=np.uint8)
    Img = cv2.imdecode(Data, cv2.IMREAD_COLOR)
    return Img


# ---------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------

def LoadModel(
    CheckpointPath: str,
    ModelSize: str = "base",
    Device: str = "cuda",
) -> Tuple[torch.nn.Module, object, int]:
    """加载 FSCD 模型和后处理器。

    Returns:
        (Model, Postprocess, Resolution)
    """
    print(f"  Loading {ModelSize} model from: {CheckpointPath}")

    if ModelSize == "large":
        Cfg = RFDETRFSCDLargeConfig()
    else:
        Cfg = RFDETRFSCDConfig()

    Resolution = Cfg.resolution

    # 构建模型结构
    download_pretrain_weights(Cfg.pretrain_weights)
    Inst = FscdModel(**Cfg.model_dump())
    Model = Inst.Model

    # 构建后处理器
    MergedKwargs = {**Cfg.model_dump()}
    MergedKwargs.update({"dataset_file": "fscd147", "num_classes": 1})
    FullArgs = populate_args(**MergedKwargs)
    _, PostprocessDict = BuildFscdCriterionAndPostprocessors(FullArgs)
    Postprocess = PostprocessDict

    # 加载训练好的权重
    Ckpt = torch.load(CheckpointPath, map_location="cpu", weights_only=False)
    StateDict = Ckpt["model"] if "model" in Ckpt else Ckpt
    MissingKeys, UnexpectedKeys = Model.load_state_dict(StateDict, strict=False)
    if MissingKeys:
        print(f"  Warning: missing keys: {MissingKeys[:5]}...")
    if UnexpectedKeys:
        print(f"  Warning: unexpected keys: {UnexpectedKeys[:5]}...")

    Model.to(Device)
    Model.eval()

    Epoch = Ckpt.get("epoch", "?")
    print(f"  Model loaded (epoch {Epoch}), resolution={Resolution}")

    return Model, Postprocess, Resolution


# ---------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------

# ImageNet 均值/标准差
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def PreprocessImage(
    ImgBgr: np.ndarray,
    Resolution: int,
    Device: str = "cuda",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """将 BGR OpenCV 图像预处理为模型输入。

    Returns:
        (BatchTensor [1, 3, R, R], OrigSize (H, W))
    """
    ImgRgb = cv2.cvtColor(ImgBgr, cv2.COLOR_BGR2RGB)
    OrigH, OrigW = ImgRgb.shape[:2]

    # PIL → tensor
    Pil = Image.fromarray(ImgRgb)
    Tensor = TF.to_tensor(Pil)  # [3, H, W], float32 [0, 1]
    Tensor = TF.normalize(Tensor, IMAGENET_MEAN, IMAGENET_STD)
    Tensor = TF.resize(Tensor, [Resolution, Resolution])
    Batch = Tensor.unsqueeze(0).to(Device)

    return Batch, (OrigH, OrigW)


# ---------------------------------------------------------------
# 推理
# ---------------------------------------------------------------

@torch.inference_mode()
def RunInference(
    Model: torch.nn.Module,
    Postprocess: object,
    ImgBgr: np.ndarray,
    ExemplarBoxes: List[List[int]],
    Resolution: int,
    Threshold: float = 0.3,
    Device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """执行 FSCD 推理。

    Args:
        ExemplarBoxes: [[x1, y1, x2, y2], ...] 像素坐标

    Returns:
        (Boxes [N, 4] xyxy 像素坐标, Scores [N], Count)
    """
    Batch, (OrigH, OrigW) = PreprocessImage(ImgBgr, Resolution, Device)

    # 归一化 exemplar boxes 到 [0, 1] (相对于模型输入分辨率)
    ExNorm = []
    for Box in ExemplarBoxes:
        X1, Y1, X2, Y2 = Box
        ExNorm.append([
            X1 / OrigW,
            Y1 / OrigH,
            X2 / OrigW,
            Y2 / OrigH,
        ])

    ExTensor = torch.tensor([ExNorm], dtype=torch.float32, device=Device)

    # Forward
    Outputs = Model(Batch, ExTensor)

    # Postprocess → 原始图像坐标
    TargetSizes = torch.tensor([[OrigH, OrigW]], device=Device)
    Results = Postprocess(Outputs, TargetSizes)

    Result = Results[0]
    Scores = Result["scores"].cpu().numpy()
    Boxes = Result["boxes"].cpu().numpy()

    # 按阈值过滤
    Keep = Scores > Threshold
    Boxes = Boxes[Keep]
    Scores = Scores[Keep]
    Count = len(Scores)

    return Boxes, Scores, Count


# ---------------------------------------------------------------
# 交互式 UI
# ---------------------------------------------------------------

class FscdDemo:
    """FSCD 交互式推理 demo。"""

    def __init__(
        self,
        Model: torch.nn.Module,
        Postprocess: object,
        Resolution: int,
        Device: str = "cuda",
    ) -> None:
        self.Model = Model
        self.Postprocess = Postprocess
        self.Resolution = Resolution
        self.Device = Device
        self.Threshold = 0.3

        # 状态
        self.ImgOrig: np.ndarray = None
        self.ImgDisplay: np.ndarray = None
        self.ExemplarBoxes: List[List[int]] = []  # [[x1, y1, x2, y2], ...]
        self.Drawing = False
        self.DrawStart = (0, 0)
        self.DrawEnd = (0, 0)
        self.DetectionBoxes = None
        self.DetectionScores = None
        self.DetectionCount = 0
        self.ShowDetections = False
        self.WindowName = "FSCD Demo"
        self.PanelH = 40

    def _OpenFileDialog(self) -> str:
        """打开文件选择对话框。"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            Root = tk.Tk()
            Root.withdraw()
            Root.attributes("-topmost", True)
            FilePath = filedialog.askopenfilename(
                title="选择图片",
                filetypes=[
                    ("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                    ("所有文件", "*.*"),
                ],
            )
            Root.destroy()
            return FilePath
        except Exception:
            return input("  请输入图片路径: ").strip().strip('"')

    def _MouseCallback(self, Event: int, X: int, Y: int, Flags: int, Param) -> None:
        """鼠标事件回调。"""
        if self.ShowDetections:
            return

        # 鼠标坐标包含顶部面板, 需要换算到图像坐标
        YImg = Y - self.PanelH
        if YImg < 0:
            return

        if Event == cv2.EVENT_LBUTTONDOWN:
            self.Drawing = True
            self.DrawStart = (X, YImg)
            self.DrawEnd = (X, YImg)

        elif Event == cv2.EVENT_MOUSEMOVE and self.Drawing:
            self.DrawEnd = (X, YImg)
            self._Redraw()

        elif Event == cv2.EVENT_LBUTTONUP and self.Drawing:
            self.Drawing = False
            self.DrawEnd = (X, YImg)
            X1 = min(self.DrawStart[0], self.DrawEnd[0])
            Y1 = min(self.DrawStart[1], self.DrawEnd[1])
            X2 = max(self.DrawStart[0], self.DrawEnd[0])
            Y2 = max(self.DrawStart[1], self.DrawEnd[1])

            # 最小框大小 (避免误点击)
            if (X2 - X1) > 5 and (Y2 - Y1) > 5:
                self.ExemplarBoxes.append([X1, Y1, X2, Y2])
            self._Redraw()

    def _GetConfidenceColor(self, Score: float) -> Tuple[int, int, int]:
        """根据置信度返回颜色。"""
        if Score > 0.7:
            return COLOR_HIGH_CONF
        elif Score > 0.5:
            return COLOR_MED_CONF
        else:
            return COLOR_LOW_CONF

    def _DrawPanel(self, Img: np.ndarray) -> np.ndarray:
        """在图像顶部绘制信息面板。"""
        H, W = Img.shape[:2]
        PanelH = self.PanelH

        # 创建带面板的图像
        Canvas = np.zeros((H + PanelH, W, 3), dtype=np.uint8)
        Canvas[:PanelH] = COLOR_PANEL_BG
        Canvas[PanelH:] = Img

        # 状态文字
        if self.ShowDetections:
            StatusText = (
                f"Count: {self.DetectionCount} | "
                f"Threshold: {self.Threshold:.2f} | "
                f"[T+/T-] Adjust | [R] Reset | [N] New Image | [Q] Quit"
            )
        else:
            NumEx = len(self.ExemplarBoxes)
            StatusText = (
                f"Exemplars: {NumEx} | "
                f"Draw boxes on target objects | "
                f"[Enter] Infer | [C] Undo | [R] Clear | [N] New Image | [Q] Quit"
            )

        cv2.putText(
            Canvas, StatusText, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA,
        )

        return Canvas

    def _Redraw(self) -> None:
        """重绘当前显示。"""
        if self.ImgOrig is None:
            return

        Img = self.ImgOrig.copy()

        if self.ShowDetections:
            # 绘制检测结果
            if self.DetectionBoxes is not None:
                for I in range(len(self.DetectionBoxes)):
                    X1, Y1, X2, Y2 = self.DetectionBoxes[I].astype(int)
                    Score = self.DetectionScores[I]
                    Color = self._GetConfidenceColor(Score)

                    # 半透明填充
                    Overlay = Img.copy()
                    cv2.rectangle(Overlay, (X1, Y1), (X2, Y2), Color, -1)
                    cv2.addWeighted(Overlay, 0.15, Img, 0.85, 0, Img)

                    # 边框
                    cv2.rectangle(Img, (X1, Y1), (X2, Y2), Color, 2)

                    # 置信度标签
                    Label = f"{Score:.2f}"
                    (Tw, Th), _ = cv2.getTextSize(
                        Label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1,
                    )
                    cv2.rectangle(
                        Img, (X1, Y1 - Th - 6), (X1 + Tw + 4, Y1), Color, -1,
                    )
                    cv2.putText(
                        Img, Label, (X1 + 2, Y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0), 1, cv2.LINE_AA,
                    )

            # 绘制 exemplar 框 (虚线效果)
            for Box in self.ExemplarBoxes:
                X1, Y1, X2, Y2 = Box
                cv2.rectangle(Img, (X1, Y1), (X2, Y2), COLOR_EXEMPLAR, 2)
                cv2.putText(
                    Img, "E", (X1 + 3, Y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_EXEMPLAR, 1, cv2.LINE_AA,
                )

            # 计数显示 (右上角)
            CountText = f"Count: {self.DetectionCount}"
            (Tw, Th), _ = cv2.getTextSize(
                CountText, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3,
            )
            H, W = Img.shape[:2]
            Cx, Cy = W - Tw - 20, 50
            cv2.rectangle(
                Img, (Cx - 10, Cy - Th - 10), (Cx + Tw + 10, Cy + 10),
                COLOR_PANEL_BG, -1,
            )
            cv2.putText(
                Img, CountText, (Cx, Cy),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_ACCENT, 3, cv2.LINE_AA,
            )
        else:
            # 绘制已有 exemplar 框
            for I, Box in enumerate(self.ExemplarBoxes):
                X1, Y1, X2, Y2 = Box
                cv2.rectangle(Img, (X1, Y1), (X2, Y2), COLOR_EXEMPLAR, 2)
                cv2.putText(
                    Img, f"E{I + 1}", (X1 + 3, Y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EXEMPLAR, 1, cv2.LINE_AA,
                )

            # 绘制正在绘制的框
            if self.Drawing:
                cv2.rectangle(
                    Img, self.DrawStart, self.DrawEnd,
                    COLOR_EXEMPLAR_DRAW, 2,
                )

        self.ImgDisplay = self._DrawPanel(Img)
        cv2.imshow(self.WindowName, self.ImgDisplay)

    def _RunInference(self) -> None:
        """执行推理并显示结果。"""
        if not self.ExemplarBoxes:
            print("  No exemplar boxes! Please draw at least one box.")
            return

        print(f"  Running inference with {len(self.ExemplarBoxes)} exemplar(s), "
              f"threshold={self.Threshold:.2f} ...")

        Boxes, Scores, Count = RunInference(
            Model=self.Model,
            Postprocess=self.Postprocess,
            ImgBgr=self.ImgOrig,
            ExemplarBoxes=self.ExemplarBoxes,
            Resolution=self.Resolution,
            Threshold=self.Threshold,
            Device=self.Device,
        )

        self.DetectionBoxes = Boxes
        self.DetectionScores = Scores
        self.DetectionCount = Count
        self.ShowDetections = True

        print(f"  Detected {Count} objects")
        self._Redraw()

    def _LoadImage(self, ImgPath: str) -> bool:
        """加载图片。"""
        if not ImgPath or not os.path.isfile(ImgPath):
            print(f"  Invalid file: {ImgPath}")
            return False

        Img = CvImread(ImgPath)
        if Img is None:
            print(f"  Cannot read: {ImgPath}")
            return False

        # 限制显示尺寸
        MaxDim = 1200
        H, W = Img.shape[:2]
        if max(H, W) > MaxDim:
            Scale = MaxDim / max(H, W)
            Img = cv2.resize(
                Img,
                (int(W * Scale), int(H * Scale)),
                interpolation=cv2.INTER_AREA,
            )

        self.ImgOrig = Img
        self.ExemplarBoxes = []
        self.ShowDetections = False
        self.DetectionBoxes = None
        self.DetectionScores = None
        self.DetectionCount = 0

        print(f"  Loaded: {os.path.basename(ImgPath)} ({Img.shape[1]}x{Img.shape[0]})")
        return True

    def Run(self, InitImgPath: str = None) -> None:
        """启动交互式 demo。"""
        print()
        print("  ┌────────────────────────────────────────────┐")
        print("  │         FSCD Interactive Demo              │")
        print("  ├────────────────────────────────────────────┤")
        print("  │  Mouse : Draw exemplar boxes               │")
        print("  │  Enter : Run inference                     │")
        print("  │  T+/T- : Adjust threshold (±0.05)         │")
        print("  │  C     : Undo last box                    │")
        print("  │  R     : Reset (clear all)                │")
        print("  │  N     : Open new image                   │")
        print("  │  S     : Save result                      │")
        print("  │  Q/Esc : Quit                             │")
        print("  └────────────────────────────────────────────┘")
        print()

        # 加载初始图片
        if InitImgPath and os.path.isfile(InitImgPath):
            self._LoadImage(InitImgPath)
        else:
            ImgPath = self._OpenFileDialog()
            if not self._LoadImage(ImgPath):
                print("  No image selected, exiting.")
                return

        cv2.namedWindow(self.WindowName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WindowName, self._MouseCallback)
        self._Redraw()

        while True:
            Key = cv2.waitKey(30) & 0xFF

            if Key == ord("q") or Key == 27:  # Q / Esc
                break

            elif Key == 13:  # Enter → 推理
                self._RunInference()

            elif Key == ord("c"):  # C → 撤销最后一个 exemplar
                if not self.ShowDetections and self.ExemplarBoxes:
                    self.ExemplarBoxes.pop()
                    self._Redraw()

            elif Key == ord("r"):  # R → 重置
                self.ExemplarBoxes = []
                self.ShowDetections = False
                self.DetectionBoxes = None
                self.DetectionScores = None
                self.DetectionCount = 0
                self._Redraw()

            elif Key == ord("n"):  # N → 新图片
                ImgPath = self._OpenFileDialog()
                if self._LoadImage(ImgPath):
                    self._Redraw()

            elif Key == ord("s"):  # S → 保存结果
                if self.ImgDisplay is not None:
                    SavePath = "fscd_result.png"
                    cv2.imwrite(SavePath, self.ImgDisplay)
                    print(f"  Saved: {SavePath}")

            elif Key == ord("t"):  # T → 提高阈值
                self.Threshold = min(0.99, self.Threshold + 0.01)
                print(f"  Threshold: {self.Threshold:.2f}")
                if self.ShowDetections:
                    self._RunInference()

            elif Key == ord("g"):  # G → 降低阈值
                self.Threshold = max(0.01, self.Threshold - 0.05)
                print(f"  Threshold: {self.Threshold:.2f}")
                if self.ShowDetections:
                    self._RunInference()

        cv2.destroyAllWindows()
        print("  Demo closed.")


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def ParseArgs() -> argparse.Namespace:
    """解析命令行参数。"""
    P = argparse.ArgumentParser(description="FSCD Interactive Demo")

    P.add_argument(
        "--checkpoint", type=str,
        default="output_fscd/checkpoint0009.pth",
        help="训练好的 checkpoint 路径",
    )
    P.add_argument(
        "--model_size", type=str, default="base",
        choices=["base", "large"],
        help="模型规模 (base / large)",
    )
    P.add_argument(
        "--image", type=str, default=None,
        help="初始图片路径 (不指定则弹出文件选择框)",
    )
    P.add_argument(
        "--threshold", type=float, default=0.3,
        help="初始置信度阈值",
    )
    P.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备 (cuda / cpu)",
    )

    return P.parse_args()


if __name__ == "__main__":
    Args = ParseArgs()

    print()
    print("  ═══════════════════════════════════════════════")
    print("  FSCD-147 Interactive Inference Demo")
    print("  ═══════════════════════════════════════════════")
    print()

    # 加载模型
    Model, Postprocess, Resolution = LoadModel(
        CheckpointPath=Args.checkpoint,
        ModelSize=Args.model_size,
        Device=Args.device,
    )

    # 启动 demo
    Demo = FscdDemo(
        Model=Model,
        Postprocess=Postprocess,
        Resolution=Resolution,
        Device=Args.device,
    )
    Demo.Threshold = Args.threshold
    Demo.Run(InitImgPath=Args.image)
