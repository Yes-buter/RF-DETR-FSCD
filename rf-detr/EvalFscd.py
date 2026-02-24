# ---------------------------------------------------------------
# FSCD Benchmark 评估脚本
#
# 支持:
#   - FSCD-147 val/test 集
#   - FSCD-LVIS test 集 (待数据下载后可用)
#
# 评估指标:
#   - 计数: MAE, RMSE, NAE, SRE
#   - 检测: AP, AP50, AP75, AP_S, AP_M, AP_L
#
# 用法:
#   python EvalFscd.py --checkpoint output_fscd/checkpoint_best.pth \
#                      --dataset_dir C:\Users\20683\Desktop\FSCD147 \
#                      --split test
#
#   python EvalFscd.py --checkpoint output_fscd/checkpoint_best.pth \
#                      --dataset fscd_lvis \
#                      --dataset_dir C:\Users\20683\Desktop\FSCD_LVIS \
#                      --split test
# ---------------------------------------------------------------

"""FSCD Benchmark Evaluation Script.

对标 Counting-DETR / DAVE / GeCo 等论文的评估协议:
- FSCD-147: val/test 集, 每张图 3 个 exemplar box, class-agnostic
- FSCD-LVIS: test 集, 每张图 3 个 exemplar box, multi-class (377 categories)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfdetr.config import RFDETRFSCDConfig, FSCDTrainConfig
from rfdetr.FscdMain import FscdModel
from rfdetr.main import download_pretrain_weights, populate_args
from rfdetr.models.FscdDetr import BuildFscdCriterionAndPostprocessors


# ---------------------------------------------------------------
# 常量
# ---------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------

def LoadModel(
    CheckpointPath: str,
    ModelSize: str = "base",
    Device: str = "cuda",
) -> Tuple[torch.nn.Module, object, int]:
    """加载 FSCD 模型。

    Returns:
        (Model, Postprocess, Resolution)
    """
    # 延迟导入, 避免循环依赖
    try:
        from rfdetr.config import RFDETRFSCDLargeConfig
    except ImportError:
        RFDETRFSCDLargeConfig = None

    if ModelSize == "large":
        if RFDETRFSCDLargeConfig is None:
            raise RuntimeError("RFDETRFSCDLargeConfig 未定义, 请更新 config.py")
        Cfg = RFDETRFSCDLargeConfig()
    else:
        Cfg = RFDETRFSCDConfig()

    Resolution = Cfg.resolution

    # 构建模型
    download_pretrain_weights(Cfg.pretrain_weights)
    Inst = FscdModel(**Cfg.model_dump())
    Model = Inst.Model

    # 构建后处理器
    MergedKwargs = {**Cfg.model_dump()}
    MergedKwargs.update({"dataset_file": "fscd147", "num_classes": 1})
    FullArgs = populate_args(**MergedKwargs)
    _, Postprocess = BuildFscdCriterionAndPostprocessors(FullArgs)

    # 加载训练好的权重
    Ckpt = torch.load(CheckpointPath, map_location="cpu", weights_only=False)
    StateDict = Ckpt["model"] if "model" in Ckpt else Ckpt
    Missing, Unexpected = Model.load_state_dict(StateDict, strict=False)
    if Missing:
        print(f"  Warning: {len(Missing)} missing keys")
    if Unexpected:
        print(f"  Warning: {len(Unexpected)} unexpected keys")

    Model.to(Device)
    Model.eval()

    Epoch = Ckpt.get("epoch", "?")
    print(f"  Model loaded (epoch {Epoch}), size={ModelSize}, resolution={Resolution}")

    return Model, Postprocess, Resolution


# ---------------------------------------------------------------
# 图像预处理
# ---------------------------------------------------------------

def PreprocessImage(
    ImgPil: Image.Image,
    Resolution: int,
    Device: str = "cuda",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """预处理单张图像为模型输入 tensor。"""
    W, H = ImgPil.size
    Tensor = TF.to_tensor(ImgPil)
    Tensor = TF.normalize(Tensor, IMAGENET_MEAN, IMAGENET_STD)
    Tensor = TF.resize(Tensor, [Resolution, Resolution])
    return Tensor.unsqueeze(0).to(Device), (H, W)


# ---------------------------------------------------------------
# FSCD-147 数据加载
# ---------------------------------------------------------------

def LoadFscd147(
    DatasetDir: str,
    Split: str = "test",
) -> Tuple[List[dict], Optional[object]]:
    """加载 FSCD-147 数据集。

    Returns:
        (Samples, CocoGt)
        - Samples: [{filename, image_path, exemplar_boxes_xyxy, gt_count, coco_image_id}, ...]
        - CocoGt: pycocotools.coco.COCO 对象 (如果有 instances_*.json)
    """
    AnnFile = os.path.join(DatasetDir, "annotation_FSC147_384.json")
    SplitFile = os.path.join(DatasetDir, "Train_Test_Val_FSC_147.json")
    ImgDir = os.path.join(DatasetDir, "images_384_VarV2")

    with open(AnnFile, "r") as Fh:
        Annotations = json.load(Fh)
    with open(SplitFile, "r") as Fh:
        Splits = json.load(Fh)

    FileNames = Splits[Split]

    # COCO GT (val/test 有 instances_*.json)
    CocoAnnFile = os.path.join(DatasetDir, f"instances_{Split}.json")
    CocoGt = None
    CocoFileToId = {}
    if os.path.exists(CocoAnnFile):
        from pycocotools.coco import COCO
        CocoGt = COCO(CocoAnnFile)
        for ImgInfo in CocoGt.dataset["images"]:
            CocoFileToId[ImgInfo["file_name"]] = ImgInfo["id"]

    Samples = []
    for FName in FileNames:
        if FName not in Annotations:
            continue

        Ann = Annotations[FName]
        ImgPath = os.path.join(ImgDir, FName)
        if not os.path.exists(ImgPath):
            continue

        # Exemplar boxes: 4角点 → xyxy (384 缩放版)
        RatioH = Ann.get("ratio_h", 1.0)
        RatioW = Ann.get("ratio_w", 1.0)
        ExBoxes = []
        for Corners in Ann["box_examples_coordinates"][:3]:
            Ys = [C[0] for C in Corners]
            Xs = [C[1] for C in Corners]
            X1 = min(Xs) * RatioW
            Y1 = min(Ys) * RatioH
            X2 = max(Xs) * RatioW
            Y2 = max(Ys) * RatioH
            ExBoxes.append([X1, Y1, X2, Y2])

        # GT count = len(points)
        GtCount = len(Ann.get("points", []))

        Sample = {
            "filename": FName,
            "image_path": ImgPath,
            "exemplar_boxes": ExBoxes,
            "gt_count": GtCount,
            "coco_image_id": CocoFileToId.get(FName, None),
        }
        Samples.append(Sample)

    print(f"  FSCD-147 {Split}: {len(Samples)} images loaded")
    return Samples, CocoGt


# ---------------------------------------------------------------
# FSCD-LVIS 数据加载
# ---------------------------------------------------------------

def LoadFscdLvis(
    DatasetDir: str,
    Split: str = "test",
) -> Tuple[List[dict], Optional[object]]:
    """加载 FSCD-LVIS 数据集。

    FSCD-LVIS 目录结构 (预期):
      FSCD_LVIS/
      ├── images/          或 images_384/
      ├── annotations/
      │   ├── fscd_lvis_test.json    (LVIS/COCO 格式)
      │   └── fscd_lvis_train.json
      └── exemplars.json             (每张图的 exemplar boxes)

    Returns:
        (Samples, CocoGt)
    """
    # 尝试多种可能的目录布局
    PossibleAnnFiles = [
        os.path.join(DatasetDir, "annotations", f"fscd_lvis_{Split}.json"),
        os.path.join(DatasetDir, f"fscd_lvis_{Split}.json"),
        os.path.join(DatasetDir, "annotations", f"{Split}.json"),
        os.path.join(DatasetDir, f"instances_{Split}.json"),
    ]

    AnnFile = None
    for P in PossibleAnnFiles:
        if os.path.exists(P):
            AnnFile = P
            break

    if AnnFile is None:
        print(f"  ERROR: FSCD-LVIS {Split} annotation not found in {DatasetDir}")
        print(f"  Searched: {PossibleAnnFiles}")
        return [], None

    # 图像目录
    PossibleImgDirs = [
        os.path.join(DatasetDir, "images"),
        os.path.join(DatasetDir, "images_384"),
        DatasetDir,
    ]
    ImgDir = None
    for P in PossibleImgDirs:
        if os.path.isdir(P):
            ImgDir = P
            break

    # Exemplar boxes
    PossibleExemplarFiles = [
        os.path.join(DatasetDir, "exemplars.json"),
        os.path.join(DatasetDir, "annotations", "exemplars.json"),
        os.path.join(DatasetDir, "exemplar_boxes.json"),
    ]
    ExemplarFile = None
    for P in PossibleExemplarFiles:
        if os.path.exists(P):
            ExemplarFile = P
            break

    from pycocotools.coco import COCO
    CocoGt = COCO(AnnFile)

    # 加载 exemplar boxes
    ExemplarData = {}
    if ExemplarFile:
        with open(ExemplarFile, "r") as Fh:
            ExemplarData = json.load(Fh)

    Samples = []
    for ImgInfo in CocoGt.dataset["images"]:
        ImgId = ImgInfo["id"]
        FName = ImgInfo["file_name"]

        # 查找图片路径
        ImgPath = os.path.join(ImgDir, FName)
        if not os.path.exists(ImgPath):
            # 尝试去掉子目录前缀
            ImgPath = os.path.join(ImgDir, os.path.basename(FName))
            if not os.path.exists(ImgPath):
                continue

        # 获取 exemplar boxes
        ExKey = str(ImgId) if str(ImgId) in ExemplarData else FName
        ExBoxes = []
        if ExKey in ExemplarData:
            RawBoxes = ExemplarData[ExKey]
            if isinstance(RawBoxes, dict) and "boxes" in RawBoxes:
                RawBoxes = RawBoxes["boxes"]
            # 格式: [[x1,y1,x2,y2], ...] 或 [[x,y,w,h], ...]
            for Box in RawBoxes[:3]:
                if len(Box) == 4:
                    ExBoxes.append(Box)
        else:
            # 没有预定义 exemplar: 从 GT annotations 随机选 3 个
            AnnIds = CocoGt.getAnnIds(imgIds=[ImgId])
            Anns = CocoGt.loadAnns(AnnIds)
            if len(Anns) < 1:
                continue
            SelectedAnns = Anns[:3] if len(Anns) >= 3 else Anns * 3
            SelectedAnns = SelectedAnns[:3]
            for Ann in SelectedAnns:
                X, Y, W, H = Ann["bbox"]
                ExBoxes.append([X, Y, X + W, Y + H])

        GtCount = len(CocoGt.getAnnIds(imgIds=[ImgId]))

        Sample = {
            "filename": FName,
            "image_path": ImgPath,
            "exemplar_boxes": ExBoxes,
            "gt_count": GtCount,
            "coco_image_id": ImgId,
        }
        Samples.append(Sample)

    print(f"  FSCD-LVIS {Split}: {len(Samples)} images loaded")
    return Samples, CocoGt


# ---------------------------------------------------------------
# 推理 (逐张)
# ---------------------------------------------------------------

@torch.inference_mode()
def RunSingleInference(
    Model: torch.nn.Module,
    Postprocess: object,
    ImgPil: Image.Image,
    ExemplarBoxes: List[List[float]],
    Resolution: int,
    Device: str = "cuda",
) -> dict:
    """单张图推理。

    Args:
        ExemplarBoxes: [[x1,y1,x2,y2], ...] 像素坐标 (384 版本图像尺度)

    Returns:
        {
            "scores": np.ndarray [N],
            "boxes": np.ndarray [N, 4] xyxy 原图坐标,
            "soft_count": float,
            "labels": np.ndarray [N],
        }
    """
    W, H = ImgPil.size
    Batch, (OrigH, OrigW) = PreprocessImage(ImgPil, Resolution, Device)

    # 归一化 exemplar boxes 到 [0, 1]
    ExNorm = []
    for Box in ExemplarBoxes:
        X1, Y1, X2, Y2 = Box
        ExNorm.append([X1 / W, Y1 / H, X2 / W, Y2 / H])

    ExTensor = torch.tensor([ExNorm], dtype=torch.float32, device=Device)

    # Forward
    Outputs = Model(Batch, ExTensor)

    # Soft count
    PredLogits = Outputs["pred_logits"]  # [1, NQ, 1]
    SoftCount = PredLogits.sigmoid().squeeze(-1).sum().item()

    # Postprocess → 原图坐标
    TargetSizes = torch.tensor([[OrigH, OrigW]], device=Device)
    Results = Postprocess(Outputs, TargetSizes)

    R = Results[0]
    return {
        "scores": R["scores"].cpu().numpy(),
        "boxes": R["boxes"].cpu().numpy(),
        "labels": R["labels"].cpu().numpy(),
        "soft_count": SoftCount,
    }


# ---------------------------------------------------------------
# 计数指标
# ---------------------------------------------------------------

def ComputeCountingMetrics(
    PredCounts: np.ndarray,
    GtCounts: np.ndarray,
) -> dict:
    """计算计数评估指标。

    Returns:
        {mae, rmse, nae, sre}
    """
    AbsErr = np.abs(PredCounts - GtCounts)
    SqErr = (PredCounts - GtCounts) ** 2

    Mae = np.mean(AbsErr)
    Rmse = np.sqrt(np.mean(SqErr))

    # NAE: Normalized Absolute Error
    Nae = np.mean(AbsErr / np.maximum(GtCounts, 1.0))

    # SRE: Squared Relative Error
    Sre = np.mean(SqErr / np.maximum(GtCounts, 1.0) ** 2)

    return {
        "mae": float(Mae),
        "rmse": float(Rmse),
        "nae": float(Nae),
        "sre": float(Sre),
    }


# ---------------------------------------------------------------
# 评估主逻辑
# ---------------------------------------------------------------

def Evaluate(
    Model: torch.nn.Module,
    Postprocess: object,
    Samples: List[dict],
    CocoGt: Optional[object],
    Resolution: int,
    ThresholdForCount: float = 0.5,
    Device: str = "cuda",
) -> dict:
    """完整评估。

    Returns:
        Stats dict 包含所有指标
    """
    SoftCounts = []
    HardCounts = []
    GtCounts = []

    # COCO 评估
    CocoEvaluator = None
    if CocoGt is not None:
        from rfdetr.datasets.coco_eval import CocoEvaluator as _CocoEval
        CocoEvaluator = _CocoEval(CocoGt, ("bbox",))

    Pbar = tqdm(Samples, desc="  Evaluating", bar_format="{l_bar}{bar:30}{r_bar}")

    for Sample in Pbar:
        ImgPath = Sample["image_path"]
        ExBoxes = Sample["exemplar_boxes"]
        GtCount = Sample["gt_count"]

        ImgPil = Image.open(ImgPath).convert("RGB")

        Result = RunSingleInference(
            Model, Postprocess, ImgPil, ExBoxes, Resolution, Device,
        )

        # Soft count
        SoftCounts.append(Result["soft_count"])

        # Hard count
        HardCount = int((Result["scores"] > ThresholdForCount).sum())
        HardCounts.append(HardCount)
        GtCounts.append(GtCount)

        # COCO 评估
        CocoImgId = Sample.get("coco_image_id")
        if CocoEvaluator is not None and CocoImgId is not None:
            # class-agnostic label 0 → COCO category 1
            Res = {
                CocoImgId: {
                    "scores": torch.tensor(Result["scores"]),
                    "labels": torch.tensor(Result["labels"]) + 1,
                    "boxes": torch.tensor(Result["boxes"]),
                }
            }
            CocoEvaluator.update(Res)

        # 进度条
        CurSoftMae = np.mean(np.abs(
            np.array(SoftCounts) - np.array(GtCounts)
        ))
        Pbar.set_postfix_str(f"soft_MAE={CurSoftMae:.2f}")

    Pbar.close()

    SoftCounts = np.array(SoftCounts)
    HardCounts = np.array(HardCounts, dtype=float)
    GtCounts = np.array(GtCounts, dtype=float)

    # 计数指标
    SoftMetrics = ComputeCountingMetrics(SoftCounts, GtCounts)
    HardMetrics = ComputeCountingMetrics(HardCounts, GtCounts)

    Stats = {
        "soft_mae": SoftMetrics["mae"],
        "soft_rmse": SoftMetrics["rmse"],
        "soft_nae": SoftMetrics["nae"],
        "soft_sre": SoftMetrics["sre"],
        "hard_mae": HardMetrics["mae"],
        "hard_rmse": HardMetrics["rmse"],
        "hard_nae": HardMetrics["nae"],
        "hard_sre": HardMetrics["sre"],
        "threshold": ThresholdForCount,
        "num_samples": len(Samples),
        "mean_gt_count": float(np.mean(GtCounts)),
        "mean_soft_count": float(np.mean(SoftCounts)),
        "mean_hard_count": float(np.mean(HardCounts)),
    }

    # COCO AP 指标
    if CocoEvaluator is not None:
        CocoEvaluator.synchronize_between_processes()
        CocoEvaluator.accumulate()
        CocoEvaluator.summarize()

        CocoStats = CocoEvaluator.coco_eval["bbox"].stats
        Stats.update({
            "ap": float(CocoStats[0]),
            "ap50": float(CocoStats[1]),
            "ap75": float(CocoStats[2]),
            "ap_s": float(CocoStats[3]),
            "ap_m": float(CocoStats[4]),
            "ap_l": float(CocoStats[5]),
            "ar_1": float(CocoStats[6]),
            "ar_10": float(CocoStats[7]),
            "ar_100": float(CocoStats[8]),
        })

    return Stats


# ---------------------------------------------------------------
# 结果打印
# ---------------------------------------------------------------

def PrintResults(
    Stats: dict,
    DatasetName: str,
    Split: str,
) -> None:
    """格式化打印评估结果。"""
    print()
    W = 58
    print(f"  {'═' * W}")
    print(f"  ║  {DatasetName} ({Split}) — {Stats['num_samples']} images")
    print(f"  {'═' * W}")

    # ---- 计数指标 ----
    print(f"  ║")
    print(f"  ║  Counting Metrics")
    print(f"  ║  {'─' * 50}")
    print(f"  ║  {'Metric':<20} {'Soft Count':>14} {'Hard Count':>14}")
    print(f"  ║  {'─' * 50}")
    print(f"  ║  {'MAE':<20} {Stats['soft_mae']:>14.4f} {Stats['hard_mae']:>14.4f}")
    print(f"  ║  {'RMSE':<20} {Stats['soft_rmse']:>14.4f} {Stats['hard_rmse']:>14.4f}")
    print(f"  ║  {'NAE':<20} {Stats['soft_nae']:>14.4f} {Stats['hard_nae']:>14.4f}")
    print(f"  ║  {'SRE':<20} {Stats['soft_sre']:>14.4f} {Stats['hard_sre']:>14.4f}")
    print(f"  ║  {'─' * 50}")
    print(f"  ║  Avg GT Count:   {Stats['mean_gt_count']:.2f}")
    print(f"  ║  Avg Soft Count: {Stats['mean_soft_count']:.2f}")
    print(f"  ║  Avg Hard Count: {Stats['mean_hard_count']:.2f}  (threshold={Stats['threshold']:.2f})")

    # ---- 检测指标 ----
    if "ap" in Stats:
        print(f"  ║")
        print(f"  ║  Detection Metrics (COCO)")
        print(f"  ║  {'─' * 50}")
        print(f"  ║  {'AP':.<25} {Stats['ap']:.4f}")
        print(f"  ║  {'AP50':.<25} {Stats['ap50']:.4f}")
        print(f"  ║  {'AP75':.<25} {Stats['ap75']:.4f}")
        print(f"  ║  {'AP (small)':.<25} {Stats['ap_s']:.4f}")
        print(f"  ║  {'AP (medium)':.<25} {Stats['ap_m']:.4f}")
        print(f"  ║  {'AP (large)':.<25} {Stats['ap_l']:.4f}")
        print(f"  ║  {'AR@1':.<25} {Stats['ar_1']:.4f}")
        print(f"  ║  {'AR@10':.<25} {Stats['ar_10']:.4f}")
        print(f"  ║  {'AR@100':.<25} {Stats['ar_100']:.4f}")

    print(f"  {'═' * W}")
    print()


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def ParseArgs() -> argparse.Namespace:
    """解析命令行参数。"""
    P = argparse.ArgumentParser(
        description="FSCD Benchmark Evaluation",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    P.add_argument(
        "--checkpoint", type=str, required=True,
        help="训练好的 checkpoint 路径",
    )
    P.add_argument(
        "--dataset", type=str, default="fscd147",
        choices=["fscd147", "fscd_lvis"],
        help="评估数据集: fscd147 / fscd_lvis",
    )
    P.add_argument(
        "--dataset_dir", type=str, required=True,
        help="数据集根目录",
    )
    P.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
        help="数据集划分: val / test",
    )
    P.add_argument(
        "--model_size", type=str, default="base",
        choices=["base", "large"],
        help="模型规模: base (ViT-S) / large (ViT-B)",
    )
    P.add_argument(
        "--threshold", type=float, default=0.5,
        help="Hard count 的置信度阈值 (仅影响 hard_mae/hard_rmse)",
    )
    P.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备",
    )
    P.add_argument(
        "--output_json", type=str, default=None,
        help="保存评估结果到 JSON 文件",
    )

    return P.parse_args()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    Args = ParseArgs()

    print()
    print("  ═══════════════════════════════════════════════")
    print("  FSCD Benchmark Evaluation")
    print("  ═══════════════════════════════════════════════")
    print()
    print(f"  Checkpoint:  {Args.checkpoint}")
    print(f"  Dataset:     {Args.dataset} ({Args.split})")
    print(f"  Model size:  {Args.model_size}")
    print(f"  Device:      {Args.device}")
    print()

    # 加载模型
    Model, Postprocess, Resolution = LoadModel(
        CheckpointPath=Args.checkpoint,
        ModelSize=Args.model_size,
        Device=Args.device,
    )

    # 加载数据集
    if Args.dataset == "fscd147":
        Samples, CocoGt = LoadFscd147(Args.dataset_dir, Args.split)
    elif Args.dataset == "fscd_lvis":
        Samples, CocoGt = LoadFscdLvis(Args.dataset_dir, Args.split)
    else:
        raise ValueError(f"Unknown dataset: {Args.dataset}")

    if not Samples:
        print("  ERROR: No samples loaded. Check dataset path and split.")
        sys.exit(1)

    # 评估
    StartTime = time.time()

    Stats = Evaluate(
        Model=Model,
        Postprocess=Postprocess,
        Samples=Samples,
        CocoGt=CocoGt,
        Resolution=Resolution,
        ThresholdForCount=Args.threshold,
        Device=Args.device,
    )

    ElapsedTime = time.time() - StartTime

    # 打印结果
    DatasetLabel = "FSCD-147" if Args.dataset == "fscd147" else "FSCD-LVIS"
    PrintResults(Stats, DatasetLabel, Args.split)
    print(f"  Total time: {ElapsedTime:.1f}s ({ElapsedTime / len(Samples):.2f}s/image)")

    # 保存 JSON
    if Args.output_json:
        Stats["dataset"] = Args.dataset
        Stats["split"] = Args.split
        Stats["checkpoint"] = Args.checkpoint
        Stats["model_size"] = Args.model_size
        Stats["elapsed_seconds"] = ElapsedTime
        with open(Args.output_json, "w") as Fh:
            json.dump(Stats, Fh, indent=2)
        print(f"  Results saved to: {Args.output_json}")

    print()
