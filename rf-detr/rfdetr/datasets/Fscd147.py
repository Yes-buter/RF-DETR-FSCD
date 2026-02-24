# ------------------------------------------------------------------------
# RF-DETR — FSCD Extension
# FSCD-147 数据集加载器
# ------------------------------------------------------------------------

"""
FSCD-147 数据集:
  - 6135 张图像 / 147 个类别
  - 每张图 3 个 exemplar bounding box (box_examples_coordinates)
  - 点标注 (points) 标记所有目标中心
  - val/test 集额外提供全量 COCO 格式 bounding box

数据目录结构:
  FSCD147/
  ├── images_384_VarV2/           # 图像
  ├── gt_density_map_adaptive_512_512_object_VarV2/  # 密度图 (.npy)
  ├── annotation_FSC147_384.json  # 注释 (exemplar boxes + points)
  ├── Train_Test_Val_FSC_147.json # 数据集划分
  ├── instances_val.json          # COCO 格式全量 bbox (val)
  └── instances_test.json         # COCO 格式全量 bbox (test)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

import rfdetr.datasets.transforms as T


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _ExemplarCoordsToXyxy(
    BoxCoords: List[List[List[float]]],
    RatioH: float,
    RatioW: float,
) -> torch.Tensor:
    """把 FSC-147 的 4 角点格式转为 [x1,y1,x2,y2]。

    FSC-147 box_examples_coordinates 格式:
        [[y1,x1], [y1,x2], [y2,x2], [y2,x1]]  (4 个角点, 注意 y 在前)
    这些坐标是原图尺度，需用 ratio_h / ratio_w 缩放到 384 版本。

    Returns:
        Tensor [N, 4] (x1, y1, x2, y2) 归一化到 [0, 1] 相对坐标时需除以 H/W。
    """
    Boxes = []
    for Corners in BoxCoords:
        # Corners: [[y1,x1], [y1,x2], [y2,x2], [y2,x1]]
        Ys = [C[0] for C in Corners]
        Xs = [C[1] for C in Corners]
        X1 = min(Xs) * RatioW
        Y1 = min(Ys) * RatioH
        X2 = max(Xs) * RatioW
        Y2 = max(Ys) * RatioH
        Boxes.append([X1, Y1, X2, Y2])
    return torch.tensor(Boxes, dtype=torch.float32)


def _LoadCocoBoxes(
    AnnFile: str,
) -> Dict[str, List[List[float]]]:
    """加载 COCO 格式 instances JSON，返回 {filename: [[x1,y1,x2,y2], ...]}。"""
    with open(AnnFile, "r") as F:
        Data = json.load(F)

    ImgIdToName = {Img["id"]: Img["file_name"] for Img in Data["images"]}
    ImgIdToBoxes: Dict[int, List[List[float]]] = {}
    for Ann in Data["annotations"]:
        ImgId = Ann["image_id"]
        X, Y, W, H = Ann["bbox"]
        if ImgId not in ImgIdToBoxes:
            ImgIdToBoxes[ImgId] = []
        ImgIdToBoxes[ImgId].append([X, Y, X + W, Y + H])

    Result = {}
    for ImgId, Boxes in ImgIdToBoxes.items():
        Result[ImgIdToName[ImgId]] = Boxes
    return Result


# ---------------------------------------------------------------------------
# 数据集类
# ---------------------------------------------------------------------------

class Fscd147Dataset(data.Dataset):
    """FSCD-147 数据集。

    输出 dict:
        - image: Tensor [3, H, W]
        - exemplar_boxes: Tensor [NumExemplars, 4] (xyxy, 像素坐标)
        - target:
            - boxes: Tensor [N, 4] (xyxy, 像素坐标, 来自 COCO 注释或点标注)
            - labels: Tensor [N] (全为 0, class-agnostic)
            - points: Tensor [M, 2] (所有目标中心 xy)
            - count: int (目标总数)
            - image_id: Tensor [1]
            - orig_size: Tensor [2]
            - size: Tensor [2]
    """

    def __init__(
        self,
        DataRoot: str,
        Split: Literal["train", "val", "test"] = "train",
        NumExemplars: int = 3,
        Transforms: Optional[object] = None,
        UseDensityMap: bool = False,
    ) -> None:
        """
        Args:
            DataRoot: FSCD-147 数据集根目录路径。
            Split: 数据集划分 (train / val / test)。
            NumExemplars: 使用的 exemplar 数量（默认 3）。
            Transforms: 数据增强 transforms。
            UseDensityMap: 是否加载密度图（默认 False）。
        """
        super().__init__()
        self._Root = Path(DataRoot)
        self._Split = Split
        self._NumExemplars = NumExemplars
        self._Transforms = Transforms
        self._UseDensityMap = UseDensityMap

        # 加载数据集划分
        SplitPath = self._Root / "Train_Test_Val_FSC_147.json"
        with open(SplitPath, "r") as F:
            SplitData = json.load(F)
        self._FileNames: List[str] = SplitData[Split]

        # 加载注释 (exemplar boxes + points)
        AnnPath = self._Root / "annotation_FSC147_384.json"
        with open(AnnPath, "r") as F:
            self._Annotations: Dict = json.load(F)

        # 加载 COCO 格式全量 bbox (仅 val/test)
        self._CocoBoxes: Optional[Dict[str, List[List[float]]]] = None
        self._CocoAnnFile: Optional[str] = None
        self._FileNameToCocoId: Dict[str, int] = {}
        CocoAnnFile = self._Root / f"instances_{Split}.json"
        if CocoAnnFile.exists():
            self._CocoAnnFile = str(CocoAnnFile)
            self._CocoBoxes = _LoadCocoBoxes(self._CocoAnnFile)
            # 建立 filename → COCO image_id 映射
            with open(self._CocoAnnFile, "r") as F:
                CocoData = json.load(F)
            for ImgInfo in CocoData["images"]:
                self._FileNameToCocoId[ImgInfo["file_name"]] = ImgInfo["id"]

        # 图像与密度图目录
        self._ImgDir = self._Root / "images_384_VarV2"
        self._DensityDir = self._Root / "gt_density_map_adaptive_512_512_object_VarV2"

    def GetCocoApi(self):
        """返回 pycocotools COCO 对象 (仅 val/test 有效)。"""
        if self._CocoAnnFile is None:
            return None
        from pycocotools.coco import COCO
        import contextlib, os
        with open(os.devnull, "w") as Devnull:
            with contextlib.redirect_stdout(Devnull):
                return COCO(self._CocoAnnFile)

    def __len__(self) -> int:
        return len(self._FileNames)

    def __getitem__(self, Idx: int) -> Tuple[torch.Tensor, dict]:
        FileName = self._FileNames[Idx]
        Ann = self._Annotations[FileName]

        # ---- 加载图像 ----
        ImgPath = self._ImgDir / FileName
        Img = Image.open(ImgPath).convert("RGB")
        W, H = Img.size

        # ---- Exemplar boxes ----
        RatioH = Ann.get("ratio_h", 1.0)
        RatioW = Ann.get("ratio_w", 1.0)
        AllExemplarCoords = Ann["box_examples_coordinates"]
        # 截取前 NumExemplars 个
        ExemplarCoords = AllExemplarCoords[: self._NumExemplars]
        ExemplarBoxes = _ExemplarCoordsToXyxy(ExemplarCoords, RatioH, RatioW)
        # clamp 到图像范围
        ExemplarBoxes[:, 0::2].clamp_(min=0, max=W)
        ExemplarBoxes[:, 1::2].clamp_(min=0, max=H)

        # ---- 目标 boxes (来自 COCO 或从 points 生成伪 box) ----
        Points = torch.tensor(Ann["points"], dtype=torch.float32)  # [M, 2] (x, y)
        GtCount = len(Points)

        if self._CocoBoxes is not None and FileName in self._CocoBoxes:
            # val/test: 使用全量 COCO bbox
            RawBoxes = torch.tensor(
                self._CocoBoxes[FileName], dtype=torch.float32
            )  # [N, 4] xyxy
        else:
            # train: 没有全量 bbox，用 exemplar 估算平均尺寸生成伪 bbox
            ExW = (ExemplarBoxes[:, 2] - ExemplarBoxes[:, 0]).mean()
            ExH = (ExemplarBoxes[:, 3] - ExemplarBoxes[:, 1]).mean()
            HalfW, HalfH = ExW / 2.0, ExH / 2.0
            if len(Points) > 0:
                X1 = (Points[:, 0] - HalfW).clamp(min=0)
                Y1 = (Points[:, 1] - HalfH).clamp(min=0)
                X2 = (Points[:, 0] + HalfW).clamp(max=W)
                Y2 = (Points[:, 1] + HalfH).clamp(max=H)
                RawBoxes = torch.stack([X1, Y1, X2, Y2], dim=1)
            else:
                RawBoxes = torch.zeros(0, 4, dtype=torch.float32)

        # 过滤退化 box
        if RawBoxes.numel() > 0:
            Keep = (RawBoxes[:, 2] > RawBoxes[:, 0]) & (
                RawBoxes[:, 3] > RawBoxes[:, 1]
            )
            RawBoxes = RawBoxes[Keep]

        Labels = torch.zeros(len(RawBoxes), dtype=torch.int64)

        # ---- 组装 target ----
        Target = {
            "boxes": RawBoxes,
            "labels": Labels,
            "points": Points,
            "count": torch.tensor([GtCount], dtype=torch.int64),
            "exemplar_boxes": ExemplarBoxes,
            "image_id": torch.tensor([self._FileNameToCocoId.get(FileName, Idx)]),
            "orig_size": torch.tensor([H, W]),
            "size": torch.tensor([H, W]),
        }

        # ---- 密度图 (可选) ----
        if self._UseDensityMap:
            DensityPath = self._DensityDir / (
                os.path.splitext(FileName)[0] + ".npy"
            )
            if DensityPath.exists():
                DensityMap = np.load(str(DensityPath))
                Target["density_map"] = torch.from_numpy(DensityMap).float()
            else:
                Target["density_map"] = torch.zeros(H, W, dtype=torch.float32)

        # ---- Transforms ----
        if self._Transforms is not None:
            Img, Target = self._Transforms(Img, Target)
            # transforms 内部已同步处理 boxes/exemplar_boxes/points 的缩放,
            # 只需更新 size
            if isinstance(Img, torch.Tensor):
                NewH, NewW = Img.shape[-2], Img.shape[-1]
                Target["size"] = torch.tensor([NewH, NewW])

        return Img, Target


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def MakeFscdTransforms(
    ImageSet: str,
    Resolution: int = 560,
) -> T.Compose:
    """FSCD-147 数据增强。

    使用 SquareResize 确保输出为方形 (NxN), 满足 DINOv2 backbone
    block_size=56 的整除要求。

    Train: ColorJitter + RandomHorizontalFlip + RandomSizeCrop + SquareResize
    Val/Test: SquareResize + ToTensor + Normalize
    """
    Normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if ImageSet == "train":
        return T.Compose([
            # 颜色抖动 — 不影响 bbox/points, 仅改变像素值
            T.PhotometricDistort(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.5,
            ),
            T.RandomHorizontalFlip(),
            # 50% 概率进行随机裁剪 (保留 60%~100% 的区域)
            T.RandomSelect(
                T.SquareResize([Resolution]),
                T.Compose([
                    T.RandomSizeCrop(
                        int(Resolution * 0.6),
                        int(Resolution * 1.0),
                    ),
                    T.SquareResize([Resolution]),
                ]),
                p=0.5,
            ),
            Normalize,
        ])
    else:
        return T.Compose([
            T.SquareResize([Resolution]),
            Normalize,
        ])


# ---------------------------------------------------------------------------
# 构建函数
# ---------------------------------------------------------------------------

def BuildFscd147(
    ImageSet: str,
    DataRoot: str,
    Resolution: int = 560,
    NumExemplars: int = 3,
    UseDensityMap: bool = False,
) -> Fscd147Dataset:
    """构建 FSCD-147 数据集实例。

    Args:
        ImageSet: "train" / "val" / "test"
        DataRoot: FSCD-147 数据集根目录
        Resolution: 目标图像分辨率
        NumExemplars: exemplar 数量
        UseDensityMap: 是否加载密度图
    """
    Transforms = MakeFscdTransforms(ImageSet, Resolution)
    return Fscd147Dataset(
        DataRoot=DataRoot,
        Split=ImageSet,
        NumExemplars=NumExemplars,
        Transforms=Transforms,
        UseDensityMap=UseDensityMap,
    )
