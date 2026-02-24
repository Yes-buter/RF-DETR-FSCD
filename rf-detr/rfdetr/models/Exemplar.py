# ------------------------------------------------------------------------
# RF-DETR — FSCD Extension
# Exemplar Prototype Extraction and Feature Conditioning Modules
# ------------------------------------------------------------------------

"""
Exemplar 模块: 从少量 exemplar bbox 提取 prototype 特征，
并将其注入到多尺度特征图中以实现条件化检测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import List, Optional, Tuple


class ExemplarPrototypeExtractor(nn.Module):
    """从 exemplar bounding boxes 中提取 prototype 特征。

    通过 RoI-Align 从特征图裁剪 exemplar 区域，
    然后使用 attention pooling 聚合多个 exemplar 为统一的 prototype 表示。

    Args:
        HiddenDim: 特征通道维度
        RoiSize: RoI-Align 输出的空间分辨率
        PrototypeDim: prototype 最终输出维度
        NumHeads: attention pooling 的注意力头数
    """

    def __init__(
        self,
        HiddenDim: int = 256,
        RoiSize: int = 7,
        PrototypeDim: int = 256,
        NumHeads: int = 8,
    ) -> None:
        super().__init__()
        self.HiddenDim = HiddenDim
        self.RoiSize = RoiSize
        self.PrototypeDim = PrototypeDim

        # 将 RoI 区域特征聚合为向量的 attention pooling
        self.AttnPool = nn.MultiheadAttention(
            embed_dim=HiddenDim,
            num_heads=NumHeads,
            batch_first=True,
        )
        # 用于 attention pooling 的可学习 query token
        self.PoolQuery = nn.Parameter(torch.randn(1, 1, HiddenDim))
        nn.init.xavier_uniform_(self.PoolQuery)

        # 形状 prototype: 从 bbox 宽高编码几何信息
        self.ShapeMlp = nn.Sequential(
            nn.Linear(2, HiddenDim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(HiddenDim // 2, PrototypeDim),
        )

        # 融合外观 + 形状 prototype
        self.FusionMlp = nn.Sequential(
            nn.Linear(PrototypeDim * 2, PrototypeDim),
            nn.ReLU(inplace=True),
            nn.Linear(PrototypeDim, PrototypeDim),
        )

        # 多 exemplar 融合: attention over exemplars
        self.ExemplarAttn = nn.MultiheadAttention(
            embed_dim=PrototypeDim,
            num_heads=NumHeads,
            batch_first=True,
        )
        self.ExemplarQuery = nn.Parameter(torch.randn(1, 1, PrototypeDim))
        nn.init.xavier_uniform_(self.ExemplarQuery)

        self.Norm = nn.LayerNorm(PrototypeDim)

    def _ExtractRoiFeatures(
        self,
        FeatureMap: torch.Tensor,
        ExemplarBoxes: torch.Tensor,
    ) -> torch.Tensor:
        """使用 RoI-Align 从特征图中提取 exemplar 区域特征。

        Args:
            FeatureMap: [B, C, H, W] 骨干网络输出的特征图（取最高分辨率的一层）
            ExemplarBoxes: [B, NumExemplars, 4] 归一化 xyxy 格式的 exemplar bbox

        Returns:
            RoiFeats: [B, NumExemplars, C, RoiSize, RoiSize]
        """
        B, NumExemplars, _ = ExemplarBoxes.shape
        _, C, H, W = FeatureMap.shape

        # 将归一化坐标转换为绝对坐标（RoI-Align 需要）
        ScaleFactors = torch.tensor(
            [W, H, W, H], dtype=ExemplarBoxes.dtype, device=ExemplarBoxes.device
        )
        AbsBoxes = ExemplarBoxes * ScaleFactors  # [B, NumExemplars, 4]

        # 构建 roi_align 需要的 (BatchIdx, x1, y1, x2, y2) 格式
        BatchIdxs = torch.arange(B, device=FeatureMap.device)
        BatchIdxs = BatchIdxs[:, None].expand(B, NumExemplars).reshape(-1, 1).float()
        FlatBoxes = AbsBoxes.reshape(-1, 4)
        Rois = torch.cat([BatchIdxs, FlatBoxes], dim=-1)  # [B*NumExemplars, 5]

        RoiFeats = roi_align(
            FeatureMap, Rois, output_size=self.RoiSize, aligned=True
        )  # [B*NumExemplars, C, RoiSize, RoiSize]

        return RoiFeats.reshape(B, NumExemplars, C, self.RoiSize, self.RoiSize)

    def _AttentionPooling(self, RoiFeats: torch.Tensor) -> torch.Tensor:
        """将 RoI 特征图通过 attention pooling 聚合为向量。

        Args:
            RoiFeats: [B*NumExemplars, C, RoiSize, RoiSize]

        Returns:
            Pooled: [B*NumExemplars, C]
        """
        N, C, H, W = RoiFeats.shape
        Tokens = RoiFeats.flatten(2).permute(0, 2, 1)  # [N, H*W, C]
        Query = self.PoolQuery.expand(N, -1, -1)  # [N, 1, C]

        Pooled, _ = self.AttnPool(Query, Tokens, Tokens)  # [N, 1, C]
        return Pooled.squeeze(1)  # [N, C]

    def forward(
        self,
        Features: List[torch.Tensor],
        ExemplarBoxes: torch.Tensor,
    ) -> torch.Tensor:
        """提取统一的 prototype 特征。

        Args:
            Features: 多尺度特征图列表，每个 [B, C, Hi, Wi]。取第 0 层(最高分辨率)
            ExemplarBoxes: [B, NumExemplars, 4] 归一化 xyxy 格式

        Returns:
            Prototype: [B, PrototypeDim] 融合后的 prototype 特征
        """
        # 使用最高分辨率特征图提取 RoI 特征
        FeatureMap = Features[0]  # [B, C, H, W]
        B, NumExemplars, _ = ExemplarBoxes.shape

        # 1. 提取 RoI 外观特征
        RoiFeats = self._ExtractRoiFeatures(
            FeatureMap, ExemplarBoxes
        )  # [B, NumExemplars, C, RoiSize, RoiSize]

        RoiFlat = RoiFeats.reshape(
            B * NumExemplars, self.HiddenDim, self.RoiSize, self.RoiSize
        )
        AppearanceFeats = self._AttentionPooling(RoiFlat)  # [B*NumExemplars, C]
        AppearanceFeats = AppearanceFeats.reshape(
            B, NumExemplars, self.PrototypeDim
        )  # [B, NumExemplars, C]

        # 2. 提取形状 prototype (从 bbox 宽高)
        BoxWH = ExemplarBoxes[..., 2:] - ExemplarBoxes[..., :2]  # [B, NumExemplars, 2]
        ShapeFeats = self.ShapeMlp(BoxWH)  # [B, NumExemplars, PrototypeDim]

        # 3. 融合外观 + 形状
        CombinedFeats = torch.cat(
            [AppearanceFeats, ShapeFeats], dim=-1
        )  # [B, NumExemplars, 2*PrototypeDim]
        FusedFeats = self.FusionMlp(CombinedFeats)  # [B, NumExemplars, PrototypeDim]

        # 4. 多 exemplar 聚合: attention pooling over exemplars
        Query = self.ExemplarQuery.expand(B, -1, -1)  # [B, 1, PrototypeDim]
        Prototype, _ = self.ExemplarAttn(
            Query, FusedFeats, FusedFeats
        )  # [B, 1, PrototypeDim]
        Prototype = self.Norm(Prototype.squeeze(1))  # [B, PrototypeDim]

        return Prototype


class ExemplarConditioningModule(nn.Module):
    """将 prototype 信息注入到多尺度特征图中。

    通过轻量级 cross-attention，让特征图的每个空间位置
    attend to exemplar prototype，实现特征的条件化。

    Args:
        HiddenDim: 特征通道维度
        NumHeads: cross-attention 头数
        NumLevels: 多尺度层级数
    """

    def __init__(
        self,
        HiddenDim: int = 256,
        NumHeads: int = 8,
        NumLevels: int = 3,
    ) -> None:
        super().__init__()
        self.HiddenDim = HiddenDim
        self.NumLevels = NumLevels

        # 每层一个独立的 cross-attention + FFN
        self.CrossAttnLayers = nn.ModuleList()
        self.NormLayers = nn.ModuleList()
        self.FfnLayers = nn.ModuleList()
        self.FfnNormLayers = nn.ModuleList()

        for _ in range(NumLevels):
            self.CrossAttnLayers.append(
                nn.MultiheadAttention(
                    embed_dim=HiddenDim,
                    num_heads=NumHeads,
                    batch_first=True,
                )
            )
            self.NormLayers.append(nn.LayerNorm(HiddenDim))
            self.FfnLayers.append(
                nn.Sequential(
                    nn.Linear(HiddenDim, HiddenDim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(HiddenDim * 4, HiddenDim),
                )
            )
            self.FfnNormLayers.append(nn.LayerNorm(HiddenDim))

        # Prototype 投影（统一维度）
        self.ProtoProj = nn.Linear(HiddenDim, HiddenDim)

    def forward(
        self,
        Features: List[torch.Tensor],
        Prototype: torch.Tensor,
    ) -> List[torch.Tensor]:
        """将 prototype 条件化注入到每层特征图中。

        Args:
            Features: 多尺度特征图列表, 每个 [B, C, H, W]
            Prototype: [B, C] prototype 特征

        Returns:
            ConditionedFeatures: 与输入同形状的条件化特征图列表
        """
        ProtoKV = self.ProtoProj(Prototype).unsqueeze(1)  # [B, 1, C]

        ConditionedFeatures = []
        for LvlIdx, Feat in enumerate(Features):
            if LvlIdx >= self.NumLevels:
                # 超出配置层数的直接透传
                ConditionedFeatures.append(Feat)
                continue

            B, C, H, W = Feat.shape
            # Flatten 空间维度作为 query sequence
            FeatFlat = Feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

            # Cross-Attention: 每个空间位置 attend to prototype
            AttnOut, _ = self.CrossAttnLayers[LvlIdx](
                FeatFlat, ProtoKV, ProtoKV
            )  # [B, H*W, C]
            FeatFlat = self.NormLayers[LvlIdx](
                FeatFlat + AttnOut
            )  # Residual + LayerNorm

            # FFN
            FfnOut = self.FfnLayers[LvlIdx](FeatFlat)
            FeatFlat = self.FfnNormLayers[LvlIdx](FeatFlat + FfnOut)

            # 恢复空间形状
            CondFeat = FeatFlat.permute(0, 2, 1).reshape(B, C, H, W)
            ConditionedFeatures.append(CondFeat)

        return ConditionedFeatures
