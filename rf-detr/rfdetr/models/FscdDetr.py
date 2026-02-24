# ------------------------------------------------------------------------
# RF-DETR — FSCD Extension
# Few-Shot Counting and Detection Model (基于 LWDETR)
# ------------------------------------------------------------------------

"""
FSCD 版本的 LWDETR 模型。
复用 RF-DETR 的 backbone、transformer、bbox_embed，
替换分类头为 class-agnostic objectness + prototype 相似度。
"""

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.Exemplar import (
    ExemplarConditioningModule,
    ExemplarPrototypeExtractor,
)
from rfdetr.models.lwdetr import (
    MLP,
    PostProcess,
    SetCriterion,
    build_model,
    sigmoid_focal_loss,
)
from rfdetr.models.matcher import build_matcher
from rfdetr.util.misc import (
    NestedTensor,
    get_world_size,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


class FscdLWDETR(nn.Module):
    """Few-Shot Counting and Detection 模型，基于 RF-DETR 架构。

    在 LWDETR 基础上新增:
    1. ExemplarPrototypeExtractor: 从 exemplar bbox 提取 prototype
    2. ExemplarConditioningModule: 将 prototype 注入多尺度特征
    3. Class-agnostic objectness head: 替换 N-class 分类头

    Args:
        Backbone: 骨干网络（Joiner: DINOv2 + Projector + PositionEmbedding）
        TransformerModule: Transformer decoder
        SegmentationHead: 可选的分割头（FSCD 暂不使用）
        NumQueries: 检测 query 数量
        AuxLoss: 是否使用辅助损失
        GroupDetr: Group DETR 分组数
        TwoStage: 是否使用 Two-Stage
        LiteRefpointRefine: 参考点精炼模式
        BboxReparam: bbox 参数化方式
        ExemplarRoiSize: RoI Align 输出尺寸
        PrototypeDim: Prototype 特征维度
        NumExemplars: Exemplar 数量
    """

    def __init__(
        self,
        Backbone: nn.Module,
        TransformerModule: nn.Module,
        SegmentationHead: Optional[nn.Module],
        NumQueries: int = 500,
        AuxLoss: bool = True,
        GroupDetr: int = 13,
        TwoStage: bool = True,
        LiteRefpointRefine: bool = False,
        BboxReparam: bool = True,
        ExemplarRoiSize: int = 7,
        PrototypeDim: int = 256,
        NumExemplars: int = 3,
        PriorProb: float = 0.01,
    ) -> None:
        super().__init__()
        self.NumQueries = NumQueries
        self.TransformerModule = TransformerModule
        HiddenDim = TransformerModule.d_model

        # ========== 从原 LWDETR 复用的组件 ==========
        # class-agnostic: 只有 1 类 (objectness)
        self.ObjectnessEmbed = nn.Linear(HiddenDim, 1)
        self.BboxEmbed = MLP(HiddenDim, HiddenDim, 4, 3)
        self.SegmentationHead = SegmentationHead

        QueryDim = 4
        self.RefpointEmbed = nn.Embedding(NumQueries * GroupDetr, QueryDim)
        self.QueryFeat = nn.Embedding(NumQueries * GroupDetr, HiddenDim)
        nn.init.constant_(self.RefpointEmbed.weight.data, 0)

        self.Backbone = Backbone
        self.AuxLoss = AuxLoss
        self.GroupDetr = GroupDetr

        # Iterative refinement
        self.LiteRefpointRefine = LiteRefpointRefine
        if not self.LiteRefpointRefine:
            self.TransformerModule.decoder.bbox_embed = self.BboxEmbed
        else:
            self.TransformerModule.decoder.bbox_embed = None

        self.BboxReparam = BboxReparam

        # 初始化 objectness bias（focal loss 先验）
        PriorProb = float(PriorProb)
        PriorProb = min(max(PriorProb, 1e-6), 1 - 1e-6)
        BiasValue = -math.log((1 - PriorProb) / PriorProb)
        self.ObjectnessEmbed.bias.data = torch.ones(1) * BiasValue

        # 初始化 bbox embed
        nn.init.constant_(self.BboxEmbed.layers[-1].weight.data, 0)
        nn.init.constant_(self.BboxEmbed.layers[-1].bias.data, 0)

        # Two-Stage
        self.TwoStage = TwoStage
        if self.TwoStage:
            self.TransformerModule.enc_out_bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.BboxEmbed) for _ in range(GroupDetr)]
            )
            # enc stage 也用 objectness (1类)
            self.TransformerModule.enc_out_class_embed = nn.ModuleList(
                [copy.deepcopy(self.ObjectnessEmbed) for _ in range(GroupDetr)]
            )

        # ========== FSCD 新增组件 ==========
        self.ExemplarExtractor = ExemplarPrototypeExtractor(
            HiddenDim=HiddenDim,
            RoiSize=ExemplarRoiSize,
            PrototypeDim=PrototypeDim,
        )

        NumLevels = TransformerModule.num_feature_levels
        self.ExemplarConditioning = ExemplarConditioningModule(
            HiddenDim=HiddenDim,
            NumLevels=NumLevels,
        )

        # Prototype-enhanced objectness: 余弦相似度增强
        self.ProtoProjection = nn.Linear(HiddenDim, PrototypeDim)

        self._Export = False

    def forward(
        self,
        Samples: NestedTensor,
        ExemplarBoxes: torch.Tensor,
        Targets: Optional[List[dict]] = None,
    ) -> dict:
        """前向传播。

        Args:
            Samples: NestedTensor (images + masks)
            ExemplarBoxes: [B, NumExemplars, 4] 归一化 xyxy 格式
            Targets: 训练时的标注列表

        Returns:
            输出字典, 包含 pred_logits [B, NQ, 1] 和 pred_boxes [B, NQ, 4]
        """
        if isinstance(Samples, (list, torch.Tensor)):
            Samples = nested_tensor_from_tensor_list(Samples)

        # ① Backbone 提取特征
        Features, Poss = self.Backbone(Samples)

        Srcs = []
        Masks = []
        for _, Feat in enumerate(Features):
            Src, Mask = Feat.decompose()
            Srcs.append(Src)
            Masks.append(Mask)

        # ② Exemplar prototype 提取
        Prototype = self.ExemplarExtractor(Srcs, ExemplarBoxes)  # [B, PrototypeDim]

        # ③ 条件化: 将 prototype 注入多尺度特征图
        CondSrcs = self.ExemplarConditioning(Srcs, Prototype)

        # ④ Transformer decoder
        if self.training:
            RefpointEmbedWeight = self.RefpointEmbed.weight
            QueryFeatWeight = self.QueryFeat.weight
        else:
            RefpointEmbedWeight = self.RefpointEmbed.weight[: self.NumQueries]
            QueryFeatWeight = self.QueryFeat.weight[: self.NumQueries]

        Bs = CondSrcs[0].shape[0]
        Hs, RefUnsigmoid, HsEnc, RefEnc = self.TransformerModule(
            CondSrcs, Masks, Poss, RefpointEmbedWeight, QueryFeatWeight
        )

        if Hs is not None:
            # BBox regression
            if self.BboxReparam:
                OutputsCoordDelta = self.BboxEmbed(Hs)
                OutputsCoordCxcy = (
                    OutputsCoordDelta[..., :2] * RefUnsigmoid[..., 2:]
                    + RefUnsigmoid[..., :2]
                )
                OutputsCoordWh = (
                    OutputsCoordDelta[..., 2:].exp() * RefUnsigmoid[..., 2:]
                )
                OutputsCoord = torch.cat(
                    [OutputsCoordCxcy, OutputsCoordWh], dim=-1
                )
            else:
                OutputsCoord = (
                    self.BboxEmbed(Hs) + RefUnsigmoid
                ).sigmoid()

            # ⑤ Class-agnostic objectness (线性头 + prototype 相似度增强)
            OutputsObjectness = self.ObjectnessEmbed(Hs)  # [NumLayers, B, NQ, 1]

            # Prototype similarity boost
            QueryProj = self.ProtoProjection(Hs)  # [NumLayers, B, NQ, PrototypeDim]
            ProtoExpanded = Prototype.unsqueeze(0).unsqueeze(2)  # [1, B, 1, PrototypeDim]
            CosineSim = F.cosine_similarity(
                QueryProj, ProtoExpanded, dim=-1
            ).unsqueeze(-1)  # [NumLayers, B, NQ, 1]

            OutputsClass = OutputsObjectness + CosineSim  # 融合

            Out = {
                "pred_logits": OutputsClass[-1],
                "pred_boxes": OutputsCoord[-1],
            }
            if self.AuxLoss:
                Out["aux_outputs"] = self._SetAuxLoss(OutputsClass, OutputsCoord)

        # Two-Stage encoder outputs
        if self.TwoStage:
            GroupDetr = self.GroupDetr if self.training else 1
            HsEncList = HsEnc.chunk(GroupDetr, dim=1)
            ClsEnc = []
            for GIdx in range(GroupDetr):
                ClsEncGidx = self.TransformerModule.enc_out_class_embed[GIdx](
                    HsEncList[GIdx]
                )
                ClsEnc.append(ClsEncGidx)
            ClsEnc = torch.cat(ClsEnc, dim=1)

            if Hs is not None:
                Out["enc_outputs"] = {
                    "pred_logits": ClsEnc,
                    "pred_boxes": RefEnc,
                }
            else:
                Out = {"pred_logits": ClsEnc, "pred_boxes": RefEnc}

        return Out

    @torch.jit.unused
    def _SetAuxLoss(self, OutputsClass, OutputsCoord):
        """辅助损失的输出格式化。"""
        return [
            {"pred_logits": A, "pred_boxes": B}
            for A, B in zip(OutputsClass[:-1], OutputsCoord[:-1])
        ]


class FscdSetCriterion(SetCriterion):
    """FSCD 版本的损失函数。

    主要差异:
    1. loss_labels → class-agnostic binary focal loss（1类 objectness）
    2. 新增 loss_count: 预测数量与 GT 数量的 L1 loss
    3. loss_boxes / loss_giou 完全复用父类
    """

    def __init__(
        self,
        Matcher: nn.Module,
        WeightDict: dict,
        FocalAlpha: float,
        FocalGamma: float,
        Losses: List[str],
        GroupDetr: int = 1,
        SumGroupLosses: bool = False,
        CountLossCoef: float = 0.5,
        UseTopkCountLoss: bool = True,
        CountTopkFactor: float = 3.0,
        CountTopkMin: int = 1,
        CountTopkMax: int = 150,
    ) -> None:
        # num_classes=1 for class-agnostic
        super().__init__(
            num_classes=1,
            matcher=Matcher,
            weight_dict=WeightDict,
            focal_alpha=FocalAlpha,
            losses=Losses,
            group_detr=GroupDetr,
            sum_group_losses=SumGroupLosses,
        )
        self.CountLossCoef = CountLossCoef
        self.FocalGamma = FocalGamma
        self.UseTopkCountLoss = UseTopkCountLoss
        self.CountTopkFactor = CountTopkFactor
        self.CountTopkMin = CountTopkMin
        self.CountTopkMax = CountTopkMax

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Class-agnostic binary focal loss。

        所有匹配到的 query 标签为 0（唯一的 objectness 类），
        未匹配的为 background。
        """
        src_logits = outputs["pred_logits"]  # [B, NQ, 1]

        idx = self._get_src_permutation_idx(indices)

        # 所有目标都是类别 0 (objectness)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,  # = 1, 即 background
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = 0  # 匹配到的 → class 0

        # one-hot 编码
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]  # [B, NQ, 1]

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes,
            alpha=self.focal_alpha,
            gamma=self.FocalGamma,
        ) * src_logits.shape[1]

        losses = {"loss_ce": loss_ce}

        if log:
            # 计数误差 (用于 logging)
            with torch.no_grad():
                MatchedPerImage = torch.tensor(
                    [len(src_idx) for src_idx, _ in indices],
                    dtype=torch.float,
                    device=src_logits.device,
                ).mean()
                GroupDetrNorm = float(max(1, getattr(self, "group_detr", 1)))
                MatchedPerImage = MatchedPerImage / GroupDetrNorm
                PredCount = (src_logits.sigmoid() > 0.5).sum(dim=1).float()
                GtCount = torch.tensor(
                    [len(t["labels"]) for t in targets],
                    dtype=torch.float,
                    device=src_logits.device,
                )
                losses["matched_queries_per_image"] = MatchedPerImage
                losses["count_error"] = F.l1_loss(
                    PredCount.squeeze(-1), GtCount
                )

        return losses

    def loss_count(self, outputs, targets, indices, num_boxes, **kwargs):
        """直接的计数损失: 预测激活数量与 GT 目标数量的 L1 loss。"""
        src_logits = outputs["pred_logits"]  # [B, NQ, 1]
        pred_scores = src_logits.sigmoid().squeeze(-1)  # [B, NQ]
        GtCount = torch.tensor(
            [len(t["labels"]) for t in targets],
            dtype=torch.float,
            device=src_logits.device,
        )

        if self.UseTopkCountLoss:
            num_queries = pred_scores.shape[1]
            topk_max = min(self.CountTopkMax, num_queries)
            topk_min = min(self.CountTopkMin, topk_max)
            topk_per_image = torch.clamp(
                (GtCount * self.CountTopkFactor).long(),
                min=topk_min,
                max=topk_max,
            )
            max_k_in_batch = int(topk_per_image.max().item())
            topk_scores, _ = torch.topk(pred_scores, k=max_k_in_batch, dim=1)
            topk_mask = (
                torch.arange(max_k_in_batch, device=pred_scores.device)
                .unsqueeze(0)
                < topk_per_image.unsqueeze(1)
            )
            SoftCount = (topk_scores * topk_mask).sum(dim=1)
        else:
            SoftCount = pred_scores.sum(dim=1)

        LossCount = F.l1_loss(SoftCount, GtCount)
        return {"loss_count": LossCount}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "count": self.loss_count,
        }
        assert loss in loss_map, f"Unsupported loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)


class FscdPostProcess(nn.Module):
    """FSCD 版本的后处理: class-agnostic (1类 objectness)。"""

    def __init__(self, NumSelect: int = 500) -> None:
        super().__init__()
        self.NumSelect = NumSelect

    @torch.no_grad()
    def forward(
        self,
        Outputs: dict,
        TargetSizes: torch.Tensor,
    ) -> list:
        """后处理: 取 top-K 检测结果。

        Args:
            Outputs: 模型输出 dict
            TargetSizes: [B, 2] 原始图像尺寸 (h, w)

        Returns:
            结果列表，每个元素包含 scores, labels, boxes, count
        """
        OutLogits = Outputs["pred_logits"]  # [B, NQ, 1]
        OutBbox = Outputs["pred_boxes"]  # [B, NQ, 4]

        Prob = OutLogits.sigmoid().squeeze(-1)  # [B, NQ]
        TopkValues, TopkIndexes = torch.topk(
            Prob, min(self.NumSelect, Prob.shape[1]), dim=1
        )

        from rfdetr.util import box_ops

        Boxes = box_ops.box_cxcywh_to_xyxy(OutBbox)
        Boxes = torch.gather(
            Boxes, 1, TopkIndexes.unsqueeze(-1).repeat(1, 1, 4)
        )

        # 归一化坐标 → 绝对坐标
        ImgH, ImgW = TargetSizes.unbind(1)
        ScaleFct = torch.stack([ImgW, ImgH, ImgW, ImgH], dim=1)
        Boxes = Boxes * ScaleFct[:, None, :]

        Results = []
        for I in range(OutLogits.shape[0]):
            Count = (Prob[I] > 0.5).sum().item()
            Results.append(
                {
                    "scores": TopkValues[I],
                    "labels": torch.zeros_like(TopkValues[I], dtype=torch.int64),
                    "boxes": Boxes[I],
                    "count": Count,
                }
            )
        return Results


def BuildFscdModel(Args) -> FscdLWDETR:
    """构建 FSCD 模型。

    复用 RF-DETR 的 backbone 构建流程，替换上层为 FSCD 专用模块。
    """
    Device = torch.device(Args.device)

    # 复用原始 build_model 构建 backbone 和 transformer
    # 先临时设置 num_classes
    OrigNumClasses = Args.num_classes
    Args.num_classes = 0  # 只用于 build backbone+transformer，class_embed 会被忽略

    from rfdetr.models.backbone import build_backbone
    from rfdetr.models.transformer import build_transformer

    Backbone = build_backbone(
        encoder=Args.encoder,
        vit_encoder_num_layers=Args.vit_encoder_num_layers,
        pretrained_encoder=Args.pretrained_encoder,
        window_block_indexes=Args.window_block_indexes,
        drop_path=Args.drop_path,
        out_channels=Args.hidden_dim,
        out_feature_indexes=Args.out_feature_indexes,
        projector_scale=Args.projector_scale,
        use_cls_token=Args.use_cls_token,
        hidden_dim=Args.hidden_dim,
        position_embedding=Args.position_embedding,
        freeze_encoder=Args.freeze_encoder,
        layer_norm=Args.layer_norm,
        target_shape=(
            Args.shape
            if hasattr(Args, "shape")
            else (
                (Args.resolution, Args.resolution)
                if hasattr(Args, "resolution")
                else (640, 640)
            )
        ),
        rms_norm=Args.rms_norm,
        backbone_lora=Args.backbone_lora,
        force_no_pretrain=getattr(Args, "force_no_pretrain", False),
        gradient_checkpointing=Args.gradient_checkpointing,
        load_dinov2_weights=Args.pretrain_weights is None,
        patch_size=Args.patch_size,
        num_windows=Args.num_windows,
        positional_encoding_size=Args.positional_encoding_size,
    )

    Args.num_feature_levels = len(Args.projector_scale)
    TransformerModule = build_transformer(Args)

    Args.num_classes = OrigNumClasses

    Model = FscdLWDETR(
        Backbone=Backbone,
        TransformerModule=TransformerModule,
        SegmentationHead=None,
        NumQueries=Args.num_queries,
        AuxLoss=Args.aux_loss,
        GroupDetr=Args.group_detr,
        TwoStage=Args.two_stage,
        LiteRefpointRefine=Args.lite_refpoint_refine,
        BboxReparam=Args.bbox_reparam,
        ExemplarRoiSize=getattr(Args, "exemplar_roi_size", 7),
        PrototypeDim=Args.hidden_dim,
        NumExemplars=getattr(Args, "num_exemplars", 3),
        PriorProb=getattr(Args, "prior_prob", 0.01),
    )

    return Model


def BuildFscdCriterionAndPostprocessors(Args):
    """构建 FSCD 损失函数和后处理器。"""
    Device = torch.device(Args.device)

    # 构建 matcher — class-agnostic，cost_class 权重降低
    Matcher = build_matcher(Args)

    WeightDict = {
        "loss_ce": Args.cls_loss_coef,
        "loss_bbox": Args.bbox_loss_coef,
        "loss_giou": Args.giou_loss_coef,
        "loss_count": getattr(Args, "count_loss_coef", 0.5),
    }

    if Args.aux_loss:
        AuxWeightDict = {}
        for I in range(Args.dec_layers - 1):
            AuxWeightDict.update({K + f"_{I}": V for K, V in WeightDict.items()})
        if Args.two_stage:
            AuxWeightDict.update(
                {K + "_enc": V for K, V in WeightDict.items()}
            )
        WeightDict.update(AuxWeightDict)

    Losses = ["labels", "boxes", "cardinality", "count"]

    Criterion = FscdSetCriterion(
        Matcher=Matcher,
        WeightDict=WeightDict,
        FocalAlpha=Args.focal_alpha,
        FocalGamma=getattr(Args, "focal_gamma", 2.0),
        Losses=Losses,
        GroupDetr=Args.group_detr,
        CountLossCoef=getattr(Args, "count_loss_coef", 0.5),
        UseTopkCountLoss=getattr(Args, "use_topk_count_loss", True),
        CountTopkFactor=getattr(Args, "count_topk_factor", 3.0),
        CountTopkMin=getattr(Args, "count_topk_min", 1),
        CountTopkMax=getattr(Args, "count_topk_max", 150),
    )
    Criterion.to(Device)

    Postprocess = FscdPostProcess(NumSelect=Args.num_select)

    return Criterion, Postprocess
