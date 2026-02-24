# ------------------------------------------------------------------------
# RF-DETR — FSCD Extension
# FSCD Model Management (类似 main.py 的 Model 类)
# ------------------------------------------------------------------------

"""
FSCD 版本的模型管理类。
负责从 RF-DETR 预训练权重构建 FSCD 模型，
处理权重映射（class_embed → ObjectnessEmbed 等）。
"""

import argparse
from logging import getLogger
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model

from rfdetr.main import download_pretrain_weights, populate_args
from rfdetr.models.FscdDetr import (
    BuildFscdCriterionAndPostprocessors,
    BuildFscdModel,
    FscdPostProcess,
)

Logger = getLogger(__name__)


class FscdModel:
    """FSCD 模型管理类。

    类似 rfdetr.main.Model，但构建 FscdLWDETR 而非 LWDETR，
    并正确处理从 RF-DETR 预训练权重到 FSCD 模型的权重映射。
    """

    def __init__(self, **kwargs) -> None:
        Args = populate_args(**kwargs)
        self.Args = Args
        self.Resolution = Args.resolution
        self.Device = torch.device(Args.device)

        # 构建 FSCD 模型
        self.Model = BuildFscdModel(Args)

        # 加载 RF-DETR 预训练权重（部分匹配）
        if Args.pretrain_weights is not None:
            self._LoadPretrainWeights(Args)

        # 可选: backbone LoRA
        if Args.backbone_lora:
            Logger.info("Applying LoRA to backbone")
            LoraConf = LoraConfig(
                r=16,
                lora_alpha=16,
                use_dora=True,
                target_modules=[
                    "q_proj", "v_proj", "k_proj",
                    "qkv",
                    "query", "key", "value", "cls_token", "register_tokens",
                ],
            )
            self.Model.Backbone[0].encoder = get_peft_model(
                self.Model.Backbone[0].encoder, LoraConf
            )

        self.Model = self.Model.to(self.Device)
        self.Postprocess = FscdPostProcess(NumSelect=Args.num_select)
        self.ClassNames = ["object"]  # FSCD 只有一类: 目标

    def _LoadPretrainWeights(self, Args) -> None:
        """从 RF-DETR 预训练权重加载，处理键名映射。

        RF-DETR 权重中的 class_embed/backbone/transformer 等需要
        映射到 FSCD 模型中对应的 ObjectnessEmbed/Backbone/TransformerModule 等。
        """
        Logger.info(f"Loading pretrain weights from: {Args.pretrain_weights}")
        try:
            Checkpoint = torch.load(
                Args.pretrain_weights, map_location="cpu", weights_only=False
            )
        except Exception as E:
            Logger.warning(f"Failed to load weights: {E}, re-downloading...")
            download_pretrain_weights(Args.pretrain_weights, redownload=True)
            Checkpoint = torch.load(
                Args.pretrain_weights, map_location="cpu", weights_only=False
            )

        if "model" in Checkpoint:
            StateDict = Checkpoint["model"]
        else:
            StateDict = Checkpoint

        # 键名映射: RF-DETR → FSCD
        MappedState = {}
        SkippedKeys = []

        for Key, Value in StateDict.items():
            NewKey = self._MapKey(Key)
            if NewKey is None:
                SkippedKeys.append(Key)
                continue
            MappedState[NewKey] = Value

        # 加载（strict=False 忽略不匹配的键）
        MissingKeys, UnexpectedKeys = self.Model.load_state_dict(
            MappedState, strict=False
        )

        Logger.info(f"Loaded pretrain weights. "
                     f"Skipped {len(SkippedKeys)} original keys, "
                     f"Missing {len(MissingKeys)} keys in FSCD model, "
                     f"Unexpected {len(UnexpectedKeys)} keys.")
        if SkippedKeys:
            Logger.debug(f"Skipped keys: {SkippedKeys[:10]}...")
        if MissingKeys:
            Logger.info(f"Missing keys (new FSCD modules): {MissingKeys[:10]}...")

    def _MapKey(self, Key: str) -> Optional[str]:
        """将 RF-DETR 的 state_dict 键映射到 FSCD 模型的键。

        主要映射:
        - backbone.* → Backbone.*
        - transformer.* → TransformerModule.*
        - bbox_embed.* → BboxEmbed.*
        - refpoint_embed.* → RefpointEmbed.*
        - query_feat.* → QueryFeat.*
        - class_embed.* → 跳过（维度不匹配, FSCD 用 ObjectnessEmbed）

        Returns:
            映射后的新键名，None 表示跳过
        """
        # 跳过分类头（维度从 91 → 1，无法直接加载）
        if Key.startswith("class_embed"):
            return None

        # 跳过 enc_out_class_embed（同理）
        if "enc_out_class_embed" in Key:
            return None

        # 简单重命名映射
        _Mappings = {
            "backbone.": "Backbone.",
            "transformer.": "TransformerModule.",
            "bbox_embed.": "BboxEmbed.",
            "refpoint_embed.": "RefpointEmbed.",
            "query_feat.": "QueryFeat.",
            "segmentation_head.": "SegmentationHead.",
        }

        for OldPrefix, NewPrefix in _Mappings.items():
            if Key.startswith(OldPrefix):
                return NewPrefix + Key[len(OldPrefix):]

        # enc_out_bbox_embed 在 transformer 内部
        if Key.startswith("transformer.enc_out_bbox_embed"):
            return "TransformerModule." + Key[len("transformer."):]

        # 未知键保持原样（可能不会被加载，strict=False 会忽略）
        return Key
