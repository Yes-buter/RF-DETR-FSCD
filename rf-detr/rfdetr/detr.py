# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import json
import os
from collections import defaultdict
from logging import getLogger
from typing import Union, List
from copy import deepcopy

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image

try:
    torch.set_float32_matmul_precision('high')
except:
    pass

from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
    RFDETRMediumConfig,
    RFDETRSegPreviewConfig,
    RFDETRFSCDConfig,
    TrainConfig,
    SegmentationTrainConfig,
    FSCDTrainConfig,
    ModelConfig
)
from rfdetr.main import Model, download_pretrain_weights
from rfdetr.util.metrics import MetricsPlotSink, MetricsTensorBoardSink, MetricsWandBSink
from rfdetr.util.coco_classes import COCO_CLASSES

logger = getLogger(__name__)
class RFDETR:
    """
    The base RF-DETR class implements the core methods for training RF-DETR models,
    running inference on the models, optimising models, and uploading trained
    models for deployment.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def maybe_download_pretrain_weights(self):
        """
        Download pre-trained weights if they are not already downloaded.
        """
        download_pretrain_weights(self.model_config.pretrain_weights)

    def get_model_config(self, **kwargs):
        """
        Retrieve the configuration parameters used by the model.
        """
        return ModelConfig(**kwargs)

    def train(self, **kwargs):
        """
        Train an RF-DETR model.
        """
        config = self.get_train_config(**kwargs)
        self.train_from_config(config, **kwargs)
    
    def optimize_for_inference(self, compile=True, batch_size=1, dtype=torch.float32):
        self.remove_optimized_model()

        self.model.inference_model = deepcopy(self.model.model)
        self.model.inference_model.eval()
        self.model.inference_model.export()

        self._optimized_resolution = self.model.resolution
        self._is_optimized_for_inference = True

        self.model.inference_model = self.model.inference_model.to(dtype=dtype)
        self._optimized_dtype = dtype

        if compile:
            self.model.inference_model = torch.jit.trace(
                self.model.inference_model,
                torch.randn(
                    batch_size, 3, self.model.resolution, self.model.resolution, 
                    device=self.model.device,
                    dtype=dtype
                )
            )
            self._optimized_has_been_compiled = True
            self._optimized_batch_size = batch_size
    
    def remove_optimized_model(self):
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_half = False
    
    def export(self, **kwargs):
        """
        Export your model to an ONNX file.

        See [the ONNX export documentation](https://rfdetr.roboflow.com/learn/train/#onnx-export) for more information.
        """
        self.model.export(**kwargs)

    def train_from_config(self, config: TrainConfig, **kwargs):
        if config.dataset_file == "roboflow":
            with open(
                os.path.join(config.dataset_dir, "train", "_annotations.coco.json"), "r"
            ) as f:
                anns = json.load(f)
                num_classes = len(anns["categories"])
                class_names = [c["name"] for c in anns["categories"] if c["supercategory"] != "none"]
                self.model.class_names = class_names
        elif config.dataset_file == "coco":
            class_names = COCO_CLASSES
            num_classes = 90
        else:
            raise ValueError(f"Invalid dataset file: {config.dataset_file}")

        if self.model_config.num_classes != num_classes:
            self.model.reinitialize_detection_head(num_classes)
        
        train_config = config.dict()
        model_config = self.model_config.dict()
        model_config.pop("num_classes")
        if "class_names" in model_config:
            model_config.pop("class_names")
        
        if "class_names" in train_config and train_config["class_names"] is None:
            train_config["class_names"] = class_names

        for k, v in train_config.items():
            if k in model_config:
                model_config.pop(k)
            if k in kwargs:
                kwargs.pop(k)
        
        all_kwargs = {**model_config, **train_config, **kwargs, "num_classes": num_classes}

        metrics_plot_sink = MetricsPlotSink(output_dir=config.output_dir)
        self.callbacks["on_fit_epoch_end"].append(metrics_plot_sink.update)
        self.callbacks["on_train_end"].append(metrics_plot_sink.save)

        if config.tensorboard:
            metrics_tensor_board_sink = MetricsTensorBoardSink(output_dir=config.output_dir)
            self.callbacks["on_fit_epoch_end"].append(metrics_tensor_board_sink.update)
            self.callbacks["on_train_end"].append(metrics_tensor_board_sink.close)

        if config.wandb:
            metrics_wandb_sink = MetricsWandBSink(
                output_dir=config.output_dir,
                project=config.project,
                run=config.run,
                config=config.model_dump()
            )
            self.callbacks["on_fit_epoch_end"].append(metrics_wandb_sink.update)
            self.callbacks["on_train_end"].append(metrics_wandb_sink.close)

        if config.early_stopping:
            from rfdetr.util.early_stopping import EarlyStoppingCallback
            early_stopping_callback = EarlyStoppingCallback(
                model=self.model,
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
                use_ema=config.early_stopping_use_ema,
                segmentation_head=config.segmentation_head
            )
            self.callbacks["on_fit_epoch_end"].append(early_stopping_callback.update)

        self.model.train(
            **all_kwargs,
            callbacks=self.callbacks,
        )

    def get_train_config(self, **kwargs):
        """
        Retrieve the configuration parameters that will be used for training.
        """
        return TrainConfig(**kwargs)

    def get_model(self, config: ModelConfig):
        """
        Retrieve a model instance based on the provided configuration.
        """
        return Model(**config.dict())
    
    # Get class_names from the model
    @property
    def class_names(self):
        """
        Retrieve the class names supported by the loaded model.

        Returns:
            dict: A dictionary mapping class IDs to class names. The keys are integers starting from
        """
        if hasattr(self.model, 'class_names') and self.model.class_names:
            return {i+1: name for i, name in enumerate(self.model.class_names)}
            
        return COCO_CLASSES

    def predict(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        threshold: float = 0.5,
        **kwargs,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images (Union[str, Image.Image, np.ndarray, torch.Tensor, List[Union[str, np.ndarray, Image.Image, torch.Tensor]]]):
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold (float, optional):
                The minimum confidence score needed to consider a detected bounding box valid.
            **kwargs:
                Additional keyword arguments.

        Returns:
            Union[sv.Detections, List[sv.Detections]]: A single or multiple Detections
                objects, each containing bounding box coordinates, confidence scores,
                and class IDs.
        """
        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. "
                "Latency may be higher than expected. "
                "You can optimize the model for inference by calling model.optimize_for_inference()."
            )
            self._has_warned_about_not_being_optimized_for_inference = True

            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []

        for img in images:

            if isinstance(img, str):
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                img = F.to_tensor(img)
            
            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is "
                    "normalized (scaled to [0, 1])."
                )
            if img.shape[0] != 3:
                raise ValueError(
                    f"Invalid image shape. Expected 3 channels (RGB), but got "
                    f"{img.shape[0]} channels."
                )
            img_tensor = img
            
            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = img_tensor.to(self.model.device)
            img_tensor = F.normalize(img_tensor, self.means, self.stds)
            img_tensor = F.resize(img_tensor, (self.model.resolution, self.model.resolution))

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        if self._is_optimized_for_inference:
            if self._optimized_resolution != batch_tensor.shape[2]:
                # this could happen if someone manually changes self.model.resolution after optimizing the model
                raise ValueError(f"Resolution mismatch. "
                                 f"Model was optimized for resolution {self._optimized_resolution}, "
                                 f"but got {batch_tensor.shape[2]}. "
                                 "You can explicitly remove the optimized model by calling model.remove_optimized_model().")
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(f"Batch size mismatch. "
                                     f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                                     f"but got {batch_tensor.shape[0]}. "
                                     "You can explicitly remove the optimized model by calling model.remove_optimized_model(). "
                                     "Alternatively, you can recompile the optimized model for a different batch size "
                                     "by calling model.optimize_for_inference(batch_size=<new_batch_size>).")

        with torch.inference_mode():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
                if len(predictions) == 3:
                    predictions["pred_masks"] = predictions[2]
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocess(predictions, target_sizes=target_sizes)

        detections_list = []
        for result in results:
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                )

            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]
    
    def deploy_to_roboflow(self, workspace: str, project_id: str, version: str, api_key: str = None, size: str = None):
        """
        Deploy the trained RF-DETR model to Roboflow.

        Deploying with Roboflow will create a Serverless API to which you can make requests.

        You can also download weights into a Roboflow Inference deployment for use in Roboflow Workflows and on-device deployment.

        Args:
            workspace (str): The name of the Roboflow workspace to deploy to.
            project_ids (List[str]): A list of project IDs to which the model will be deployed
            api_key (str, optional): Your Roboflow API key. If not provided,
                it will be read from the environment variable `ROBOFLOW_API_KEY`.
            size (str, optional): The size of the model to deploy. If not provided,
                it will default to the size of the model being trained (e.g., "rfdetr-base", "rfdetr-large", etc.).
            model_name (str, optional): The name you want to give the uploaded model.
            If not provided, it will default to "<size>-uploaded".
        Raises:
            ValueError: If the `api_key` is not provided and not found in the environment
                variable `ROBOFLOW_API_KEY`, or if the `size` is not set for custom architectures.
        """
        from roboflow import Roboflow
        import shutil
        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")


        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        size = self.size or size
        tmp_out_dir = ".roboflow_temp_upload"
        os.makedirs(tmp_out_dir, exist_ok=True)
        outpath = os.path.join(tmp_out_dir, "weights.pt")
        torch.save(
            {
                "model": self.model.model.state_dict(),
                "args": self.model.args
            }, outpath
        )
        project = workspace.project(project_id)
        version = project.version(version)
        version.deploy(
            model_type=size,
            model_path=tmp_out_dir,
            filename="weights.pt"
        )
        shutil.rmtree(tmp_out_dir)



class RFDETRBase(RFDETR):
    """
    Train an RF-DETR Base model (29M parameters).
    """
    size = "rfdetr-base"
    def get_model_config(self, **kwargs):
        return RFDETRBaseConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRLarge(RFDETR):
    """
    Train an RF-DETR Large model.
    """
    size = "rfdetr-large"
    def get_model_config(self, **kwargs):
        return RFDETRLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRNano(RFDETR):
    """
    Train an RF-DETR Nano model.
    """
    size = "rfdetr-nano"
    def get_model_config(self, **kwargs):
        return RFDETRNanoConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRSmall(RFDETR):
    """
    Train an RF-DETR Small model.
    """
    size = "rfdetr-small"
    def get_model_config(self, **kwargs):
        return RFDETRSmallConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRMedium(RFDETR):
    """
    Train an RF-DETR Medium model.
    """
    size = "rfdetr-medium"
    def get_model_config(self, **kwargs):
        return RFDETRMediumConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)

class RFDETRSegPreview(RFDETR):
    size = "rfdetr-seg-preview"
    def get_model_config(self, **kwargs):
        return RFDETRSegPreviewConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return SegmentationTrainConfig(**kwargs)


class RFDETRFSCD:
    """RF-DETR Few-Shot Counting and Detection 模型。

    接受图像 + exemplar bounding boxes，输出所有同类目标的检测结果和总数。
    基于 RF-DETR Base 架构，新增 exemplar 条件化模块。

    Example:
        >>> model = RFDETRFSCD()
        >>> exemplar_boxes = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        >>> detections, count = model.predict("image.jpg", exemplar_boxes)
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    def __init__(self, **kwargs):
        from rfdetr.config import RFDETRFSCDConfig, FSCDTrainConfig
        from rfdetr.FscdMain import FscdModel

        self.ModelConfig = RFDETRFSCDConfig(**kwargs)
        download_pretrain_weights(self.ModelConfig.pretrain_weights)
        self.FscdModelInst = FscdModel(**self.ModelConfig.dict())
        self.Model = self.FscdModelInst.Model
        self.Device = self.FscdModelInst.Device
        self.Resolution = self.FscdModelInst.Resolution

    def predict(
        self,
        Images: Union[str, Image.Image, np.ndarray, torch.Tensor,
                       List[Union[str, np.ndarray, Image.Image, torch.Tensor]]],
        ExemplarBoxes: Union[List[List[float]], torch.Tensor, np.ndarray],
        Threshold: float = 0.3,
    ):
        """执行 FSCD 推理。

        Args:
            Images: 输入图像（单张或批量）
            ExemplarBoxes: exemplar bounding boxes, 归一化 xyxy 格式
                形状 [NumExemplars, 4] 或 [B, NumExemplars, 4]
            Threshold: 置信度阈值

        Returns:
            (Detections, Count): sv.Detections 对象和预测数量
        """
        self.Model.eval()

        if not isinstance(Images, list):
            Images = [Images]

        OrigSizes = []
        ProcessedImages = []

        for Img in Images:
            if isinstance(Img, str):
                Img = Image.open(Img)
            if not isinstance(Img, torch.Tensor):
                Img = F.to_tensor(Img)

            H, W = Img.shape[1:]
            OrigSizes.append((H, W))

            ImgTensor = Img.to(self.Device)
            ImgTensor = F.normalize(ImgTensor, self.means, self.stds)
            ImgTensor = F.resize(
                ImgTensor,
                (self.Resolution, self.Resolution),
            )
            ProcessedImages.append(ImgTensor)

        BatchTensor = torch.stack(ProcessedImages)
        BatchSize = BatchTensor.shape[0]

        # 处理 exemplar boxes
        if isinstance(ExemplarBoxes, (list, np.ndarray)):
            ExemplarBoxesTensor = torch.tensor(
                ExemplarBoxes, dtype=torch.float32
            )
        else:
            ExemplarBoxesTensor = ExemplarBoxes.float()

        if ExemplarBoxesTensor.dim() == 2:
            ExemplarBoxesTensor = ExemplarBoxesTensor.unsqueeze(0).expand(
                BatchSize, -1, -1
            )

        ExemplarBoxesTensor = ExemplarBoxesTensor.to(self.Device)

        with torch.inference_mode():
            Predictions = self.Model(BatchTensor, ExemplarBoxesTensor)

            TargetSizes = torch.tensor(OrigSizes, device=self.Device)
            Results = self.FscdModelInst.Postprocess(Predictions, TargetSizes)

        DetectionsList = []
        Counts = []
        for Result in Results:
            Scores = Result["scores"]
            Labels = Result["labels"]
            Boxes = Result["boxes"]
            Count = Result.get("count", (Scores > Threshold).sum().item())

            Keep = Scores > Threshold
            Scores = Scores[Keep]
            Labels = Labels[Keep]
            Boxes = Boxes[Keep]

            Detections = sv.Detections(
                xyxy=Boxes.float().cpu().numpy(),
                confidence=Scores.float().cpu().numpy(),
                class_id=Labels.cpu().numpy(),
            )
            DetectionsList.append(Detections)
            Counts.append(Count)

        if len(DetectionsList) == 1:
            return DetectionsList[0], Counts[0]
        return DetectionsList, Counts

    def get_train_config(self, **kwargs):
        """获取 FSCD 训练配置。"""
        return FSCDTrainConfig(**kwargs)

    def train(self, **kwargs):
        """启动 FSCD 训练。

        Usage::

            model = RFDETRFSCD()
            model.train(
                dataset_dir="path/to/FSCD147",
                epochs=50,
                batch_size=4,
            )
        """
        import datetime
        import json
        import math
        import random
        import shutil
        import time
        from collections import defaultdict
        from pathlib import Path

        import numpy as np
        from torch.utils.data import DataLoader, DistributedSampler

        from rfdetr.config import FSCDTrainConfig
        from rfdetr.datasets.Fscd147 import BuildFscd147
        from rfdetr.FscdEngine import (
            FscdCollateFn,
            FscdEvaluate,
            FscdTrainOneEpoch,
        )
        from rfdetr.models.FscdDetr import (
            BuildFscdCriterionAndPostprocessors,
        )
        from rfdetr.main import populate_args
        from rfdetr.util.get_param_dicts import get_param_dict
        from rfdetr.util.utils import ModelEma, BestMetricHolder

        # ---- 配置合并 ----
        TrainCfg = self.get_train_config(**kwargs)
        MergedKwargs = {**self.ModelConfig.dict(), **TrainCfg.dict()}
        Args = populate_args(**MergedKwargs)
        # FSCD 强制配置
        Args.dataset_file = "fscd147"
        Args.num_classes = 1
        Args.ia_bce_loss = False
        # dataset_dir → coco_path (populate_args 中 coco_path 默认 None)
        if getattr(Args, "coco_path", None) is None and hasattr(Args, "dataset_dir"):
            Args.coco_path = Args.dataset_dir


        utils.init_distributed_mode(Args)
        print(Args)
        Device = torch.device(Args.device)

        # 种子
        Seed = Args.seed + utils.get_rank()
        torch.manual_seed(Seed)
        np.random.seed(Seed)
        random.seed(Seed)

        # ---- 模型 ----
        Model = self.Model
        Model.to(Device)
        Model.train()

        ModelWithoutDdp = Model
        if Args.distributed:
            Model = torch.nn.parallel.DistributedDataParallel(
                Model, device_ids=[Args.gpu], find_unused_parameters=True,
            )
            ModelWithoutDdp = Model.module

        NParameters = sum(P.numel() for P in Model.parameters() if P.requires_grad)
        print(f"Number of trainable params: {NParameters}")

        # ---- 损失 + 后处理 ----
        Criterion, Postprocess = BuildFscdCriterionAndPostprocessors(Args)
        Criterion.to(Device)

        # ---- 数据集 ----
        DatasetTrain = BuildFscd147(
            ImageSet="train",
            DataRoot=Args.coco_path if hasattr(Args, "coco_path") else Args.dataset_dir,
            Resolution=Args.resolution,
            NumExemplars=getattr(Args, "num_exemplars", 3),
        )
        DatasetVal = BuildFscd147(
            ImageSet="val",
            DataRoot=Args.coco_path if hasattr(Args, "coco_path") else Args.dataset_dir,
            Resolution=Args.resolution,
            NumExemplars=getattr(Args, "num_exemplars", 3),
        )

        print(f"Train: {len(DatasetTrain)} samples | Val: {len(DatasetVal)} samples")

        # ---- Sampler ----
        if Args.distributed:
            SamplerTrain = DistributedSampler(DatasetTrain)
            SamplerVal = DistributedSampler(DatasetVal, shuffle=False)
        else:
            SamplerTrain = torch.utils.data.RandomSampler(DatasetTrain)
            SamplerVal = torch.utils.data.SequentialSampler(DatasetVal)

        EffectiveBatchSize = Args.batch_size * Args.grad_accum_steps

        BatchSamplerTrain = torch.utils.data.BatchSampler(
            SamplerTrain, EffectiveBatchSize, drop_last=True,
        )
        DataLoaderTrain = DataLoader(
            DatasetTrain,
            batch_sampler=BatchSamplerTrain,
            collate_fn=FscdCollateFn,
            num_workers=Args.num_workers,
        )
        DataLoaderVal = DataLoader(
            DatasetVal,
            batch_size=Args.batch_size,
            sampler=SamplerVal,
            drop_last=False,
            collate_fn=FscdCollateFn,
            num_workers=Args.num_workers,
        )

        # ---- Optimizer ----
        ParamDicts = get_param_dict(Args, ModelWithoutDdp)
        ParamDicts = [P for P in ParamDicts if P["params"].requires_grad]
        Optimizer = torch.optim.AdamW(
            ParamDicts, lr=Args.lr, weight_decay=Args.weight_decay,
        )

        # ---- LR Scheduler (cosine + warmup) ----
        TotalBatchSizeForLr = Args.batch_size * utils.get_world_size() * Args.grad_accum_steps
        NumTrainingStepsPerEpochLr = (
            (len(DatasetTrain) + TotalBatchSizeForLr - 1) // TotalBatchSizeForLr
        )
        TotalTrainingStepsLr = NumTrainingStepsPerEpochLr * Args.epochs
        WarmupStepsLr = NumTrainingStepsPerEpochLr * Args.warmup_epochs

        def LrLambda(CurrentStep: int) -> float:
            if CurrentStep < WarmupStepsLr:
                return float(CurrentStep) / float(max(1, WarmupStepsLr))
            Progress = float(CurrentStep - WarmupStepsLr) / float(
                max(1, TotalTrainingStepsLr - WarmupStepsLr)
            )
            return Args.lr_min_factor + (1 - Args.lr_min_factor) * 0.5 * (
                1 + math.cos(math.pi * Progress)
            )

        LrScheduler = torch.optim.lr_scheduler.LambdaLR(Optimizer, lr_lambda=LrLambda)

        # ---- EMA ----
        if Args.use_ema:
            EmaM = ModelEma(ModelWithoutDdp, decay=Args.ema_decay, tau=Args.ema_tau)
        else:
            EmaM = None

        # ---- Output dir ----
        OutputDir = Path(Args.output_dir)
        OutputDir.mkdir(parents=True, exist_ok=True)

        # ---- Resume ----
        if Args.resume:
            Checkpoint = torch.load(Args.resume, map_location="cpu", weights_only=False)
            ModelWithoutDdp.load_state_dict(Checkpoint["model"], strict=True)
            if "optimizer" in Checkpoint and "lr_scheduler" in Checkpoint and "epoch" in Checkpoint:
                Optimizer.load_state_dict(Checkpoint["optimizer"])
                LrScheduler.load_state_dict(Checkpoint["lr_scheduler"])
                Args.start_epoch = Checkpoint["epoch"] + 1

        # ---- Training Loop ----
        print("Start FSCD training")
        StartTime = time.time()
        BestMae = float("inf")
        BestEpoch = 0
        Callbacks = defaultdict(list)

        TotalBatchSize = EffectiveBatchSize * utils.get_world_size()
        NumTrainingStepsPerEpoch = (
            (len(DatasetTrain) + TotalBatchSize - 1) // TotalBatchSize
        )

        for Epoch in range(getattr(Args, "start_epoch", 0), Args.epochs):
            EpochStart = time.time()
            if Args.distributed:
                SamplerTrain.set_epoch(Epoch)

            Model.train()
            Criterion.train()

            TrainStats = FscdTrainOneEpoch(
                Model=Model,
                Criterion=Criterion,
                LrScheduler=LrScheduler,
                DataLoader=DataLoaderTrain,
                Optimizer=Optimizer,
                Device=Device,
                Epoch=Epoch,
                BatchSize=EffectiveBatchSize,
                MaxNorm=Args.clip_max_norm,
                EmaM=EmaM,
                NumTrainingStepsPerEpoch=NumTrainingStepsPerEpoch,
                Args=Args,
                Callbacks=Callbacks,
            )

            TrainTime = str(datetime.timedelta(seconds=int(time.time() - EpochStart)))

            # ---- Checkpoint ----
            if Args.output_dir:
                CheckpointPaths = [OutputDir / "checkpoint.pth"]
                if (Epoch + 1) % getattr(Args, "checkpoint_interval", 5) == 0:
                    CheckpointPaths.append(OutputDir / f"checkpoint{Epoch:04}.pth")

                Weights = {
                    "model": ModelWithoutDdp.state_dict(),
                    "optimizer": Optimizer.state_dict(),
                    "lr_scheduler": LrScheduler.state_dict(),
                    "epoch": Epoch,
                    "args": Args,
                }
                if EmaM is not None:
                    Weights["ema_model"] = EmaM.module.state_dict()

                for Cp in CheckpointPaths:
                    Cp.parent.mkdir(parents=True, exist_ok=True)
                    utils.save_on_master(Weights, Cp)

            # ---- Validation ----
            with torch.inference_mode():
                ValStats = FscdEvaluate(
                    Model=Model,
                    Criterion=Criterion,
                    Postprocess=Postprocess,
                    DataLoader=DataLoaderVal,
                    Device=Device,
                    Args=Args,
                )

            Mae = ValStats["mae"]
            Rmse = ValStats["rmse"]

            # best model 以 MAE 为准
            IsBest = Mae < BestMae
            if IsBest:
                BestMae = Mae
                BestEpoch = Epoch
                BestCp = OutputDir / "checkpoint_best.pth"
                utils.save_on_master(
                    {"model": ModelWithoutDdp.state_dict(), "epoch": Epoch, "args": Args},
                    BestCp,
                )

            # EMA 评估
            if EmaM is not None:
                EmaValStats = FscdEvaluate(
                    Model=EmaM.module,
                    Criterion=Criterion,
                    Postprocess=Postprocess,
                    DataLoader=DataLoaderVal,
                    Device=Device,
                    Args=Args,
                )
                EmaMae = EmaValStats["mae"]
                if EmaMae < BestMae:
                    BestMae = EmaMae
                    BestEpoch = Epoch
                    BestCp = OutputDir / "checkpoint_best.pth"
                    utils.save_on_master(
                        {"model": EmaM.module.state_dict(), "epoch": Epoch, "args": Args},
                        BestCp,
                    )

            # ---- Log ----
            LogStats = {
                **{f"train_{K}": V for K, V in TrainStats.items()},
                **{f"val_{K}": V for K, V in ValStats.items()},
                "epoch": Epoch,
                "n_parameters": NParameters,
                "best_mae": BestMae,
                "best_epoch": BestEpoch,
                "train_epoch_time": TrainTime,
            }
            if EmaM is not None:
                LogStats.update({f"ema_val_{K}": V for K, V in EmaValStats.items()})

            if Args.output_dir and utils.is_main_process():
                with (OutputDir / "log.txt").open("a") as F:
                    F.write(json.dumps(LogStats) + "\n")

            EpochTime = str(datetime.timedelta(seconds=int(time.time() - EpochStart)))
            print(
                f"Epoch [{Epoch}/{Args.epochs}] done in {EpochTime} | "
                f"MAE: {Mae:.2f} | RMSE: {Rmse:.2f} | "
                f"Best MAE: {BestMae:.2f} @ epoch {BestEpoch}"
            )

        # ---- 训练结束 ----
        TotalTime = str(datetime.timedelta(seconds=int(time.time() - StartTime)))
        print(f"Training complete in {TotalTime}")
        print(f"Best MAE: {BestMae:.2f} @ epoch {BestEpoch}")

        # 加载 best 权重
        if (OutputDir / "checkpoint_best.pth").exists():
            BestState = torch.load(
                OutputDir / "checkpoint_best.pth", map_location="cpu", weights_only=False
            )["model"]
            ModelWithoutDdp.load_state_dict(BestState)
            print("Loaded best checkpoint")

        self.Model = ModelWithoutDdp
        self.Model.eval()

