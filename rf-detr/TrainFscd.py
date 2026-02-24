# ---------------------------------------------------------------
# FSCD 训练启动脚本
# 用法: python TrainFscd.py --dataset_dir <path-to-FSCD147>
# ---------------------------------------------------------------

"""
FSCD (Few-Shot Counting and Detection) 完整训练脚本。

使用方法:
    python TrainFscd.py --dataset_dir D:/datasets/FSCD147 --epochs 50 --batch_size 4
    python TrainFscd.py --dataset_dir D:/datasets/FSCD147 --resume output_fscd/checkpoint.pth

数据集 splits:
    - Train: 3659 张 (用于训练)
    - Val:   1286 张 (每 epoch 评估, 选 best model)
    - Test:  1190 张 (训练结束后最终评估)
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rfdetr.util.misc as utils
from rfdetr.config import RFDETRFSCDConfig, RFDETRFSCDLargeConfig, FSCDTrainConfig
from rfdetr.datasets.Fscd147 import BuildFscd147
from rfdetr.FscdEngine import FscdCollateFn, FscdTrainOneEpoch, FscdEvaluate
from rfdetr.FscdMain import FscdModel
from rfdetr.main import download_pretrain_weights, populate_args
from rfdetr.models.FscdDetr import BuildFscdCriterionAndPostprocessors


# ---------------------------------------------------------------
# 输出格式化
# ---------------------------------------------------------------

def PrintHeader() -> None:
    """打印训练标题。"""
    print()
    print("\u250c" + "\u2500" * 68 + "\u2510")
    print("\u2502" + "  FSCD-147 Training Pipeline".center(68) + "\u2502")
    print("\u2502" + "  Few-Shot Counting and Detection".center(68) + "\u2502")
    print("\u2514" + "\u2500" * 68 + "\u2518")
    print()


def PrintConfig(Args: argparse.Namespace, NParams: int, NTotal: int,
                TrainSize: int, ValSize: int, TestSize: int) -> None:
    """打印训练配置摘要表格。"""
    print("\u250c" + "\u2500" * 34 + "\u252c" + "\u2500" * 33 + "\u2510")
    print("\u2502" + "  Model Config".ljust(34) + "\u2502" + "  Training Config".ljust(33) + "\u2502")
    print("\u251c" + "\u2500" * 34 + "\u253c" + "\u2500" * 33 + "\u2524")
    print(f"\u2502  Trainable: {NParams:>14,}      \u2502  Epochs:    {Args.epochs:>14}   \u2502")
    print(f"\u2502  Total:     {NTotal:>14,}      \u2502  Batch:     {Args.batch_size:>14}   \u2502")
    print(f"\u2502  Device:    {'CUDA (GPU)':>14}      \u2502  LR:        {Args.lr:>14.1e}   \u2502")
    print(f"\u2502  Resolution:       {560:>7}      \u2502  LR enc:    {Args.lr_encoder:>14.1e}   \u2502")
    print(f"\u2502  Exemplars:        {Args.num_exemplars:>7}      \u2502  Warmup:    {Args.warmup_epochs:>11} ep   \u2502")
    print("\u251c" + "\u2500" * 34 + "\u253c" + "\u2500" * 33 + "\u2524")
    print("\u2502" + "  Dataset Splits".ljust(34) + "\u2502" + "  Features".ljust(33) + "\u2502")
    print("\u251c" + "\u2500" * 34 + "\u253c" + "\u2500" * 33 + "\u2524")
    print(f"\u2502  Train:    {TrainSize:>7} samples      \u2502  AMP:       {'Yes' if Args.amp else 'No':>14}   \u2502")
    print(f"\u2502  Val:      {ValSize:>7} samples      \u2502  EMA:       {'Yes' if Args.use_ema else 'No':>14}   \u2502")
    print(f"\u2502  Test:     {TestSize:>7} samples      \u2502  Grad clip: {Args.clip_max_norm:>14.2f}   \u2502")
    print(f"\u2502  Output: {str(Args.output_dir):>24}  \u2502  Grad accum:{Args.grad_accum_steps:>14}   \u2502")
    print("\u2514" + "\u2500" * 34 + "\u2534" + "\u2500" * 33 + "\u2518")
    print()


def PrintEpochSummary(Epoch: int, TotalEpochs: int,
                      TrainStats: dict, ValStats: dict,
                      BestMae: float, BestEpoch: int,
                      EpochTime: str, EmaValStats: dict = None) -> None:
    """打印 epoch 结束时的摘要表格。"""
    print()
    print(f"  \u2500\u2500\u2500 Epoch {Epoch + 1}/{TotalEpochs} Summary ({EpochTime}) " + "\u2500" * 30)
    print()

    # 表头
    print(f"  {'Metric':>20s}  {'Train':>12s}  {'Val':>12s}", end="")
    if EmaValStats:
        print(f"  {'EMA Val':>12s}", end="")
    print()
    print(f"  {'':>20s}  {'':>12s}  {'':>12s}", end="")
    if EmaValStats:
        print(f"  {'':>12s}", end="")
    print()
    print("  " + "\u2500" * 20 + "  " + "\u2500" * 12 + "  " + "\u2500" * 12, end="")
    if EmaValStats:
        print("  " + "\u2500" * 12, end="")
    print()

    # 指标行
    Rows = [
        ("Total Loss",   TrainStats.get("loss", 0),        ValStats.get("loss", 0)),
        ("Loss CE",      TrainStats.get("loss_ce", 0),     None),
        ("Loss BBox",    TrainStats.get("loss_bbox", 0),   None),
        ("Loss GIoU",    TrainStats.get("loss_giou", 0),   None),
        ("Loss Count",   TrainStats.get("loss_count", 0),  None),
        ("Count Error",  TrainStats.get("count_error", 0), None),
        ("MAE (soft)",   None,                              ValStats.get("mae", 0)),
        ("MAE (hard)",   None,                              ValStats.get("hard_mae", 0)),
        ("RMSE",         None,                              ValStats.get("rmse", 0)),
        ("AP",           None,                              ValStats.get("ap", 0)),
        ("AP50",         None,                              ValStats.get("ap50", 0)),
        ("Avg Soft Cnt", None,                              ValStats.get("mean_pred_count", 0)),
        ("Avg Hard Cnt", None,                              ValStats.get("mean_hard_count", 0)),
        ("Avg GT Cnt",   None,                              ValStats.get("mean_gt_count", 0)),
    ]

    for Name, TrainVal, Val in Rows:
        TrainStr = f"{TrainVal:12.4f}" if TrainVal is not None else f"{'--':>12s}"
        ValStr = f"{Val:12.4f}" if Val is not None else f"{'--':>12s}"
        print(f"  {Name:>20s}  {TrainStr}  {ValStr}", end="")

        if EmaValStats:
            EmaKey = Name.lower().replace(" ", "_")
            EmaVal = EmaValStats.get(EmaKey if EmaKey in EmaValStats else
                                     "mae" if Name == "MAE" else
                                     "rmse" if Name == "RMSE" else
                                     "ap" if Name == "AP" else
                                     "ap50" if Name == "AP50" else
                                     "loss" if Name == "Total Loss" else None)
            if EmaVal is not None and Name in ("Total Loss", "MAE", "RMSE", "AP", "AP50"):
                print(f"  {EmaVal:12.4f}", end="")
            else:
                print(f"  {'--':>12s}", end="")
        print()

    # Best 信息
    print()
    BestMarker = " \u2190 NEW BEST!" if (Epoch == BestEpoch) else ""
    print(f"  \u2605 Best MAE: {BestMae:.4f} (epoch {BestEpoch + 1}){BestMarker}")
    print(f"  \u2605 Learning Rate: {TrainStats.get('lr', 0):.2e}")
    print()


def PrintFinalResults(BestMae: float, BestEpoch: int,
                      TestStats: dict, TotalTime: str,
                      OutputDir: str) -> None:
    """打印训练结束后的最终结果。"""
    print()
    print("\u250c" + "\u2500" * 50 + "\u2510")
    print("\u2502" + "  Training Complete".center(50) + "\u2502")
    print("\u251c" + "\u2500" * 50 + "\u2524")
    print(f"\u2502  Total Time:   {TotalTime:>32s}  \u2502")
    print(f"\u2502  Best Val MAE: {BestMae:>32.4f}  \u2502")
    print(f"\u2502  Best Epoch:   {BestEpoch + 1:>32d}  \u2502")
    if TestStats:
        print("\u251c" + "\u2500" * 50 + "\u2524")
        print("\u2502" + "  Test Set Results".center(50) + "\u2502")
        print("\u251c" + "\u2500" * 50 + "\u2524")
        print(f"\u2502  Test MAE:     {TestStats['mae']:>32.4f}  \u2502")
        print(f"\u2502  Test RMSE:    {TestStats['rmse']:>32.4f}  \u2502")
        print(f"\u2502  Test AP:      {TestStats.get('ap', 0):>32.4f}  \u2502")
        print(f"\u2502  Test AP50:    {TestStats.get('ap50', 0):>32.4f}  \u2502")
        print(f"\u2502  Avg Pred Cnt: {TestStats['mean_pred_count']:>32.2f}  \u2502")
        print(f"\u2502  Avg GT Cnt:   {TestStats['mean_gt_count']:>32.2f}  \u2502")
    print("\u251c" + "\u2500" * 50 + "\u2524")
    print(f"\u2502  Output Dir: {OutputDir:>35s}  \u2502")
    print("\u2514" + "\u2500" * 50 + "\u2518")
    print()


# ---------------------------------------------------------------
# 参数分组 (FSCD 专用版)
# ---------------------------------------------------------------

def GetFscdParamDicts(
    Args: argparse.Namespace,
    Model: torch.nn.Module,
) -> list:
    """为 FscdLWDETR 构建分层学习率参数组。

    三组参数:
    1. Backbone (encoder): 最低学习率 lr_encoder
    2. Transformer decoder: lr * lr_component_decay
    3. 其他 (FSCD 新增模块 + heads): 基础 lr
    """
    BackboneParams = []
    DecoderParams = []
    OtherParams = []
    FrozenCount = 0

    for Name, Param in Model.named_parameters():
        if not Param.requires_grad:
            FrozenCount += 1
            continue

        if Name.startswith("Backbone."):
            BackboneParams.append(Param)
        elif "TransformerModule." in Name and "decoder" in Name:
            DecoderParams.append(Param)
        else:
            OtherParams.append(Param)

    LrEncoder = getattr(Args, "lr_encoder", Args.lr * 0.1)
    LrDecay = getattr(Args, "lr_component_decay", 1.0)

    ParamDicts = []
    if OtherParams:
        ParamDicts.append({"params": OtherParams, "lr": Args.lr})
    if BackboneParams:
        ParamDicts.append({"params": BackboneParams, "lr": LrEncoder})
    if DecoderParams:
        ParamDicts.append({"params": DecoderParams, "lr": Args.lr * LrDecay})

    print(f"  Param Groups: backbone={len(BackboneParams)}, "
          f"decoder={len(DecoderParams)}, other={len(OtherParams)}, "
          f"frozen={FrozenCount}")

    return ParamDicts


# ---------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------

def Train(Args: argparse.Namespace) -> None:
    """FSCD 训练主流程。"""
    PrintHeader()

    # ---- 初始化 ----
    utils.init_distributed_mode(Args)
    Device = torch.device(Args.device)

    Seed = Args.seed + utils.get_rank()
    torch.manual_seed(Seed)
    np.random.seed(Seed)
    random.seed(Seed)

    # ---- 模型 ----
    print("  [1/5] Building model ...")
    if Args.model_size == "large":
        ModelConfig = RFDETRFSCDLargeConfig()
        print("  Using RF-DETR Large (DINOv2 ViT-B, hidden_dim=384)")
    else:
        ModelConfig = RFDETRFSCDConfig()
        print("  Using RF-DETR Base (DINOv2 ViT-S, hidden_dim=256)")
    download_pretrain_weights(ModelConfig.pretrain_weights)
    ModelKwargs = ModelConfig.model_dump()
    ModelKwargs["prior_prob"] = Args.prior_prob
    FscdModelInst = FscdModel(**ModelKwargs)
    Model = FscdModelInst.Model
    Model.to(Device)
    Model.train()

    ModelWithoutDdp = Model
    if Args.distributed:
        Model = torch.nn.parallel.DistributedDataParallel(
            Model, device_ids=[Args.gpu], find_unused_parameters=True,
        )
        ModelWithoutDdp = Model.module

    NParams = sum(P.numel() for P in Model.parameters() if P.requires_grad)
    NTotal = sum(P.numel() for P in Model.parameters())

    # ---- Criterion + PostProcess ----
    print("  [2/5] Building criterion ...")
    MergedKwargs = {**ModelConfig.model_dump()}
    MergedKwargs.update({
        "dataset_file": "fscd147",
        "num_classes": 1,
        "lr": Args.lr,
        "lr_encoder": Args.lr_encoder,
        "batch_size": Args.batch_size,
        "epochs": Args.epochs,
        "prior_prob": Args.prior_prob,
        "cls_loss_coef": Args.cls_loss_coef,
        "focal_alpha": Args.focal_alpha,
        "focal_gamma": Args.focal_gamma,
        "set_cost_class": Args.set_cost_class,
        "set_cost_bbox": Args.set_cost_bbox,
        "set_cost_giou": Args.set_cost_giou,
        "count_loss_coef": Args.count_loss_coef,
        "use_topk_count_loss": Args.use_topk_count_loss,
        "count_topk_factor": Args.count_topk_factor,
        "count_topk_min": Args.count_topk_min,
        "count_topk_max": Args.count_topk_max,
    })
    FullArgs = populate_args(**MergedKwargs)
    FullArgs.amp = Args.amp
    FullArgs.grad_accum_steps = Args.grad_accum_steps
    FullArgs.clip_max_norm = Args.clip_max_norm

    Criterion, Postprocess = BuildFscdCriterionAndPostprocessors(FullArgs)
    Criterion.to(Device)

    # ---- 数据集 ----
    print("  [3/5] Building datasets ...")
    DatasetTrain = BuildFscd147(
        ImageSet="train",
        DataRoot=Args.dataset_dir,
        Resolution=ModelConfig.resolution,
        NumExemplars=Args.num_exemplars,
    )
    DatasetVal = BuildFscd147(
        ImageSet="val",
        DataRoot=Args.dataset_dir,
        Resolution=ModelConfig.resolution,
        NumExemplars=Args.num_exemplars,
    )
    DatasetTest = BuildFscd147(
        ImageSet="test",
        DataRoot=Args.dataset_dir,
        Resolution=ModelConfig.resolution,
        NumExemplars=Args.num_exemplars,
    )

    # COCO GT 对象 (用于 AP/AP50 评估, 仅 val/test 有)
    ValCocoGt = DatasetVal.GetCocoApi()
    TestCocoGt = DatasetTest.GetCocoApi()

    # ---- DataLoader ----
    print("  [4/5] Building dataloaders ...")
    if Args.distributed:
        SamplerTrain = torch.utils.data.DistributedSampler(DatasetTrain)
        SamplerVal = torch.utils.data.DistributedSampler(DatasetVal, shuffle=False)
        SamplerTest = torch.utils.data.DistributedSampler(DatasetTest, shuffle=False)
    else:
        SamplerTrain = torch.utils.data.RandomSampler(DatasetTrain)
        SamplerVal = torch.utils.data.SequentialSampler(DatasetVal)
        SamplerTest = torch.utils.data.SequentialSampler(DatasetTest)

    EffectiveBatchSize = Args.batch_size * Args.grad_accum_steps
    BatchSamplerTrain = torch.utils.data.BatchSampler(
        SamplerTrain, EffectiveBatchSize, drop_last=True,
    )
    LoaderTrain = DataLoader(
        DatasetTrain,
        batch_sampler=BatchSamplerTrain,
        collate_fn=FscdCollateFn,
        num_workers=Args.num_workers,
        pin_memory=True,
    )
    LoaderVal = DataLoader(
        DatasetVal,
        batch_size=Args.batch_size,
        sampler=SamplerVal,
        drop_last=False,
        collate_fn=FscdCollateFn,
        num_workers=Args.num_workers,
        pin_memory=True,
    )
    LoaderTest = DataLoader(
        DatasetTest,
        batch_size=Args.batch_size,
        sampler=SamplerTest,
        drop_last=False,
        collate_fn=FscdCollateFn,
        num_workers=Args.num_workers,
        pin_memory=True,
    )

    # ---- Optimizer ----
    print("  [5/5] Building optimizer ...")
    ParamDicts = GetFscdParamDicts(FullArgs, ModelWithoutDdp)
    Optimizer = torch.optim.AdamW(
        ParamDicts, lr=Args.lr, weight_decay=Args.weight_decay,
    )

    # ---- LR Scheduler (cosine + warmup) ----
    TotalBatchSize = EffectiveBatchSize * max(1, utils.get_world_size())
    NumStepsPerEpoch = max(1, len(DatasetTrain) // TotalBatchSize)
    TotalSteps = NumStepsPerEpoch * Args.epochs
    WarmupSteps = NumStepsPerEpoch * Args.warmup_epochs

    def LrLambda(Step: int) -> float:
        if Step < WarmupSteps:
            return float(Step) / float(max(1, WarmupSteps))
        Progress = float(Step - WarmupSteps) / float(max(1, TotalSteps - WarmupSteps))
        MinFactor = getattr(Args, "lr_min_factor", 0.01)
        return MinFactor + (1 - MinFactor) * 0.5 * (1 + math.cos(math.pi * Progress))

    LrScheduler = torch.optim.lr_scheduler.LambdaLR(Optimizer, lr_lambda=LrLambda)

    # ---- EMA ----
    EmaM = None
    if Args.use_ema:
        from rfdetr.util.utils import ModelEma
        EmaM = ModelEma(ModelWithoutDdp, decay=Args.ema_decay)

    # ---- Output dir ----
    OutputDir = Path(Args.output_dir)
    OutputDir.mkdir(parents=True, exist_ok=True)

    # ---- Resume ----
    StartEpoch = 0
    if Args.resume:
        print(f"\n  Resuming from {Args.resume}")
        Ckpt = torch.load(Args.resume, map_location="cpu", weights_only=False)
        ModelWithoutDdp.load_state_dict(Ckpt["model"], strict=True)
        if EmaM is not None and "ema_model" in Ckpt:
            EmaM.module.load_state_dict(Ckpt["ema_model"], strict=True)
            print("  Restored EMA weights from checkpoint")
        if "optimizer" in Ckpt and "epoch" in Ckpt:
            Optimizer.load_state_dict(Ckpt["optimizer"])
            if "lr_scheduler" in Ckpt:
                LrScheduler.load_state_dict(Ckpt["lr_scheduler"])
            StartEpoch = Ckpt["epoch"] + 1
            print(f"  Resumed at epoch {StartEpoch}")

    # ---- 打印配置 ----
    PrintConfig(Args, NParams, NTotal,
                len(DatasetTrain), len(DatasetVal), len(DatasetTest))

    # ---- Training Loop ----
    TrainStart = time.time()
    BestMae = float("inf")
    BestEpoch = -1
    BestThreshold = float(Args.eval_threshold)
    Callbacks = defaultdict(list)

    if Args.auto_threshold:
        ThresholdCandidates = np.linspace(
            Args.threshold_min,
            Args.threshold_max,
            Args.threshold_steps,
            dtype=np.float32,
        ).tolist()
    else:
        ThresholdCandidates = [float(Args.eval_threshold)]

    for Epoch in range(StartEpoch, Args.epochs):
        EpochStart = time.time()

        if Args.distributed:
            SamplerTrain.set_epoch(Epoch)

        # -- Train --
        TrainStats = FscdTrainOneEpoch(
            Model=Model,
            Criterion=Criterion,
            LrScheduler=LrScheduler,
            DataLoader=LoaderTrain,
            Optimizer=Optimizer,
            Device=Device,
            Epoch=Epoch,
            BatchSize=EffectiveBatchSize,
            MaxNorm=Args.clip_max_norm,
            EmaM=EmaM,
            NumTrainingStepsPerEpoch=NumStepsPerEpoch,
            Args=FullArgs,
            Callbacks=Callbacks,
            TotalEpochs=Args.epochs,
        )

        # -- Save checkpoint --
        if Args.output_dir:
            CkptPaths = [OutputDir / "checkpoint.pth"]
            if (Epoch + 1) % Args.save_interval == 0:
                CkptPaths.append(OutputDir / f"checkpoint{Epoch:04d}.pth")

            State = {
                "model": ModelWithoutDdp.state_dict(),
                "optimizer": Optimizer.state_dict(),
                "lr_scheduler": LrScheduler.state_dict(),
                "epoch": Epoch,
            }
            if EmaM is not None:
                State["ema_model"] = EmaM.module.state_dict()

            for P in CkptPaths:
                utils.save_on_master(State, P)

        # -- Validate --
        with torch.inference_mode():
            ValStats = FscdEvaluate(
                Model=Model,
                Criterion=Criterion,
                Postprocess=Postprocess,
                DataLoader=LoaderVal,
                Device=Device,
                Threshold=Args.eval_threshold,
                ThresholdCandidates=ThresholdCandidates,
                CalibrateThreshold=Args.auto_threshold,
                Args=FullArgs,
                Label="Val",
                CocoGt=ValCocoGt,
            )

        Mae = ValStats["mae"]

        IsBest = Mae < BestMae
        if IsBest:
            BestMae = Mae
            BestEpoch = Epoch
            BestThreshold = float(ValStats.get("threshold", Args.eval_threshold))
            utils.save_on_master(
                {"model": ModelWithoutDdp.state_dict(), "epoch": Epoch},
                OutputDir / "checkpoint_best.pth",
            )

        # EMA 评估
        EmaValStats = None
        if EmaM is not None:
            with torch.inference_mode():
                EmaValStats = FscdEvaluate(
                    Model=EmaM.module,
                    Criterion=Criterion,
                    Postprocess=Postprocess,
                    DataLoader=LoaderVal,
                    Device=Device,
                    Threshold=Args.eval_threshold,
                    ThresholdCandidates=ThresholdCandidates,
                    CalibrateThreshold=Args.auto_threshold,
                    Args=FullArgs,
                    Label="EMA Val",
                    CocoGt=ValCocoGt,
                )
            EmaMae = EmaValStats["mae"]
            if EmaMae < BestMae:
                BestMae = EmaMae
                BestEpoch = Epoch
                BestThreshold = float(EmaValStats.get("threshold", Args.eval_threshold))
                utils.save_on_master(
                    {"model": EmaM.module.state_dict(), "epoch": Epoch},
                    OutputDir / "checkpoint_best.pth",
                )

        # -- Epoch Summary --
        EpochTime = str(datetime.timedelta(seconds=int(time.time() - EpochStart)))
        PrintEpochSummary(Epoch, Args.epochs, TrainStats, ValStats,
                         BestMae, BestEpoch, EpochTime, EmaValStats)

        # -- Log --
        LogEntry = {
            **{f"train_{K}": V for K, V in TrainStats.items()},
            **{f"val_{K}": V for K, V in ValStats.items()},
            "epoch": Epoch,
            "best_mae": BestMae,
            "best_epoch": BestEpoch,
            "epoch_time": EpochTime,
        }
        if EmaValStats is not None:
            LogEntry.update({f"ema_val_{K}": V for K, V in EmaValStats.items()})

        if utils.is_main_process():
            with (OutputDir / "log.jsonl").open("a") as F:
                F.write(json.dumps(LogEntry) + "\n")

    # ---- 训练完成: 加载 best 权重并在 Test 集评估 ----
    BestPath = OutputDir / "checkpoint_best.pth"
    TestStats = None
    if BestPath.exists():
        BestState = torch.load(BestPath, map_location="cpu", weights_only=False)["model"]
        ModelWithoutDdp.load_state_dict(BestState)

        print("\n  Running final evaluation on TEST set with best model ...")
        with torch.inference_mode():
            TestStats = FscdEvaluate(
                Model=Model,
                Criterion=Criterion,
                Postprocess=Postprocess,
                DataLoader=LoaderTest,
                Device=Device,
                Threshold=BestThreshold,
                ThresholdCandidates=[BestThreshold],
                CalibrateThreshold=False,
                Args=FullArgs,
                Label="Test",
                CocoGt=TestCocoGt,
            )

    TotalTime = str(datetime.timedelta(seconds=int(time.time() - TrainStart)))
    PrintFinalResults(BestMae, BestEpoch, TestStats, TotalTime, str(OutputDir))


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def ParseArgs() -> argparse.Namespace:
    """解析命令行参数。"""
    P = argparse.ArgumentParser(description="FSCD Training Script")

    # 必须参数
    P.add_argument("--dataset_dir", type=str, required=True,
                    help="FSCD-147 数据集根目录")

    # 训练超参
    P.add_argument("--epochs", type=int, default=50)
    P.add_argument("--batch_size", type=int, default=2)
    P.add_argument("--lr", type=float, default=5e-5,
                    help="FSCD 新增模块学习率")
    P.add_argument("--lr_encoder", type=float, default=1e-5,
                    help="Backbone 学习率 (应远低于 lr)")
    P.add_argument("--prior_prob", type=float, default=0.03,
                    help="Objectness bias prior probability")
    P.add_argument("--cls_loss_coef", type=float, default=4.0,
                    help="分类损失权重")
    P.add_argument("--focal_alpha", type=float, default=0.5,
                    help="Focal loss alpha")
    P.add_argument("--focal_gamma", type=float, default=1.5,
                    help="Focal loss gamma")
    P.add_argument("--count_loss_coef", type=float, default=3.0,
                    help="计数损失权重")
    P.add_argument("--use_topk_count_loss", action="store_true", default=True,
                    help="仅对 top-k query 求和计算 count loss")
    P.add_argument("--no_topk_count_loss", action="store_true",
                    help="关闭 top-k count loss，改为全 query soft count")
    P.add_argument("--count_topk_factor", type=float, default=3.0,
                    help="top-k = clamp(factor * GT_count)")
    P.add_argument("--count_topk_min", type=int, default=1)
    P.add_argument("--count_topk_max", type=int, default=150)
    P.add_argument("--set_cost_class", type=float, default=4.0,
                    help="Hungarian 匹配分类代价权重")
    P.add_argument("--set_cost_bbox", type=float, default=2.0,
                    help="Hungarian 匹配 bbox L1 代价权重")
    P.add_argument("--set_cost_giou", type=float, default=1.0,
                    help="Hungarian 匹配 GIoU 代价权重")
    P.add_argument("--weight_decay", type=float, default=1e-4)
    P.add_argument("--clip_max_norm", type=float, default=0.1,
                    help="梯度裁剪最大范数")
    P.add_argument("--warmup_epochs", type=int, default=3)
    P.add_argument("--grad_accum_steps", type=int, default=1,
                    help="梯度累积步数, 有效 batch = batch_size * grad_accum_steps")

    P.add_argument("--model_size", type=str, default="base",
                    choices=["base", "large"],
                    help="RF-DETR 模型规模: base (ViT-S, ~33M) / large (ViT-B, ~128M)")
    P.add_argument("--num_exemplars", type=int, default=3)

    # EMA
    P.add_argument("--use_ema", action="store_true", default=True)
    P.add_argument("--ema_decay", type=float, default=0.9997)

    # AMP
    P.add_argument("--amp", action="store_true", default=True,
                    help="混合精度训练")
    P.add_argument("--no_amp", action="store_true")

    # 输出
    P.add_argument("--output_dir", type=str, default="output_fscd")
    P.add_argument("--save_interval", type=int, default=5,
                    help="每 N 个 epoch 保存编号 checkpoint")

    # 恢复
    P.add_argument("--resume", type=str, default="",
                    help="恢复训练的 checkpoint 路径")

    # Eval threshold
    P.add_argument("--eval_threshold", type=float, default=0.3,
                    help="默认 hard-count 阈值")
    P.add_argument("--auto_threshold", action="store_true", default=True,
                    help="在验证集自动扫描阈值并记录最优 hard-MAE")
    P.add_argument("--no_auto_threshold", action="store_true",
                    help="关闭验证集阈值自动扫描")
    P.add_argument("--threshold_min", type=float, default=0.02)
    P.add_argument("--threshold_max", type=float, default=0.60)
    P.add_argument("--threshold_steps", type=int, default=15)

    # 其他
    P.add_argument("--num_workers", type=int, default=2)
    P.add_argument("--seed", type=int, default=42)
    P.add_argument("--device", type=str, default="cuda")

    # DDP
    P.add_argument("--world_size", type=int, default=1)
    P.add_argument("--dist_url", type=str, default="env://")

    Args = P.parse_args()

    if Args.no_amp:
        Args.amp = False
    if Args.no_auto_threshold:
        Args.auto_threshold = False
    if Args.no_topk_count_loss:
        Args.use_topk_count_loss = False
    Args.distributed = False

    return Args


if __name__ == "__main__":
    Args = ParseArgs()
    Train(Args)
