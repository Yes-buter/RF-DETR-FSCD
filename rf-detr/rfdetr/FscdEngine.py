# ------------------------------------------------------------------------
# RF-DETR — FSCD Extension
# FSCD 训练与评估引擎
# ------------------------------------------------------------------------

"""
FSCD 专用的 train_one_epoch / evaluate 函数。

与基线 engine.py 的主要差异:
1. 模型 forward 多一个 ExemplarBoxes 参数
2. 评估使用 MAE / RMSE / AP / AP50 四个核心指标
3. collate 需处理 exemplar_boxes 对齐
4. 使用 tqdm 进度条显示训练/评估进度
"""

import math
import sys
from typing import Callable, DefaultDict, Iterable, List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

import rfdetr.util.misc as utils
from rfdetr.util.misc import NestedTensor, nested_tensor_from_tensor_list

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------

def FscdCollateFn(Batch: list) -> tuple:
    """FSCD 专用 collate: 图像 → NestedTensor, target 保持 list[dict]。"""
    Imgs, Targets = list(zip(*Batch))
    NestedImgs = nested_tensor_from_tensor_list(Imgs)
    return NestedImgs, list(Targets)


# ---------------------------------------------------------------------------
# autocast 辅助
# ---------------------------------------------------------------------------

def _GetAutocastArgs(Args) -> dict:
    """与 engine.py 保持一致的 autocast 参数。"""
    if DEPRECATED_AMP:
        return {"enabled": Args.amp, "dtype": torch.bfloat16}
    else:
        return {"device_type": "cuda", "enabled": Args.amp, "dtype": torch.bfloat16}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def FscdTrainOneEpoch(
    Model: torch.nn.Module,
    Criterion: torch.nn.Module,
    LrScheduler: torch.optim.lr_scheduler.LRScheduler,
    DataLoader: Iterable,
    Optimizer: torch.optim.Optimizer,
    Device: torch.device,
    Epoch: int,
    BatchSize: int,
    MaxNorm: float = 0,
    EmaM: torch.nn.Module = None,
    NumTrainingStepsPerEpoch: int = None,
    Args=None,
    Callbacks: DefaultDict[str, List[Callable]] = None,
    TotalEpochs: int = 50,
) -> dict:
    """FSCD 版本的单 epoch 训练。

    核心差异: 从 targets 提取 exemplar_boxes 并传给 model forward。
    使用 tqdm 进度条展示训练进度和关键指标。
    """
    Model.train()
    Criterion.train()
    StartSteps = Epoch * NumTrainingStepsPerEpoch

    if DEPRECATED_AMP:
        Scaler = GradScaler(enabled=Args.amp)
    else:
        Scaler = GradScaler("cuda", enabled=Args.amp)

    Optimizer.zero_grad()
    assert BatchSize % Args.grad_accum_steps == 0
    SubBatchSize = BatchSize // Args.grad_accum_steps

    # 累积指标
    RunningLoss = 0.0
    RunningCe = 0.0
    RunningBbox = 0.0
    RunningGiou = 0.0
    RunningCount = 0.0
    RunningCountError = 0.0
    RunningMatchedQueries = 0.0
    StepCount = 0

    # tqdm 进度条
    Pbar = tqdm(
        enumerate(DataLoader),
        total=NumTrainingStepsPerEpoch,
        desc=f"  Train {Epoch + 1}/{TotalEpochs}",
        bar_format="{l_bar}{bar:25}{r_bar}",
        dynamic_ncols=True,
        leave=True,
    )

    for DataIterStep, (Samples, Targets) in Pbar:
        It = StartSteps + DataIterStep

        if Callbacks:
            CallbackDict = {"step": It, "model": Model, "epoch": Epoch}
            for Cb in Callbacks.get("on_train_batch_start", []):
                Cb(CallbackDict)

        for I in range(Args.grad_accum_steps):
            StartIdx = I * SubBatchSize
            FinalIdx = StartIdx + SubBatchSize

            NewSamplesTensors = Samples.tensors[StartIdx:FinalIdx]
            NewSamples = NestedTensor(
                NewSamplesTensors, Samples.mask[StartIdx:FinalIdx]
            )
            NewSamples = NewSamples.to(Device)

            SubTargets = [
                {K: V.to(Device) for K, V in T.items()} for T in Targets[StartIdx:FinalIdx]
            ]

            # 提取 exemplar_boxes → [SubB, NumExemplars, 4]
            ExemplarBoxes = torch.stack(
                [T["exemplar_boxes"] for T in SubTargets]
            ).to(Device)

            # 归一化到 [0, 1]
            ImgH, ImgW = NewSamplesTensors.shape[-2:]
            ExemplarBoxesNorm = ExemplarBoxes.clone()
            ExemplarBoxesNorm[..., 0::2] /= ImgW
            ExemplarBoxesNorm[..., 1::2] /= ImgH

            with autocast(**_GetAutocastArgs(Args)):
                Outputs = Model(NewSamples, ExemplarBoxesNorm, SubTargets)
                LossDict = Criterion(Outputs, SubTargets)
                WeightDict = Criterion.weight_dict
                Losses = sum(
                    (1 / Args.grad_accum_steps) * LossDict[K] * WeightDict[K]
                    for K in LossDict.keys()
                    if K in WeightDict
                )

            Scaler.scale(Losses).backward()

        LossDictReduced = utils.reduce_dict(LossDict)
        LossDictReducedScaled = {
            K: V * WeightDict[K]
            for K, V in LossDictReduced.items()
            if K in WeightDict
        }
        LossesReducedScaled = sum(LossDictReducedScaled.values())
        LossValue = LossesReducedScaled.item()

        if not math.isfinite(LossValue):
            Pbar.close()
            raise ValueError(f"Loss is {LossValue}, stopping training")

        if MaxNorm > 0:
            Scaler.unscale_(Optimizer)
            torch.nn.utils.clip_grad_norm_(Model.parameters(), MaxNorm)

        Scaler.step(Optimizer)
        Scaler.update()
        LrScheduler.step()
        Optimizer.zero_grad()

        if EmaM is not None and Epoch >= 0:
            EmaM.update(Model)

        StepCount += 1
        RunningLoss += LossValue
        RunningCe += LossDictReduced.get("loss_ce", torch.tensor(0)).item()
        RunningBbox += LossDictReduced.get("loss_bbox", torch.tensor(0)).item() * WeightDict.get("loss_bbox", 1)
        RunningGiou += LossDictReduced.get("loss_giou", torch.tensor(0)).item() * WeightDict.get("loss_giou", 1)
        RunningCount += LossDictReduced.get("loss_count", torch.tensor(0)).item() * WeightDict.get("loss_count", 1)
        RunningCountError += LossDictReduced.get("count_error_unscaled",
                             LossDictReduced.get("count_error", torch.tensor(0))).item()
        RunningMatchedQueries += LossDictReduced.get(
            "matched_queries_per_image", torch.tensor(0)
        ).item()

        # 更新 tqdm postfix
        MemMb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        Pbar.set_postfix_str(
            f"loss={RunningLoss / StepCount:.3f}  "
            f"ce={RunningCe / StepCount:.3f}  "
            f"bbox={RunningBbox / StepCount:.3f}  "
            f"giou={RunningGiou / StepCount:.3f}  "
            f"cnt={RunningCount / StepCount:.2f}  "
            f"mq={RunningMatchedQueries / StepCount:.1f}  "
            f"err={RunningCountError / StepCount:.1f}  "
            f"lr={Optimizer.param_groups[0]['lr']:.1e}  "
            f"mem={MemMb:.0f}M"
        )

    Pbar.close()

    AvgStats = {
        "loss": RunningLoss / max(1, StepCount),
        "loss_ce": RunningCe / max(1, StepCount),
        "loss_bbox": RunningBbox / max(1, StepCount),
        "loss_giou": RunningGiou / max(1, StepCount),
        "loss_count": RunningCount / max(1, StepCount),
        "matched_queries_per_image": RunningMatchedQueries / max(1, StepCount),
        "count_error": RunningCountError / max(1, StepCount),
        "lr": Optimizer.param_groups[0]["lr"],
    }
    return AvgStats


# ---------------------------------------------------------------------------
# 评估 (MAE / RMSE / AP / AP50)
# ---------------------------------------------------------------------------

@torch.no_grad()
def FscdEvaluate(
    Model: torch.nn.Module,
    Criterion: torch.nn.Module,
    Postprocess: torch.nn.Module,
    DataLoader: Iterable,
    Device: torch.device,
    Threshold: float = 0.3,
    ThresholdCandidates: Optional[List[float]] = None,
    CalibrateThreshold: bool = False,
    Args=None,
    Label: str = "Val",
    CocoGt=None,
) -> dict:
    """FSCD 评估: 计算 MAE / RMSE / AP / AP50。

    Args:
        CocoGt: pycocotools.coco.COCO 对象 (val/test 集)。
            如果提供，会额外计算 COCO AP/AP50 检测指标。
    """
    Model.eval()
    Criterion.eval()

    AllPredCounts = []    # soft count (sigmoid sum)
    AllHardCounts = []    # hard count (threshold-based)
    AllGtCounts = []
    TotalLoss = 0.0
    TotalSteps = 0

    CandidateThresholds = (
        [float(Threshold)] if not ThresholdCandidates else [float(T) for T in ThresholdCandidates]
    )
    CandidateThresholds = sorted(set(CandidateThresholds))
    if float(Threshold) not in CandidateThresholds:
        CandidateThresholds.append(float(Threshold))
    CandidateThresholds = sorted(CandidateThresholds)

    HardMaeSums = {T: 0.0 for T in CandidateThresholds}
    HardCountSums = {T: 0.0 for T in CandidateThresholds}
    HardCountNums = {T: 0 for T in CandidateThresholds}

    AutocastArgs = _GetAutocastArgs(Args)

    # COCO evaluator (optional)
    CocoEvaluator = None
    if CocoGt is not None:
        from rfdetr.datasets.coco_eval import CocoEvaluator as _CocoEvaluator
        CocoEvaluator = _CocoEvaluator(CocoGt, ("bbox",))

    NumBatches = len(DataLoader)

    # tqdm 进度条
    Pbar = tqdm(
        enumerate(DataLoader),
        total=NumBatches,
        desc=f"  {Label}",
        bar_format="{l_bar}{bar:25}{r_bar}",
        dynamic_ncols=True,
        leave=True,
    )

    for BatchIdx, (Samples, Targets) in Pbar:
        Samples = Samples.to(Device)
        Targets = [{K: V.to(Device) for K, V in T.items()} for T in Targets]

        # 提取 exemplar_boxes
        ExemplarBoxes = torch.stack(
            [T["exemplar_boxes"] for T in Targets]
        ).to(Device)

        ImgH, ImgW = Samples.tensors.shape[-2:]
        ExemplarBoxesNorm = ExemplarBoxes.clone()
        ExemplarBoxesNorm[..., 0::2] /= ImgW
        ExemplarBoxesNorm[..., 1::2] /= ImgH

        with autocast(**AutocastArgs):
            Outputs = Model(Samples, ExemplarBoxesNorm)

        # Loss
        LossDict = Criterion(Outputs, Targets)
        WeightDict = Criterion.weight_dict
        LossDictReduced = utils.reduce_dict(LossDict)
        BatchLoss = sum(
            V * WeightDict[K] for K, V in LossDictReduced.items() if K in WeightDict
        ).item()
        TotalLoss += BatchLoss
        TotalSteps += 1

        # 计数指标: 使用 soft count (sigmoid 求和), 与 loss_count 训练信号一致
        PredLogits = Outputs["pred_logits"]  # [B, NQ, 1]
        PredScores = PredLogits.sigmoid().squeeze(-1)  # [B, NQ]
        SoftCount = PredScores.sum(dim=1)  # [B] — 不依赖阈值
        HardCount = (PredScores > Threshold).sum(dim=1)  # [B] — 辅助参考
        GtCount = torch.stack([T["count"].squeeze() for T in Targets])  # [B]

        for CandThreshold in CandidateThresholds:
            CandHardCount = (PredScores > CandThreshold).sum(dim=1).float()
            HardMaeSums[CandThreshold] += (CandHardCount - GtCount).abs().sum().item()
            HardCountSums[CandThreshold] += CandHardCount.sum().item()
            HardCountNums[CandThreshold] += CandHardCount.numel()

        AllPredCounts.append(SoftCount.float().cpu())
        AllHardCounts.append(HardCount.float().cpu())
        AllGtCounts.append(GtCount.float().cpu())

        # COCO AP 评估
        if CocoEvaluator is not None:
            OrigTargetSizes = torch.stack(
                [T["orig_size"] for T in Targets], dim=0
            )
            ResultsAll = Postprocess(Outputs, OrigTargetSizes)

            # class-agnostic label 0 → COCO category_id 1
            Res = {}
            for T, R in zip(Targets, ResultsAll):
                ImgId = T["image_id"].item()
                R["labels"] = R["labels"] + 1
                Res[ImgId] = R

            CocoEvaluator.update(Res)

        # tqdm postfix
        CurPredCounts = torch.cat(AllPredCounts)
        CurGtCounts = torch.cat(AllGtCounts)
        CurMae = (CurPredCounts - CurGtCounts).abs().mean().item()
        CurHardCounts = torch.cat(AllHardCounts)
        CurHardMae = (CurHardCounts - CurGtCounts).abs().mean().item()
        Pbar.set_postfix_str(
            f"MAE={CurMae:.2f}  hard_MAE={CurHardMae:.2f}  loss={TotalLoss / TotalSteps:.3f}"
        )

    Pbar.close()

    AllPredCounts = torch.cat(AllPredCounts)
    AllHardCounts = torch.cat(AllHardCounts)
    AllGtCounts = torch.cat(AllGtCounts)

    # MAE, RMSE (基于 soft count)
    AbsErrors = (AllPredCounts - AllGtCounts).abs()
    Mae = AbsErrors.mean().item()
    Rmse = (AbsErrors ** 2).mean().sqrt().item()

    # Hard count MAE (辅助参考)
    HardAbsErrors = (AllHardCounts - AllGtCounts).abs()
    HardMae = HardAbsErrors.mean().item()
    BestThreshold = float(Threshold)
    MeanHardCount = AllHardCounts.mean().item()

    if CalibrateThreshold and len(CandidateThresholds) > 0:
        BestThreshold = min(
            CandidateThresholds,
            key=lambda T: HardMaeSums[T] / max(1, HardCountNums[T]),
        )
        HardMae = HardMaeSums[BestThreshold] / max(1, HardCountNums[BestThreshold])
        MeanHardCount = HardCountSums[BestThreshold] / max(1, HardCountNums[BestThreshold])

    AvgLoss = TotalLoss / max(1, TotalSteps)

    Stats = {
        "loss": AvgLoss,
        "mae": Mae,
        "rmse": Rmse,
        "hard_mae": HardMae,
        "mean_pred_count": AllPredCounts.mean().item(),
        "mean_hard_count": MeanHardCount,
        "mean_gt_count": AllGtCounts.mean().item(),
        "threshold": BestThreshold,
        "ap": 0.0,
        "ap50": 0.0,
    }

    # COCO AP 汇总
    if CocoEvaluator is not None:
        CocoEvaluator.synchronize_between_processes()
        CocoEvaluator.accumulate()
        import contextlib, os
        with open(os.devnull, "w") as Devnull:
            with contextlib.redirect_stdout(Devnull):
                CocoEvaluator.summarize()

        BboxEval = CocoEvaluator.coco_eval["bbox"]
        if BboxEval.stats is not None and len(BboxEval.stats) > 0:
            Stats["ap"] = BboxEval.stats[0]     # AP @ IoU=0.50:0.95
            Stats["ap50"] = BboxEval.stats[1]   # AP @ IoU=0.50

    return Stats
