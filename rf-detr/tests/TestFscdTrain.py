# ---------------------------------------------------------------
# FSCD 训练流水线冒烟测试
# 用合成数据验证完整的 forward + loss + backward 流程
# ---------------------------------------------------------------

"""
冒烟测试: 不依赖真实数据集，使用随机张量模拟一个 batch,
验证 FSCD 模型完整训练流水线是否能正常运行。
"""

import sys
import torch
import torch.nn.functional as F

# 确保项目根目录在 sys.path
sys.path.insert(0, r"c:\Users\20683\Desktop\RFdetr\rf-detr")


def TestFscdForwardAndLoss() -> None:
    """冒烟测试: FSCD 模型 forward + criterion + backward。"""
    print("=" * 60)
    print("FSCD 训练流水线冒烟测试")
    print("=" * 60)

    # ① 导入
    print("\n[1/6] 导入 RFDETRFSCD ...")
    from rfdetr import RFDETRFSCD
    print("  ✓ 导入成功")

    # ② 创建模型
    print("\n[2/6] 创建 RFDETRFSCD 实例 ...")
    Model = RFDETRFSCD()
    print(f"  ✓ 模型创建成功")

    # 获取内部模型
    FscdModel = Model.FscdModelInst.Model
    FscdModel.train()
    Device = Model.FscdModelInst.Device
    print(f"  设备: {Device}")

    # ③ 构造合成数据 (模拟 batch_size=2)
    print("\n[3/6] 构造合成数据 ...")
    BatchSize = 2
    Resolution = Model.ModelConfig.resolution  # 应该是 560
    NumExemplars = 3
    HiddenDim = Model.ModelConfig.hidden_dim

    # 随机图像 (batch)
    Images = torch.randn(BatchSize, 3, Resolution, Resolution).to(Device)

    # 随机 exemplar boxes [B, NumExemplars, 4] (归一化 xyxy)
    ExemplarBoxes = torch.tensor([
        [[0.1, 0.1, 0.3, 0.3], [0.4, 0.2, 0.6, 0.5], [0.7, 0.6, 0.9, 0.9]],
        [[0.2, 0.3, 0.4, 0.5], [0.5, 0.5, 0.7, 0.7], [0.1, 0.7, 0.3, 0.9]],
    ], dtype=torch.float32).to(Device)

    # 合成 targets: 每张图 5 个目标
    Targets = []
    for I in range(BatchSize):
        NumGt = 5
        Boxes = torch.rand(NumGt, 4).to(Device)
        # 确保 cxcywh 格式有效
        Boxes[:, 2:] = Boxes[:, 2:].clamp(min=0.05, max=0.3)
        Boxes[:, :2] = Boxes[:, :2].clamp(min=0.15, max=0.85)
        Targets.append({
            "labels": torch.zeros(NumGt, dtype=torch.int64, device=Device),
            "boxes": Boxes,
            "image_id": torch.tensor([I]),
        })
    print(f"  ✓ 合成 batch: {BatchSize} 张 {Resolution}x{Resolution} 图像, 每张 5 个 GT")

    # ④ Forward
    print("\n[4/6] Forward pass ...")
    from rfdetr.util.misc import nested_tensor_from_tensor_list
    Samples = nested_tensor_from_tensor_list(Images)

    with torch.cuda.amp.autocast(enabled=True):
        Outputs = FscdModel(Samples, ExemplarBoxes, Targets)

    print(f"  ✓ Forward 成功")
    print(f"  pred_logits: {Outputs['pred_logits'].shape}")
    print(f"  pred_boxes:  {Outputs['pred_boxes'].shape}")
    if "aux_outputs" in Outputs:
        print(f"  aux_outputs: {len(Outputs['aux_outputs'])} 层")
    if "enc_outputs" in Outputs:
        print(f"  enc_outputs: pred_logits={Outputs['enc_outputs']['pred_logits'].shape}")

    # ⑤ Criterion (loss 计算)
    print("\n[5/6] 计算 Loss ...")
    from rfdetr.main import populate_args
    from rfdetr.models.FscdDetr import BuildFscdCriterionAndPostprocessors

    Args = populate_args(**Model.ModelConfig.dict())
    Criterion, Postprocess = BuildFscdCriterionAndPostprocessors(Args)

    with torch.cuda.amp.autocast(enabled=True):
        LossDict = Criterion(Outputs, Targets)

    print(f"  ✓ Loss 计算成功")
    for K, V in LossDict.items():
        if isinstance(V, torch.Tensor):
            print(f"    {K}: {V.item():.4f}")

    # 加权总 loss
    WeightDict = Criterion.weight_dict
    TotalLoss = sum(
        LossDict[K] * WeightDict[K]
        for K in LossDict.keys()
        if K in WeightDict
    )
    print(f"  总加权 Loss: {TotalLoss.item():.4f}")

    # ⑥ Backward
    print("\n[6/6] Backward pass ...")
    TotalLoss.backward()
    print(f"  ✓ Backward 成功")

    # 验证梯度存在
    GradCount = 0
    TotalParams = 0
    for Name, Param in FscdModel.named_parameters():
        TotalParams += 1
        if Param.grad is not None:
            GradCount += 1

    print(f"  有梯度的参数: {GradCount}/{TotalParams}")

    print("\n" + "=" * 60)
    print("✓ 冒烟测试全部通过！FSCD 训练流水线正常工作。")
    print("=" * 60)


if __name__ == "__main__":
    TestFscdForwardAndLoss()
