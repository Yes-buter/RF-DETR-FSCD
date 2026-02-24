# 验证 FSCD-147 数据增强是否正常工作
import random
import torch

from rfdetr.datasets.Fscd147 import BuildFscd147

random.seed(42)
torch.manual_seed(42)

Ds = BuildFscd147("train", r"C:\Users\20683\Desktop\FSCD147", Resolution=560, NumExemplars=3)
print(f"Dataset size: {len(Ds)}")

Errors = 0
for Trial in range(50):
    Idx = random.randint(0, len(Ds) - 1)
    try:
        Img, Tgt = Ds[Idx]
        assert Img.shape == (3, 560, 560), f"Bad shape: {Img.shape}"
        assert Tgt["exemplar_boxes"].shape == (3, 4), f"Bad exemplar shape: {Tgt['exemplar_boxes'].shape}"
        assert (Tgt["exemplar_boxes"] >= 0).all(), f"Negative exemplar coords"
        assert Tgt["boxes"].shape[1] == 4, f"Bad boxes dim"
    except Exception as E:
        print(f"  Sample {Idx} failed: {E}")
        Errors += 1

print(f"Tested 50 random samples. Errors: {Errors}")
if Errors == 0:
    print("ALL PASSED")
