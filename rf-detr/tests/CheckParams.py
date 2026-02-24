"""检查 FSCD 模型各模块参数量和冻结状态。"""
import sys
sys.path.insert(0, ".")

from rfdetr.config import RFDETRFSCDConfig, RFDETRBaseConfig, RFDETRLargeConfig
from rfdetr.FscdMain import FscdModel
from rfdetr.main import download_pretrain_weights

# 构建 FSCD 模型
Cfg = RFDETRFSCDConfig()
download_pretrain_weights(Cfg.pretrain_weights)
Inst = FscdModel(**Cfg.model_dump())
M = Inst.Model

# 按顶层模块统计
from collections import OrderedDict
Mods = OrderedDict()
for N, P in M.named_parameters():
    Mod = N.split(".")[0]
    if Mod not in Mods:
        Mods[Mod] = {"total": 0, "trainable": 0}
    Mods[Mod]["total"] += P.numel()
    if P.requires_grad:
        Mods[Mod]["trainable"] += P.numel()

print()
print(f"{'Module':35s}  {'Total':>12s}  {'Trainable':>12s}  {'Frozen':>12s}  Status")
print("-" * 90)
for K, V in Mods.items():
    Frozen = V["total"] - V["trainable"]
    Status = "ALL TRAINABLE" if Frozen == 0 else "PARTIALLY FROZEN" if V["trainable"] > 0 else "ALL FROZEN"
    print(f"{K:35s}  {V['total']:>12,}  {V['trainable']:>12,}  {Frozen:>12,}  {Status}")

T = sum(V["total"] for V in Mods.values())
R = sum(V["trainable"] for V in Mods.values())
print("-" * 90)
print(f"{'TOTAL':35s}  {T:>12,}  {R:>12,}  {T - R:>12,}")

# 可用 RF-DETR 配置
print()
print("Available RF-DETR configs:")
for Name, Cls in [("Base", RFDETRBaseConfig), ("Large", RFDETRLargeConfig)]:
    C = Cls()
    print(f"  {Name}: resolution={C.resolution}, num_queries={C.num_queries}, pretrain={C.pretrain_weights}")
