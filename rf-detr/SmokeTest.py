"""FSCD 训练脚本冒烟测试: 验证所有模块可正确导入和配置合并。"""
import traceback
import sys
import os
import types
import importlib.util

ProjectRoot = os.path.dirname(os.path.abspath(__file__))
if ProjectRoot not in sys.path:
    sys.path.insert(0, ProjectRoot)

# 用 namespace 方式注册 rfdetr，避免触发完整 import chain
RfdetrPkg = types.ModuleType("rfdetr")
RfdetrPkg.__path__ = [os.path.join(ProjectRoot, "rfdetr")]
RfdetrPkg.__package__ = "rfdetr"
sys.modules["rfdetr"] = RfdetrPkg

# 注册子包
for SubPkg in ["datasets", "models", "util", "deploy"]:
    FullName = f"rfdetr.{SubPkg}"
    M = types.ModuleType(FullName)
    M.__path__ = [os.path.join(ProjectRoot, "rfdetr", SubPkg)]
    M.__package__ = FullName
    sys.modules[FullName] = M

try:
    import torch
    print(f"PyTorch: {torch.__version__}")

    # 加载 transforms
    Spec = importlib.util.spec_from_file_location(
        "rfdetr.datasets.transforms",
        os.path.join(ProjectRoot, "rfdetr", "datasets", "transforms.py"),
    )
    Mod = importlib.util.module_from_spec(Spec)
    sys.modules["rfdetr.datasets.transforms"] = Mod
    Spec.loader.exec_module(Mod)
    print("  transforms OK")

    # 加载 Fscd147
    Spec = importlib.util.spec_from_file_location(
        "rfdetr.datasets.Fscd147",
        os.path.join(ProjectRoot, "rfdetr", "datasets", "Fscd147.py"),
    )
    Mod = importlib.util.module_from_spec(Spec)
    sys.modules["rfdetr.datasets.Fscd147"] = Mod
    Spec.loader.exec_module(Mod)
    print("  Fscd147 OK")

    # 加载 misc
    Spec = importlib.util.spec_from_file_location(
        "rfdetr.util.misc",
        os.path.join(ProjectRoot, "rfdetr", "util", "misc.py"),
    )
    Mod = importlib.util.module_from_spec(Spec)
    sys.modules["rfdetr.util.misc"] = Mod
    Spec.loader.exec_module(Mod)
    print("  misc OK")

    # 加载 FscdEngine
    Spec = importlib.util.spec_from_file_location(
        "rfdetr.FscdEngine",
        os.path.join(ProjectRoot, "rfdetr", "FscdEngine.py"),
    )
    Mod = importlib.util.module_from_spec(Spec)
    sys.modules["rfdetr.FscdEngine"] = Mod
    Spec.loader.exec_module(Mod)
    print("  FscdEngine OK")

    # 验证 FscdEngine 接口
    assert hasattr(Mod, "FscdTrainOneEpoch"), "Missing FscdTrainOneEpoch"
    assert hasattr(Mod, "FscdEvaluate"), "Missing FscdEvaluate"
    assert hasattr(Mod, "FscdCollateFn"), "Missing FscdCollateFn"
    print("  FscdEngine 接口验证通过!")

    # 验证 FscdCollateFn
    from rfdetr.datasets.Fscd147 import BuildFscd147
    Ds = BuildFscd147("val", r"C:\Users\20683\Desktop\FSCD147", 560, 3)
    Sample0 = Ds[0]
    Sample1 = Ds[1]
    Batch = Mod.FscdCollateFn([Sample0, Sample1])
    print(f"  Collate: images={Batch[0].tensors.shape}, targets={len(Batch[1])}")
    print(f"  Target keys: {list(Batch[1][0].keys())}")
    assert Batch[0].tensors.shape[0] == 2, "Batch size should be 2"
    assert "exemplar_boxes" in Batch[1][0], "Missing exemplar_boxes in target"
    print("  FscdCollateFn 验证通过!")

    # 验证 config
    from pydantic import BaseModel as PydanticBaseModel
    
    # 手动加载 config.py (完整文件)
    Spec = importlib.util.spec_from_file_location(
        "rfdetr.config",
        os.path.join(ProjectRoot, "rfdetr", "config.py"),
    )
    CfgMod = importlib.util.module_from_spec(Spec)
    sys.modules["rfdetr.config"] = CfgMod
    Spec.loader.exec_module(CfgMod)
    
    FscdCfg = CfgMod.FSCDTrainConfig(dataset_dir=r"C:\Users\20683\Desktop\FSCD147")
    print(f"  FSCDTrainConfig: dataset_file={FscdCfg.dataset_file}, epochs={FscdCfg.epochs}, lr={FscdCfg.lr}")
    assert FscdCfg.dataset_file == "fscd147"
    assert FscdCfg.multi_scale == False
    assert FscdCfg.ia_bce_loss == False
    print("  FSCDTrainConfig 验证通过!")

    print("\n" + "=" * 50)
    print("ALL FSCD TRAINING SCRIPT TESTS PASSED!")
    print("=" * 50)

except Exception as E:
    traceback.print_exc()
    sys.exit(1)
