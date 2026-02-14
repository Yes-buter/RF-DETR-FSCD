# RF-DETR-FSCD: Few-Shot Counting and Detection

## <a name="Weights"></a>权重文件 (Weights)
💡 Note: 算力限制当前提供的权重仅训练了 9 个 Epoch。随着训练轮数的增加，模型性能有望进一步提升。
通过网盘分享的文件：checkpoint0009.pth
链接: https://pan.baidu.com/s/1qjRXuc6sIl0WlkLGKzSyiA?pwd=wfei 提取码: wfei


## <a name="visual-results"></a>🖼️ 可视化展示 (Visual Results)

仅需 **1 个示例框 (Single Exemplar Box)**，模型即可在极少样本条件下完成全图计数与检测。


|  |  |  |
| :---: | :---: | :---: |
| ![1](demo_pic/1.png) | ![2](demo_pic/2.png) | ![3](demo_pic/3.png) |
| ![4](demo_pic/4.png) | ![5](demo_pic/5.png) | ![6](demo_pic/6.png) |
| ![7](demo_pic/7.png) | ![8](demo_pic/8.png) | ![9](demo_pic/9.png) |
| ![10](demo_pic/10.png) | ![11](demo_pic/11.png) | ![12](demo_pic/12.png) |

> *注：黄色框为用户提供的 Exemplar，橙色框为模型检测出的目标。*

---

## <a name="introduction"></a>📖 简介 (Introduction)

**RF-DETR-FSCD** 是一个 **Class-Agnostic（类别无关）** 的少样本计数检测器。它在原始 RF-DETR (ICLR 2026) 的基础上进行了深度改造，使其不再局限于 COCO 80 类，而是能够根据用户提示（Prompts）动态检测任意类别的目标。

### 核心特性

* **One-Shot Counting:** 给定一张图片和 **1 个示例框**，自动计数同类物体。
* **Soft Count Optimization:** 引入基于 Sigmoid 响应和的计数损失，大幅提升密集场景下的计数精度。
* **Real-time Performance:** 继承 RF-DETR 的高效架构，支持实时推理。

---

## <a name="model-architecture"></a>🏗️ 核心架构 (Model Architecture)

模型从 `LWDETR` 演进为 `FscdLWDETR`，主要包含以下改造：

### 1. Exemplar Prototype Extractor

从用户提供的 **单 Exemplar Box** 中提取特征：

* **RoI Align:** 在最高分辨率特征图上截取 7x7 特征。
* **Projection:** 经 `Linear` + `LayerNorm` 投影，生成代表该类别的 **Prototype** 向量。

### 2. Exemplar Conditioning Module (FiLM)

将 Prototype 注入 Transformer Decoder：

* 生成缩放因子 () 和偏移量 ()。
* 对多尺度特征图执行 **FiLM (Feature-wise Linear Modulation)** 操作：。
* 使 Decoder 在"知道要找什么"的条件下工作。

### 3. Prototype-enhanced Objectness

改造分类头以适应 Class-Agnostic 任务：

* **Query 增强:** 计算 Transformer Query 与 Prototype 的 **余弦相似度 (Cosine Similarity)**。
* **最终得分:** 。
* **分类头变更:** 输出维度从 80 (COCO) 变为 1 (Binary Objectness)。

---

## <a name="loss-functions"></a>📉 损失函数 (Loss Functions)

| 损失项 | 原始 RF-DETR | **RF-DETR-FSCD** | 说明 |
| --- | --- | --- | --- |
| **Classification** | Multi-class Focal | **Binary Focal Loss** | , 仅区分前景/背景。 |
| **Counting** | N/A | **L1 Soft Count** | 。 |
| **Regression** | L1 + GIoU | L1 + GIoU | 保持不变。 |

---

## <a name="installation"></a>⚙️ 安装与准备 (Installation)

```bash
# 1. 克隆代码
git clone https://github.com/yourusername/rf-detr-fscd.git
cd rf-detr-fscd

# 2. 安装环境
pip install -r requirements.txt
# 编译 CUDA 算子 (如果需要)
pip install -e .

```

### 数据集准备 (FSCD-147)

请确保数据集目录结构如下，并更新 `config.py` 中的路径：

```text
data/
  FSCD147/
    images_560x560/  # 统一 Resize 到 560x560
    annotation/
      instances_val.json
      instances_test.json

```

---

## <a name="training"></a>🚀 训练 (Training)

使用 `TrainFscd.py` 启动训练。模型会自动加载 `rf-detr-base.pth` 预训练权重（跳过不匹配的分类头和新增模块）。

### 关键配置

* **学习率策略:**
* Backbone (DINOv2): `1e-5`
* Decoder / Heads: `5e-5`
* **Schedule:** 3 Epoch Warmup + Cosine Decay。


* **Group DETR:** 训练时使用 13 组 Queries (3900个) 加速收敛，推理时仅用 300 个。

```bash
python TrainFscd.py \
  --config-file config/rf_detr_fscd_config.py \
  --coco_path /path/to/fscd147 \
  --output_dir output/fscd_v1 \
  --batch_size 4

```

---

## <a name="inference"></a>📊 推理与评估 (Inference)

### 1. 交互式 Demo

启动 GUI 界面，通过鼠标画 **1个框** 进行实时计数：

```bash
python DemoFscd.py \
  --resume output/fscd_v1/checkpoint.pth \
  --image_path assets/test_image.jpg

```

### 2. 精度评估

计算 MAE (Mean Absolute Error) 和 RMSE。评估时采用 **Soft Count** (sigmoid求和) 以获得更鲁棒的结果。

```bash
python EvalFscd.py \
  --coco_path /path/to/fscd147 \
  --resume output/fscd_v1/checkpoint.pth \
  --eval_set test

```

---

<a name="future-work"></a>📅 后续计划 (Future Work)
当前版本展示了 RF-DETR 在少样本计数任务上的潜力，但仍有优化空间。我们计划在未来版本中通过以下方式进一步提升性能：

[ ] 优化边界框损失 (Refine BBox Loss): 计划改进回归损失函数（如引入 CIoU 或 DIoU），以进一步提升检测框的定位精度 (IoU)。

[ ] 引入 NMS 后处理 (Post-processing): 探索在推理阶段加入非极大值抑制 (Non-Maximum Suppression)，以减少密集场景下的重叠预测框。

[ ] 增加训练轮次 (Longer Training): 目前提供的权重仅训练了 9 个 Epoch。我们计划进行更长时间的训练，以充分挖掘模型潜力并提升泛化能力。

## Acknowledgement

* Base model: [RF-DETR](https://www.google.com/search?q=https://github.com/roboflow/rf-detr)
* Dataset: [FSCD-147](https://www.google.com/search?q=https://github.com/VisWan/FSC-147)
