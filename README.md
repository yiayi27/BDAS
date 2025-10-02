# BDAS 项目

> 一图一节，带你从 **预处理 → 模型 → 训练 → 评估 → 推理部署** 走一遍完整流程

---

## 目录

* [快速开始](#快速开始)
* [流程总览](#流程总览)
* [1. 数据预处理](#1-数据预处理)
* [2. 数据集划分与增强](#2-数据集划分与增强)
* [3. 模型结构](#3-模型结构)
* [4. 训练与超参](#4-训练与超参)
* [5. 评估指标与可视化](#5-评估指标与可视化)
* [6. 推理与部署](#6-推理与部署)
* [项目结构](#项目结构)
* [环境与安装](#环境与安装)

---

## 快速开始

```bash
# 克隆并进入项目（示例）
# git clone https://github.com/yiayi27/BDAS.git
cd visionPred

# 安装依赖
pip install -r requirements.txt

# 训练
python main.py

# 测试/评估
python test.py
```

---

## 流程总览

```mermaid
flowchart LR
    A[原始数据] --> B[预处理/清洗]
    B --> C[数据划分/增强]
    C --> D[模型构建]
    D --> E[训练与验证]
    E --> F[评估与可视化]
    F --> G[推理/部署]
```

---

## 1. 数据预处理

> 针对白内障患者术后视力预测的任务，我们采用OCT,SLO,Text三种模态数据来做回归预测。其中SLO模态存在约2/3的模态缺失。
![预处理流程图](images/01_preprocess.png)

**要点**

* 缺失值/异常值处理：…
* 尺度统一与坐标对齐：…
* 数值特征标准化 / 图像归一化：…

---

## 2. 数据集划分与增强

> 放置训练/验证/测试划分示意图与增强策略总览图。

![数据划分](images/02_split.png)

![增强策略示例](images/02_augmentations.png)

**要点**

* 划分比例（示例）：`train/val/test = 8/1/1`
* 增强（示例）：随机翻转、旋转、颜色抖动、CutMix/MixUp（若使用）

---

## 3. 模型结构

> 放置网络结构总览图或模块级别框图。

![模型结构示意](images/03_model_arch.png)

**配置示例**

* Backbone：ResNet18 / ResNet50 / ViT…
* 颈部模块（可选）：FPN/Attention…
* Head：回归/分类 Head，激活函数与输出维度说明
* 损失：`L1/L2 + 正则项`（示例）

---

## 4. 训练与超参

> 放置损失曲线/学习率调度图，或训练日志关键截图。

![训练曲线](images/04_training_curve.png)

**超参示例**

* batch_size：32
* learning_rate：1e-3（CosineDecay/StepLR）
* optimizer：AdamW (weight_decay=1e-4)
* epochs：100
* 早停策略：patience=10

---

## 5. 评估指标与可视化

> 放置指标表格截图、混淆矩阵/回归散点图、误差区间占比图。

![评估可视化示例](images/05_metrics.png)

**常用指标（回归）**

* RMSE / MAE / MedAE
* 误差区间占比：`≤ ±0.25d / ≤ ±0.5d / ≤ ±1.0d`
* R²、Pearson/Spearman（若需要）

---

## 6. 推理与部署

> 放置推理流程/接口调用图，或输入→输出样例图。

![推理流程](images/06_inference.png)

**推理示例**

```python
from utils.infer import Predictor
pred = Predictor(model_path="weights/best.pth").run(input_sample)
print(pred)
```

---

## 项目结构

```text
visionPred/
├─ README.md
├─ requirements.txt
├─ main.py
├─ test.py
├─ configs/
├─ dataloader/
├─ loss/
├─ model/
├─ utils/
└─ images/                 # ← 把本文提到的图片放这里
   ├─ 01_preprocess.png
   ├─ 02_split.png
   ├─ 02_augmentations.png
   ├─ 03_model_arch.png
   ├─ 04_training_curve.png
   ├─ 05_metrics.png
   └─ 06_inference.png
```

---

## 环境与安装

* Python：3.9+（示例）
* 关键依赖：`torch` / `torchvision` / `scikit-learn` / `numpy` / `pandas` …
* 一键安装：

```bash
pip install -r requirements.txt
```

> 如果你使用了 CUDA，请在安装 `torch` 时按照官网指引选择与你 CUDA 版本匹配的发行版。

---

### 使用说明（替换图片）

1. 在仓库中创建 `images/` 文件夹（或使用网页端 **Add file → Upload files** 直接上传）。
2. 把你的配图命名为上面列出的文件名；或按你的文件名修改本文中的图片路径。
3. 提交后，GitHub 会自动在 README 中渲染这些图片与流程图（Mermaid）。

