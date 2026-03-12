<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./images/figure1_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./images/figure1_light.png">
    <img alt="CADT" src="./images/figure1_light.png" width="600" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
    <a href="https://www.mindspore.cn/"><img alt="MindSpore" src="https://img.shields.io/badge/MindSpore-2.0+-blue?logo=mindspore"></a>
    <a href="https://e.huawei.com/en/products/computing/ascend"><img alt="Ascend 310B" src="https://img.shields.io/badge/Hardware-Ascend_910B/310B-red"></a>
    <a href="https://github.com/QwenLM/Qwen2.5"><img alt="Qwen LLM" src="https://img.shields.io/badge/Cognitive_Engine-Qwen_32B-purple"></a>
    <a href="https://github.com/your-repo/Drowning-Detection-System/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue"></a>
</p>



<h4 align="center">
    <p>
        <b>简体中文</b> | <a href="i18n/README_en.md">English</a> 
    </p>
</h4>

<p align="center">
  <a href="https://sklnst.bupt.edu.cn/index.htm" target="_blank"> <img src="images\logo\bupt.png" alt="网络与交换技术全国重点实验室" width="800" height="auto" style="border: none;">
  </a>
</p>


<h3 align="center">
    <p>“零溺之盾”——跨介质多模态溺水感知与无人化应急救援系统</p>
</h3>

<p align="left">
  <a> <img src="images/result.gif" alt="" width="800" height="auto" style="border: none;">
  </a>
</p>

本项目构建了 **"水下-水面-人体"** 三位一体的立体化感知网络，结合小型无人水面艇（USV）、无人潜航器（UUV）以及医疗级生物传感贴片，实现了实时捕捉并跨介质对齐视觉、声呐与生理（ECG/HRV/SpO₂）等多模态数据。

作为系统的核心，我们自主研发了 **Cross-modal Aquatic Distress Transformer (CADT)** 模型，实现了视觉-声呐-生理三模态信息的非对称深度交互与互监督，确保对隐蔽性被动溺水或突发心脏骤停的**零漏报**。在认知与执行层，系统基于 **Qwen (通义千问)** 边缘量化大模型进行语义推理，驱动 **DR-MAPPO** 多智能体调度引擎，在“黄金4分钟”内自主完成无人艇协同救援闭环。

<p align="center">
  <a> <img src="images\CADT.png" alt="CADT" width="500" height="auto" style="border: none;">
  </a>
</p>

------

## 🛠️ Installation / 环境安装

本系统针对 CPU/GPU 训练环境与 Ascend (昇腾) NPU 边缘部署环境进行了深度解耦。要求 Python 3.9+ 及 [MindSpore](https://www.mindspore.cn/install) 2.0+。

首先，克隆本仓库并进入项目目录：

Shell

```
git clone https://github.com/your-repo/Drowning-Detection-System.git
cd Drowning-Detection-System
```

创建虚拟环境并安装核心依赖：

Bash

```
# 创建并激活 conda 环境
conda create -n cadt_env python=3.10
conda activate cadt_env

# 安装项目依赖 (MindSpore, Transformers, Numpy 等)
pip install -r requirements.txt
```

*(可选)* 若要在边缘端运行极致加速推理，请确保当前硬件已安装昇腾 CANN 驱动及 `mindspore_lite`。

------

## 🚀 Quickstart / 快速开始

我们提供了一套高度封装的脚本，涵盖了从数据生成、模型训练到边缘部署的全生命周期。

## 💾 数据集准备

我们截取了部分数据集，大小约11GB，可在链接中找到

## 🏋️ 模型训练

一键启动 CADT 模型训练 (支持数据下沉模式)

Bash

```
bash scripts/train.sh
```

## 📤 模型导出 (Ascend NPU)

为了在算力受限的无人艇机载电脑（Ascend 310B）上运行，需要将动态图模型编译固化为硬件无关的 `.mindir` 格式。

Bash

```
# 导出静态计算图用于 NPU 部署
python deployment/export_mindir.py \
    --ckpt_path results/checkpoints/cadt_model.ckpt \
    --output_name cadt_edge_model
```

## 🏃 启动边缘守护进程

在部署目标设备上，启动全局主控链路。该进程会自动执行硬件 API 监听、CPU 信号去噪预处理、NPU 毫秒级推理以及异常状态下的大模型调度。

Bash

```
# 启动 7x24 小时实时检测守护进程
python edge_daemon.py
```

*注：首次触发紧急救援阈值时，系统会自动将 Qwen2.5-0.5B-Instruct 模型权重缓存至本地。*

------

## ✨ Why should I use this system? / 核心优势

- **👁️ 解决传统单模态视觉盲区**：针对野外水域低能见度、光照变化等恶劣条件，引入 UUV 声呐与医疗贴片，实现视觉失效下的跨介质数据互补。
- **💻 边缘侧极致性能优化**：模型不依赖云端算力。核心感知网络 CADT 通过 `mindspore_lite` 转化为 FP16 精度静态图，在昇腾 310B 上实现单帧数十毫秒级的超低延迟。
- **🧠 LLM 赋能的认知决策 (Cognitive Engine)**：抛弃传统的 `if-else` 硬编码规则。由 Qwen 大模型消化复杂的环境上下文（如高海浪、心率异常），输出确定性的 JSON 救援策略。
- **🤖 即插即用的多智能体调度**：无缝集成 DR-MAPPO RL 调度引擎，驱动 USV (无人水面艇) 投放 AED 及 UUV 协同探测，真正实现从感知到执行的自主闭环。

------

## 📂 Project Structure / 系统架构树

本系统遵循严谨的高内聚、低耦合工业级标准组织代码：

Plaintext

```
Drowning-Detection-System/
├── 📁 data/                       # 感知层：数据预处理与接口模块
│   ├── 📄 dataset.py                # MindSpore O(1) 内存流式数据加载器
│   └── 📄 alignment.py              # 异构传感器 API 接入与时间滑动窗口对齐
├── 📁 models/                       # 算法层：核心模型库
│   ├── 📄 cadt_transformer.py       # CADT 主干网络拓扑
│   └── 📁 layers/                   # 解耦的底层跨模态 Attention 与 Fusion 算子
├── 📁 agents/                       # 认知与调度层
│   └── 📄 qwen_engine.py            # Qwen LLM 推理适配器与 DR-MAPPO 调度引擎
├── 📁 deployment/                   # 部署层 (针对昇腾 310B 边缘节点)
│   ├── 📄 export_mindir.py          # 静态计算图 (MindIR) 导出脚本
│   ├── 📄 acl_inference.py          # 基于 NPU 的轻量化推理类 (带精度防线)
│   └── 📄 preprocess_edge.py        # 边缘侧 CPU 信号清洗与归一化
├── 📁 configs/                      # 配置中心 (YAML格式)
├── 📁 scripts/                      # 工具集 (数据生成、评估、一键训练)
└── 📄 edge_daemon.py                # 🚀 全局守护进程 (端侧总控入口)
```

------

## 🤝 Funding / 资助

本项目的研究与开发工作得到了以下机构与科研基金的鼎力支持与资助：

- 🌟 北邮-华为横向科研项目: "面向昇腾NPU的高性能动态Shape算子设计与实现"
- 🌟 北京邮电大学研究生创新创业项目（国家级）
- 🌟 华为ICT大赛云计算资源
- 🌟 北京邮电大学 华为学院
- 🌟 网络与交换技术全国重点实验室 网络基础服务中心

------
## ☁️ Powered By <a href="https://sklnst.bupt.edu.cn/index.htm" target="_blank"><img src="images\logo\huaweicloud.png" alt="网络与交换技术全国重点实验室" width="200" style="border: none; vertical-align: middle; margin-left: 10px;"></a>

------

## 📜 Citation / 引用

如果本系统对您的研究或比赛有所帮助，请考虑引用：

> Zhang, Y., Yifan, Y., Xiang, X., & Xue, W. (2026). CADT: A MindSpore-Based Cross-Modal Transformer for Autonomous Water Rescue at the Edge. Zenodo. https://doi.org/10.5281/zenodo.18975782

```
@misc{zhang_2026_18975782,
  author       = {Zhang, Yuanjian and
                  Yifan, Yuan and
                  Xiang, Xu and
                  Xue, Wenqi},
  title        = {CADT: A MindSpore-Based Cross-Modal Transformer
                   for Autonomous Water Rescue at the Edge
                  },
  month        = mar,
  year         = 2026,
  doi          = {10.5281/zenodo.18975782},
  url          = {https://doi.org/10.5281/zenodo.18975782},
}
```