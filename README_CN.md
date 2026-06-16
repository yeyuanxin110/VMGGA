# VMGGA

🚧 此页面正在近期更新中，目前尚不完整。

[English](README.md) | [中文](README_CN.md)

## 基于视觉模型引导与门控注意力的鲁棒无检测器多模态图像匹配

[![论文](https://img.shields.io/badge/ISPRS%20JPRS-2026-1f6feb)](https://doi.org/10.1016/j.isprsjprs.2026.05.005)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)
![模态](https://img.shields.io/badge/Modalities-4-2ea44f)

本仓库是论文 **VMGGA** 的官方实现。论文发表于
*ISPRS Journal of Photogrammetry and Remote Sensing*。[[论文链接]](https://doi.org/10.1016/j.isprsjprs.2026.05.005)

**唐腾峰，韩志强，彭韬，陈锦昊，叶沅鑫**

<!--
将总体效果图保存为 assets/teaser.png，并取消下面代码的注释。
<p align="center">
  <img src="assets/teaser.png" width="95%" alt="VMGGA 定性匹配结果">
</p>
-->

## 方法简介

VMGGA 是面向多模态图像匹配的无检测器框架，主要包括：

- **视觉模型引导：** 双流骨干网络融合 DINOv3 高层语义先验与局部几何特征。
- **门控线性注意力：** 使用输入相关的门控机制抑制特征交互中的无关全局信息。
- **由粗到细匹配：** 先建立鲁棒粗匹配，再进行像素级精细定位。

本文在四类多模态场景中进行了实验：

| 模态 | 仓库标识 | 示例目录 |
|---|---|---|
| 光学-红外 | `optical_infrared` | `examples/optical_infrared/` |
| 光学-SAR | `optical_sar` | `examples/optical_sar/` |
| 光学-地图 | `optical_map` | `examples/optical_map/` |
| 光学-深度 | `optical_depth` | `examples/optical_depth/` |

## 定性结果

<!--
请将四类模态效果图保存为：
assets/results_optical_infrared.png
assets/results_optical_sar.png
assets/results_optical_map.png
assets/results_optical_depth.png
-->

|                                                                             光学-红外                                                                             |                                                                                       光学-SAR                                                                                      |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|![光学-红外匹配效果图](assets/result_opt_inf_1.png)<br>基准图: RoadScene/inf/FLIR_05027.jpg<br>实时图: RoadScene/opt/FLIR_05027.jpg<br>模拟变换: 旋转-30°<br>内点数 / 正确匹配点数: 422 / 420|![光学-SAR匹配效果图](assets/result_opt_sar_2.png)<br>基准图: OSdataset/opt/10.png<br>实时图: OSdataset/sar/10.png<br>模拟变换: 旋转15°, 尺度1.2倍, x方向透视收缩0.001, y方向透视收缩0.001<br>内点数 / 正确匹配点数: 125 / 125|

|                                                                                       光学-地图                                                                                       |                                                                                     光学-深度                                                                                      |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| ![光学-地图匹配效果图](assets/result_opt_map.png)<br>基准图: VMGGA-Opt-Map/map/109747_53549.jpg<br>实时图: VMGGA-Opt-Map/opt/109747_53549.jpg<br>模拟变换: 旋转-15°, 尺度1.1倍<br>内点数 / 正确匹配点数: 216 / 227 |![光学-深度匹配效果图](assets/result_opt_dep_1.png)<br>基准图: NYU-DEPTH-V2/depth/1340.jpg<br>实时图: NYU-DEPTH-V2/opt/1340.jpg<br>模拟变换: 旋转10°, 尺度1.1倍, x方向透视收缩1e-5<br>内点数 / 正确匹配点数: 230 / 227 |

## 模型权重

下载对应权重后，按照下表文件名放入 `weights/`。每个 VMGGA 权重均包含完整
推理模型参数，推理时不需要额外加载独立的 DINOv3 初始化权重。

| 模态 | 文件名 | 谷歌云盘 | 百度网盘 |
|---|---|---|---|
| 光学-红外 | `vmgga_optical_infrared.pth` | 待更新 | 待更新 |
| 光学-SAR | `vmgga_optical_sar.pth` | 待更新 | 待更新 |
| 光学-地图 | `vmgga_optical_map.pth` | 待更新 | 待更新 |
| 光学-深度 | `vmgga_optical_depth.pth` | 待更新 | 待更新 |

## 数据集

论文中使用的自制 VMGGA-Opt-Map 数据集将通过外部网盘链接提供。

| 数据集           | 说明             | 谷歌云盘 | 百度网盘 |
|---------------|----------------|---|---|
| VMGGA-Opt-Map | 光学卫星影像与栅格地图图像对 | 待更新 | 待更新 |

## 环境安装

```bash
git clone https://github.com/yeyuanxin110/VMGGA.git
cd VMGGA

conda create -n vmgga python=3.10 -y
conda activate vmgga
```

请根据 CUDA 版本参考
[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装 PyTorch，
然后运行：

```bash
pip install -r requirements.txt
```

当前代码已在 Python 3.10、PyTorch 2.8.0、CUDA 12.6、OpenCV 4.12.0、
einops 0.8.1、Kornia 0.8.1 和 pydegensac 环境中测试。

## Demo 使用

我们提供了四种模态的内置示例图像，只需三步即可体验 VMGGA：

**第一步 — 下载模型权重**

从上方模型权重表格中选择目标模态，下载对应的 `.pth` 文件并放入 `weights/`。

**第二步 — （可选）准备自己的图像**

如需使用自己的图像，放入对应模态的示例目录。以光学-SAR 为例：

```text
examples/optical_sar/
├── image0.png          # 基准图
└── image1.png          # 实时图
```

> 如果跳过此步，程序会自动使用内置示例图像对。

**第三步 — 选择模态，直接运行**

打开 `demo_vmgga.py`，在 `main()` 函数顶部填写必填参数：

```python
# =====================================================================
#  必填 / REQUIRED
# =====================================================================
modality = "optical_sar"                               # ← 在此选择模态

image0 = "examples/optical_sar/image0.png"             # ← 基准图路径
image1 = "examples/optical_sar/image1.png"             # ← 实时图路径

mode = 1                                               # ← 匹配模式

if mode == 1:
    # ── Mode 1 参数（必填）──
    rotate = 15         # 旋转角度（度）
    scale = 1.2         # 尺度因子
    homography_x = 0.0  # 透视收缩 x
    homography_y = 0.0  # 透视收缩 y

elif mode == 2:
    # ── Mode 2 参数（必填）──
    homography_label = "examples/optical_sar/homography.txt"  # 标签文件路径
    invert_label = False                                      # 反转标签方向
```

修改参数后，运行：

```bash
python demo_vmgga.py
```

结果保存至 `demo_result/<modality>_matches.png`。

**三种 Mode 的区别**

| Mode | 适用场景 |
|------|---------|
| `1`（默认） | 图像对大致对齐。程序自动施加旋转、尺度和透视扰动，并可对照已知真值评估匹配精度。 |
| `2` | 图像对未对齐，但你提供了单应矩阵标签文件（`.npy`、`.npz`、`.txt` 或 `.csv`）。需设置 `homography_label` 为标签文件路径。 |
| `3` | 仅查看两张图像的匹配效果，不计算精度指标。所有 RANSAC 内点均以绿色绘制。 |

## 仓库结构

```text
VMGGA/
├── assets/                 # README 图片
├── examples/               # 四类模态示例图像
├── LICENSES/               # 第三方许可证
├── src/
│   ├── config/             # 模型构建代码
│   ├── utils/              # 图像几何变换工具
│   └── vmgga/              # VMGGA 网络
├── weights/                # 下载的模型权重，不提交至 Git
├── demo_vmgga.py
├── requirements.txt
└── README.md
```

## 引用

如果本项目对您的研究有所帮助，请引用：

```bibtex
@article{tang2026vmgga,
  title   = {Robust Detector-Free Multimodal Image Matching Based on Visual Model Guidance and Gated Attention},
  author  = {Tang, Tengfeng and Han, Zhiqiang and Peng, Tao and Chen, Jinhao and Ye, Yuanxin},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {238},
  pages   = {484--496},
  year    = {2026},
  doi     = {10.1016/j.isprsjprs.2026.05.005}
}
```

## 许可证

本项目代码采用 Apache License 2.0。`src/vmgga/backbone/dinov3/` 下由
DINOv3 衍生的文件仍受 DINOv3 License 约束，详见
`LICENSES/DINOv3_LICENSE.md`。

## 联系方式

- 唐腾峰（第一作者）: `ttf@my.swjtu.edu.cn`
- 叶沅鑫 教授（通讯作者）：`yeyuanxin@home.swjtu.edu.cn`
