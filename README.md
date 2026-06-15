# VMGGA

[English](README.md) | [简体中文](README_CN.md)

## Robust Detector-Free Multimodal Image Matching Based on Visual Model Guidance and Gated Attention

[![Paper](https://img.shields.io/badge/ISPRS%20JPRS-2026-1f6feb)](https://doi.org/10.1016/j.isprsjprs.2026.05.005)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C)
![Modalities](https://img.shields.io/badge/Modalities-4-2ea44f)

Official implementation of **VMGGA**, published in *ISPRS Journal of
Photogrammetry and Remote Sensing*.

**Tengfeng Tang, Zhiqiang Han, Tao Peng, Jinhao Chen, and Yuanxin Ye**

[[Paper]](https://doi.org/10.1016/j.isprsjprs.2026.05.005)
[[Weights: Google Drive (TBA)]](#model-zoo)
[[Weights: Baidu Netdisk (TBA)]](#model-zoo)

<!--
Add assets/teaser.png and uncomment the following block.
<p align="center">
  <img src="assets/teaser.png" width="95%" alt="VMGGA qualitative results">
</p>
-->

## Overview

VMGGA is a detector-free framework for robust multimodal image matching.
It combines:

- **Visual Model Guidance:** a dual-stream backbone fusing DINOv3 semantic
  priors with local geometric features.
- **Gated Linear Attention:** an input-dependent gate that suppresses
  irrelevant global information during feature interaction.
- **Coarse-to-fine matching:** robust coarse correspondence followed by
  pixel-level refinement.

The method is evaluated on four multimodal settings:

| Modality | Repository key | Example directory |
|---|---|---|
| Optical-Infrared | `optical_infrared` | `examples/optical_infrared/` |
| Optical-SAR | `optical_sar` | `examples/optical_sar/` |
| Optical-Map | `optical_map` | `examples/optical_map/` |
| Optical-Depth | `optical_depth` | `examples/optical_depth/` |

## Qualitative Results

<!--
Replace the placeholders below with:
assets/results_optical_infrared.png
assets/results_optical_sar.png
assets/results_optical_map.png
assets/results_optical_depth.png
-->

| Optical-Infrared | Optical-SAR |
|:---:|:---:|
| *Figure coming soon* | *Figure coming soon* |

| Optical-Map | Optical-Depth |
|:---:|:---:|
| *Figure coming soon* | *Figure coming soon* |

## Model Zoo

Download a checkpoint and place it in `weights/` using the filename below.
Each VMGGA checkpoint contains the complete inference model state; a separate
DINOv3 initialization checkpoint is not required for inference.

| Modality | Filename | Google Drive | Baidu Netdisk |
|---|---|---|---|
| Optical-Infrared | `vmgga_optical_infrared.pth` | TBA | TBA |
| Optical-SAR | `vmgga_optical_sar.pth` | TBA | TBA |
| Optical-Map | `vmgga_optical_map.pth` | TBA | TBA |
| Optical-Depth | `vmgga_optical_depth.pth` | TBA | TBA |

## Installation

```bash
git clone https://github.com/yeyuanxin110/VMGGA.git
cd VMGGA

conda create -n vmgga python=3.10 -y
conda activate vmgga
```

Install PyTorch for your CUDA version from the
[official installation guide](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.10, PyTorch 2.8.0, CUDA 12.6,
OpenCV 4.12.0, einops 0.8.1, Kornia 0.8.1, and pydegensac.

## Demo

Place a sample pair under the selected modality directory:

```text
examples/optical_sar/
├── image0.png
└── image1.png
```

Edit the configuration block in `main()`:

```python
mode = 3
modality = "optical_sar"
```

Then run:

```bash
python demo_vmgga.py
```

The visualization is saved to `demo_result/<modality>_matches.png`.

### Demo Modes

| Mode | Input | Geometric verification |
|---|---|---|
| `1` | Pre-aligned pair plus simulated rotation, scale, and perspective parameters | Yes |
| `2` | Unaligned pair plus a known 3x3 homography label | Yes |
| `3` | Any two input images without a geometric label | No |

For mode 1, edit:

```python
mode_options = {
    "rotate": 15,
    "scale": 1.2,
    "homography_x": 0.0,
    "homography_y": 0.0,
}
```

For mode 2, provide `examples/<modality>/homography.txt`. The matrix must map
coordinates from `image0` to `image1`. Set `invert_homography_label=True` when
the stored matrix has the opposite direction. Text, CSV, NPY, and NPZ formats
are supported.

For modes 1 and 2, green lines indicate correct matches and red lines indicate
matches exceeding the configured reprojection threshold. For mode 3, all
RANSAC inliers are shown in green because no ground-truth homography is
available. Yellow circles mark the matched points.

## Repository Structure

```text
VMGGA/
├── assets/                 # README figures
├── examples/               # Sample pairs for four modalities
├── LICENSES/               # Third-party license files
├── src/
│   ├── config/             # Inference model configuration
│   ├── utils/              # Image transformation utilities
│   └── vmgga/              # VMGGA network
├── weights/                # Downloaded checkpoints (not tracked by Git)
├── demo_vmgga.py
├── requirements.txt
└── README.md
```

## Citation

If this work is useful for your research, please cite:

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

## License

The project code is released under the Apache License 2.0. DINOv3-derived
files under `src/vmgga/backbone/dinov3/` remain subject to the DINOv3 License;
see `LICENSES/DINOv3_LICENSE.md`.

## Contact

- Tengfeng Tang: `ttf@my.swjtu.edu.cn`
- Yuanxin Ye: `yeyuanxin@home.swjtu.edu.cn`
