# Robust Detector-Free Multimodal Image Matching Based on Visual Model Guidance and Gated Attention

<div align="center">
  <img src="assets/framework.png" width="800"/>
</div>

<br/>

This repository contains the official implementation of the paper: **"Robust Detector-Free Multimodal Image Matching Based on Visual Model Guidance and Gated Attention"**.

> **Abstract:** *Multimodal image matching is a crucial foundational task for multi-source information fusion. However, severe geometric distortions and Non-linear Radiometric Differences (NRD) pose significant challenges to robust matching, such as the poor repeatability of keypoints. While detector-free methods emerge to mitigate this, current approaches relying on global context capture inadvertently aggregate irrelevant background noise, leading to reduced feature discriminability. To address these issues, this paper proposes a multimodal image matching method based on Visual Model Guidance and Gated Attention. First, we design a dual-stream backbone network that performs multi-scale fusion of high-level semantic priors from a visual foundation model with local geometric features from standard convolutions. This effectively enhances feature robustness against NRD and guides the model to focus on content-consistent regions. Second, to resolve the feature over-smoothing caused by global interactions, we propose a Gated Linear Attention (GLA) module. By employing a dynamic gating mechanism to filter out irrelevant global background noise, this module maximizes the preservation of local feature discriminability while incorporating effective long-range dependencies. Finally, we construct a coarse-to-fine multi-stage matching pipeline that cascades robust coarse matching with precise pixel-level refinement, achieving high-precision correspondence. Extensive experiments on four multimodal datasets (Optical-Infrared, Optical-SAR, Optical-Map, and Optical-Depth) demonstrate that our method achieves state-of-the-art performance, significantly outperforming existing detector-based and detector-free baselines in terms of accuracy and robustness.*

## 📢 News & Roadmap
This repository is currently under construction. We are organizing the code and data, which will be released progressively.

- [ ] **Release the Optical-Map Dataset** (A new dataset for optical satellite and vector map matching).
- [ ] **Release Evaluation/Testing Code**.
- [ ] **Release Pre-trained Models**.
- [ ] **Release Full Training Code** (To be released upon formal publication).

<!--
## ⚡ Introduction
We propose a robust detector-free multimodal image matching method featuring:
* **Visual Model-Guided Backbone:** Leveraging DINOv3 priors to handle Non-linear Radiometric Differences (NRD).
* **Gated Linear Attention (GLA):** A module to balance receptive field expansion and local feature discriminability.
-->

## 📂 Datasets
We will release the datasets used in our paper, including our self-constructed **Optical-Map Dataset**.
* **Optical-Map:** Contains 2,942 pairs of optical satellite images and vector maps from major Chinese cities.
* *(Links to download will be updated here)*

<!--
## 🛠️ Usage
### Environment
*(Placeholder for your requirements, e.g.)*
```bash
conda create -n rrsi python=3.8
pip install -r requirements.txt
-->
