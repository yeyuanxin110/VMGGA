# Robust Detector-Free Multimodal Image Matching Based on Visual Model Guidance and Gated Attention

<div align="center">
  <img src="assets/framework.png" width="800"/>
</div>

<br/>

This repository contains the official implementation of the paper: **"Robust Detector-Free Multimodal Image Matching Based on Visual Model Guidance and Gated Attention"**.

> **Abstract:** *Multimodal image matching is a crucial foundational task... (You can paste your abstract here)*

## 📢 News & Roadmap
This repository is currently under construction. We are organizing the code and data, which will be released progressively.

- [ ] **Release the Optical-Map Dataset** (A new dataset for optical satellite and vector map matching).
- [ ] **Release Evaluation/Testing Code**.
- [ ] **Release Pre-trained Models**.
- [ ] **Release Full Training Code** (To be released upon formal publication).

## ⚡ Introduction
We propose a robust detector-free multimodal image matching method featuring:
* **Visual Model-Guided Backbone:** Leveraging DINOv3 priors to handle Non-linear Radiometric Differences (NRD).
* **Gated Linear Attention (GLA):** A module to balance receptive field expansion and local feature discriminability.

## 📂 Datasets
We will release the datasets used in our paper, including our self-constructed **Optical-Map Dataset**.
* **Optical-Map:** Contains 2,942 pairs of optical satellite images and vector maps from major Chinese cities.
* *(Links to download will be updated here)*

## 🛠️ Usage
### Environment
*(Placeholder for your requirements, e.g.)*
```bash
conda create -n rrsi python=3.8
pip install -r requirements.txt
