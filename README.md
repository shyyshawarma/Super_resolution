# Image Super Resolution

**Author:** Abhinav  
**Project Repository:** [GitHub Link](https://github.com/shyyshawarma/Super_resolution)  
**Dataset Sample:** [Google Drive Link](https://drive.google.com/drive/folders/1Yj33rbX_OF2P7xSC4wrh-k1Zb-lnP1uO?usp=share_link)

## Overview

This project aims to generate high-resolution images from low-resolution counterparts using deep learning techniques. It implements and compares several super-resolution models: **SRCNN**, **SwinIR**, and **SRGAN**.

> **Super-Resolution (SR)** is the process of reconstructing high-resolution (HR) images from low-resolution (LR) inputs, commonly used in fields like medical imaging, satellite imagery, and more.

---

## Models Implemented

### 1. **SRCNN**  
- Based on one of the first CNN architectures for SR.  
- Uses 3 convolution layers.  
- Evaluated using MSE and PSNR.  
- **PSNR:** 23.45 dB  
- **SSIM:** 0.5452  
- **Optimization:** Used **Adam** optimizer instead of traditional SGD.

### 2. **SwinIR**  
- Based on Swin Transformer architecture.  
- Incorporates Residual Swin Transformer Blocks (RSTBs) and self-attention mechanisms.  
- Shows improvement in both local and global context modeling.  
- **PSNR:** 25.67 dB  
- Progressive improvement observed through epochs (22.70 → 25.67 dB).

### 3. **SRGAN**  
- GAN-based approach using perceptual loss with VGG feature maps.  
- Trained with a 16-block ResNet (SRResNet).  
- **PSNR:** 26.24 dB  
- **SSIM:** 0.6162  
- Produces perceptually sharper images despite lower PSNR at times.

---

## Dataset & Preprocessing

- Original: ~80 training images, 20 test images.  
- After augmentation (rotation, flipping, jitter, crop):  
  - **Training:** 800 images  
  - **Validation:** 100 images  
  - **Testing:** 20 images  

**Important Notes:**
- HR images are augmented; LR images are derived via Gaussian blur + bicubic interpolation.
- Image scaling ensures 4× super-resolution consistency.
- Some degradation pipelines inspired by BSRGAN are considered.

---

##T raining Details

- Epoch-wise model saving and evaluation using PSNR.
- SwinIR training took ~3–4 hours due to hardware limitations.
- YAML configuration and hyperparameter tuning posed challenges.

---

## Comparison Summary

| Model   | PSNR (dB) | SSIM   | Remarks                              |
|---------|-----------|--------|--------------------------------------|
| SRCNN   | 23.45     | 0.5452 | Basic, shallow network               |
| SwinIR  | 25.67     | -      | Transformer-based, improved context |
| SRGAN   | 26.24     | 0.6162 | Best perceptual quality              |

> SRGAN shows superior visual quality, as it’s optimized for perceptual loss rather than traditional pixel-based metrics.

---

## Challenges Faced

1. ESRGAN implementation difficulties due to batch norm removal and RRDB blocks.
2. Writing YAML files and hyperparameter tuning.
3. Long training times and hardware/cloud resource constraints.

---

## Future Work

- Explore **DRCT-L** and **Transformer-based** models further.
- Applications in:
  - Low-light lunar image enhancement using Chandrayaan-2 OHRC data.
  - Polar area imaging: deraining, dehazing.

More on this: [Papers with Code - Set14 SOTA](https://paperswithcode.com/sota/image-super-resolution-on-set14-4x-upscaling)

---

## References

- [SRCNN Paper](https://arxiv.org/pdf/1501.00092)
- [SwinIR Paper](https://arxiv.org/pdf/2108.10257)
- [SRGAN Paper](https://arxiv.org/pdf/1609.04802)
- [SRGAN Tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution)
- [KAIR - SwinIR Repo](https://github.com/cszn/KAIR)

---

## Sample Results

| Bicubic | SRCNN | SwinIR | SRGAN | Ground Truth |
|---------|-------|--------|-------|--------------|
| ![bicubic](path) | ![srcnn](path) | ![swinir](path) | ![srgan](path) | ![gt](path) |

---

