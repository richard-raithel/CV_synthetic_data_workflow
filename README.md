# Image Synthesis and Computer Vision Training Workflow

## Overview

This repository contains a complete workflow for generating synthetic datasets, training GANs and Mask R-CNN models, and converting PyTorch models to ONNX for deployment. The tools and scripts provided support the creation of aligned datasets, preprocessing images, and training models for advanced computer vision tasks.

---

## Features
- **Synthetic Data Generation**:
  - Create synthetic COCO-style masks (`create_synthetic_coco_masks.py`).
  - Align datasets for model training (`make_dataset_aligned.py`).

- **GAN Training**:
  - Preprocess source images for GANs (`preprocess_source_images_for_gan.py`).
  - Train and test GAN models (`train_gan.py`, `test_gan.py`).
  - Utilities for Pix2Pix GAN training (`pix2pix_utils.py`).

- **Object Detection and Segmentation**:
  - Train Mask R-CNN models for segmentation (`maskrcnn_training.py`).

- **Model Deployment**:
  - Convert PyTorch models to ONNX format for deployment (`pytorch_2_onnx.py`).

---

## Directory Structure
- **`bg-fg/`**: Contains background and foreground assets for dataset generation.
- **`misc/`**: Miscellaneous scripts and helper files.
- **`requirements.txt`**: Dependencies for the project.
- **`sample.jpg`**, **`sample.png`**: Example images for testing.

---

## Requirements

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for training)
- PyTorch 1.x or higher

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-synthesis-workflow.git
   cd image-synthesis-workflow
