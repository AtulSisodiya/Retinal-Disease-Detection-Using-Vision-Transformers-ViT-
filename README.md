# Diabetic Retinopathy Severity Classification

![Deep Learning for Medical Imaging](https://via.placeholder.com/800x200?text=Diabetic+Retinopathy+Classification)

A deep learning project that classifies retinal fundus images into 5 severity levels of diabetic retinopathy using ResNet50 and Vision Transformers (ViT), with model interpretability via Grad-CAM.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

## Dataset

The dataset contains retinal fundus images classified into 5 categories:

| Severity Level | Label | Number of Images |
|----------------|-------|------------------|
| No DR          | 0     | XXXX             |
| Mild           | 1     | XXXX             |
| Moderate       | 2     | XXXX             |
| Severe         | 3     | XXXX             |
| Proliferate DR | 4     | XXXX             |

File structure:
data/
├── train.csv
└── colored_images/
├── No_DR/
├── Mild/
├── Moderate/
├── Severe/
└── Proliferate_DR/

Installation

Make sure you have Python 3.7+ installed.

Install required packages:

pip install torch torchvision transformers timm pytorch-grad-cam matplotlib pandas

Usage

Place your dataset folder data in the project root as described above.

Run the training and evaluation script:

python diabetic_retinopathy_classification.py

The script will:

Load the dataset with proper transformations

Train ResNet50 and ViT models for diabetic retinopathy classification

Print training loss and accuracy per epoch

Evaluate and compare accuracy of both models

Visualize Grad-CAM heatmaps for sample images from both models for interpretability

Key Components

Custom Dataset: Loads images based on train.csv and matches labels to folder structure.

Models:

ResNet50 (pretrained on ImageNet) with final layer adjusted for 5 classes.

Vision Transformer (ViT) from HuggingFace Transformers, fine-tuned for classification.

Training:

Cross-entropy loss

Adam optimizer

Basic data augmentations like random horizontal flip

Evaluation:

Accuracy calculated on the training set (can be extended to validation/test set)

Interpretability:

Grad-CAM applied to visualize important regions in the images influencing model decisions.

Results

You will get printed outputs of training progress, accuracy scores, and Grad-CAM visualizations for both models.

Future Work

Add proper train/validation split for better generalization.

Hyperparameter tuning for improved performance.

Extend interpretability methods beyond Grad-CAM.

Deploy model for clinical decision support.

References

PyTorch (https://pytorch.org/)

HuggingFace Transformers - ViT (https://huggingface.co/docs/transformers/model_doc/vit)

Grad-CAM (https://arxiv.org/abs/1610.02391)

Diabetic Retinopathy Detection (https://www.kaggle.com/c/diabetic-retinopathy-detection)

Author

Your Name