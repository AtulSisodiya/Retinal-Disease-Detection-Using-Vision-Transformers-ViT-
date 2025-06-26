# 🩺 Diabetic Retinopathy Severity Classification

![Diabetic Retinopathy](https://via.placeholder.com/800x200?text=Diabetic+Retinopathy+Classification)

A deep learning project to classify retinal fundus images into 5 severity levels of diabetic retinopathy using **ResNet50** and **Vision Transformers (ViT)**, with interpretability through **Grad-CAM** visualizations.

---

## 📁 Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)
- [License](#license)

---

## 📊 Dataset

The dataset contains colored retinal fundus images categorized into five severity levels:

| Severity Level   | Label | Number of Images |
|------------------|-------|------------------|
| No DR            | 0     | XXXX             |
| Mild             | 1     | XXXX             |
| Moderate         | 2     | XXXX             |
| Severe           | 3     | XXXX             |
| Proliferative DR | 4     | XXXX             |

⭐ Key Features

    Custom Dataset Loader from train.csv and folder structure

    Models:

        ResNet50 (ImageNet pretrained, 5-class head)

        Vision Transformer (HuggingFace ViT, fine-tuned)

    Training:

        Cross-entropy loss

        Adam optimizer

        Basic augmentations (e.g., horizontal flip)

    Evaluation:

        Accuracy computation

        Grad-CAM for explainability

📈 Results

    Training loss and accuracy per epoch

    Final model evaluation on dataset

    Grad-CAM visualizations showing regions influencing model predictions

🔮 Future Work

    Add validation/test split

    Tune hyperparameters for better performance

    Integrate other interpretability tools beyond Grad-CAM

    Deploy the model for clinical decision support

📚 References

    PyTorch

    HuggingFace Transformers – ViT

    Grad-CAM Paper

    Kaggle: Diabetic Retinopathy Detection
