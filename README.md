
# Wids-Skin-Lesion-Classification-OncoNET
Skin lesion classification using CNN , ResNet transfer learning, and transformer inspired patch analysis for Wids Project

The primary objective is to understand how different vision architectures extract and represent medical image features by using deep learning.

## Repository Contents

wids_resnet_transfer_learning_notebook.ipynb
wids_resnet_transfer_learning_notebook
wids_onconet_skin_lesion.ipynb
wids_patch_based_analysis.ipynb
README.md
CNN vs ResNet Performance Comparison Report
Patch Based Analysis .png
Patch Based Analysis 2.png

## Dataset

Dataset: ISIC Dermoscopic Image Dataset
Image Type: RGB dermoscopic images
Classes Used:
  1. Nevus
  2. Melanoma (Invasive, In-situ, NOS)

A subset of the dataset was used due to computational constraints. All images were resized to 224×224 pixels. Basic data augmentation such as random horizontal flipping was applied during training.

## Notebooks Description

1. CNN Training 
 Implements a baseline CNN model from scratch using PyTorch which uses convolutional and fully connected layers. It is trained using CrossEntropyLoss and Adam optimizer.

2. ResNet Transfer Learning 
 Uses pretrained ResNet-18 (ImageNet weights). It is fine-tuned on ISIC dataset and demonstrates the effectiveness of transfer learning in medical imaging.

3. Evaluation 
It re-defines and trains both CNN and ResNet models and evaluates models using accuracy, precision, recall, F1-score, confusion matrix. It also enables direct performance comparison between architectures

4. Patch-Based Analysis
 Transformer inspired exploratory task in which images are divided into fixed-size patches (32×32). It helps explain how Vision Transformers differ from CNNs

## Key Observations

 1.ResNet generally shows better generalization than a CNN trained from scratch
 2.CNNs rely on local receptive fields and inductive bias
 3.Accuracy alone is insufficient for imbalanced medical datasets

## Software Used

1.Python
2.PyTorch
3.Torchvision
4.Scikit-learn
5.Matplotlib
6.Google Colab

## Contact

Siddhant Chowdhary
sidchow25@gmail.com

