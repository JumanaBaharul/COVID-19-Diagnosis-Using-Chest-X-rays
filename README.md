# COVID-19 Diagnosis Using Chest X-ray Images

## Overview

The project develops a deep learning-based system for detecting COVID-19 from chest X-ray images using Convolutional Neural Networks (CNNs). The model leverages advanced preprocessing techniques and a custom CRF-UNet architecture to classify X-ray images as COVID-19 positive or normal, offering a scalable and rapid diagnostic tool to complement RT-PCR testing in resource-limited settings.

---

## Problem Statement

Manual interpretation of chest X-rays for COVID-19 diagnosis faces challenges such as:

- Requirement for specialized radiological expertise
- Inter-observer variability and potential human error
- Time-consuming processes delaying diagnosis
- Increased workload for radiologists during pandemics

This project addresses these issues by providing an automated, AI-assisted diagnostic system for rapid and accurate preliminary assessments.

---

## Objectives

- Develop a robust CNN model to classify chest X-ray images as COVID-19 positive or normal.
- Implement advanced preprocessing and augmentation techniques to enhance model performance.
- Evaluate the model using multiple performance metrics (accuracy, precision, recall, F1-score, ROC-AUC, etc.).
- Explore model interpretability to understand decision-making processes.
- Assess clinical utility through rigorous validation.

---

## Motivation

- **Timeliness:** Faster diagnosis enables earlier isolation and treatment, reducing transmission.
- **Accessibility:** X-ray equipment is widely available, especially in developing regions.
- **Scalability:** AI systems process large image volumes consistently.
- **Complementary Role:** Supports prioritization for RT-PCR testing.
- **Future Preparedness:** Adaptable for other respiratory diseases and pandemics.

---

## Methodology

### Dataset

- **Source:** COVID-19 Radiography Database (Kaggle)

#### Composition:

- COVID-19 images: 3,616 confirmed cases
- Normal images: 10,192 healthy controls
- Additional classes: Lung Opacity, Viral Pneumonia

#### Preprocessing:

- Image standardization: Resized to 224x224 pixels, normalized to [0,1], converted to RGB.
- Enhanced techniques: Histogram equalization, CLAHE, image complementation, lung mask segmentation, CRF refinement.
- Quality control: Visual inspection and mask alignment verification.

#### Class Imbalance Handling:

- Dynamic class weights
- Five-fold cross-validation
- Train/validation/test split (70/15/15%)

---

### Model Architecture

#### Transfer Learning Models:

- VGG19 and EfficientNetB0 with ImageNet pre-trained weights
- Customized classification head: Global average pooling, dense layer (256 units, ReLU), dropout (0.5), final dense layer (4 units, softmax).

#### Custom CRF-UNet Architecture:

- Encoder: 2 convolutional blocks (64→128 filters) with max pooling.
- Bottleneck: 256-filter convolutional layer.
- Decoder: Upsampling with CRFAttentionBlock (sigmoid attention).
- CRF Attention Block: Single convolutional pathway, spatial softmax, and feature reweighting.

---

### Training Protocol

- Mixed precision training (float16/float32)
- Optimizer: Adam (learning rate = 1e-4)
- Batch size: 32
- Epochs: 10 with early stopping
- Callbacks: Early stopping (patience=5), learning rate reduction, model checkpointing
- Hardware: NVIDIA GPU with TensorFlow acceleration

---

### Evaluation Metrics

#### Standard Metrics:

- Accuracy
- Precision
- Recall
- F1-score

#### Advanced Metrics:

- ROC-AUC
- Precision-Recall AUC
- Cohen’s Kappa
- Matthews Correlation Coefficient

#### Clinical Metrics:

- Positive/Negative predictive values
- Likelihood ratios
- Decision curve analysis

---

## Results

### Model Performance:

#### VGG19 Accuracy: 89.58%  
#### EfficientNetB0 Accuracy: 87.3%

#### Confusion Matrix:

- True Positives (COVID): 504
- False Negatives: 20
- False Positives: 31
- True Negatives (Normal): 1380

#### Key Observations:

- Balanced performance across classes
- False negatives higher than false positives, critical for infectious diseases
- Consistent cross-validation performance (SD < 1.5%)

#### Visual Interpretability:

- Texture features more discriminative than shape
- Model avoids reliance on spurious artifacts
- False negatives common in early-stage COVID cases; false positives linked to other pulmonary abnormalities

---

## Clinical Implications

- **Triage:** Rapid screening in emergency departments and prioritization for RT-PCR testing.
- **Resource-Limited Settings:** Deployable on mobile devices with cloud connectivity, reducing reliance on radiology expertise.
- **Monitoring:** Tracks disease progression and quantifies lung involvement.

---

## Limitations

- **Dataset:** Limited to posterior-anterior views, primarily adult patients, potential selection bias.
- **Technical:** CRF processing increases computational overhead, potential overfitting, memory constraints for high-resolution images.
- **Clinical:** Requires prospective studies, workflow integration, and comparison with radiologist performance.

---

## Future Directions

- **Model Improvements:** End-to-end CRF-integrated pipelines, multi-task learning, ensemble methods, self-supervised pretraining.
- **Technical Enhancements:** CRF hyperparameter optimization, multi-view inputs, automated preprocessing, vision transformers.
- **Clinical Translation:** Randomized trials, standardized reporting, regulatory approval, explainable AI for clinical trust.

---
