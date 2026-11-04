# Malayalam Handwritten Character Recognition using Kolmogorov-Arnold Networks (KAN)

A Deep Learning approach for recognizing handwritten Malayalam characters using a hybrid **Convolutional Neural Network (CNN) and Kolmogorov-Arnold Network (KAN)** architecture.

---

## 1. Overview

Malayalam is a complex script with a large character inventory, intricate curved glyph structures, and numerous visually similar characters. Variations in handwriting styles, stroke thickness, character shapes, and writing patterns make handwritten Malayalam character recognition a challenging task.

Traditional CNN-based recognition systems commonly use fully connected Multi-Layer Perceptron (MLP) layers for classification. However, fixed activation functions in MLPs can make it difficult to capture the fine-grained, non-linear differences between visually similar Malayalam characters.

This project proposes a hybrid **CNN+KAN architecture**, replacing the conventional MLP classification head with a Kolmogorov-Arnold Network. Unlike traditional MLPs, KANs use learnable spline-based activation functions on network edges, allowing more flexible function approximation.

The proposed model combines efficient convolutional feature extraction with spline-based classification to recognize **119 classes of isolated handwritten Malayalam characters**.

## 2. Objectives

* Develop an efficient deep learning model for handwritten Malayalam character recognition.
* Explore the application of Kolmogorov-Arnold Networks in Malayalam OCR.
* Improve the classification of visually similar characters using learnable B-spline activation functions.
* Design a lightweight CNN backbone with effective feature extraction capabilities.
* Compare conventional linear classification heads with different KAN configurations through a controlled ablation study.
* Evaluate the model in terms of recognition accuracy, parameter efficiency, and training stability.

## 3. Proposed Architecture

The proposed architecture consists of two major components:

### A. Convolutional Neural Network (CNN) Backbone

The CNN backbone extracts meaningful visual features from handwritten character images.

It incorporates:

* **Depthwise-Separable Convolutions:** Reduce computational complexity and parameter count while preserving spatial feature extraction.
* **Residual Connections:** Improve information flow and support stable training across convolutional stages.
* **Squeeze-and-Excitation (SE) Attention:** Recalibrate channel-wise feature responses to emphasize informative visual patterns.
* **Global Average Pooling:** Converts the extracted feature maps into a compact 256-dimensional feature embedding.

The backbone uses four channel widths: 32, 64, 128, and 256.

### B. Kolmogorov-Arnold Network (KAN) Classification Head

The extracted 256-dimensional feature embedding is passed to a two-layer KAN classification head.

Unlike conventional MLPs, which use fixed activation functions at individual neurons, KANs employ learnable univariate functions on the edges connecting network nodes.

The proposed KAN head uses:

* Two KAN layers.
* Architecture: `[256, 128, 119]`.
* Grid size: 8.
* Cubic B-spline functions (order 3).
* SiLU residual basis functions.
* Batch normalization, dropout, and tanh before the KAN head to maintain stable spline-domain inputs.

The final layer produces predictions across the 119 Malayalam character classes.

### Architecture Flow

```text
Input Character Image (64 × 64)
              |
              v
       CNN Stem (3×3)
              |
              v
   Depthwise-Separable Conv
              |
       Residual + SE
              |
              v
       CNN Stage 1 (64)
              |
              v
       CNN Stage 2 (128)
              |
              v
       CNN Stage 3 (256)
              |
              v
     Global Average Pooling
              |
              v
      256-D Feature Embedding
              |
              v
    BatchNorm + Dropout + Tanh
              |
              v
       KAN Layer 1 (256 → 128)
              |
              v
       KAN Layer 2 (128 → 119)
              |
              v
          Softmax
              |
              v
   Predicted Malayalam Character
```

## 4. Dataset

The model is trained using a combined dataset assembled from two publicly available Kaggle collections:

1. [Handwritten Malayalam Dataset](https://www.kaggle.com/datasets/amaljossy/handwritten-malayalam)
2. [Malayalam Hand-Written Letters Dataset](https://www.kaggle.com/datasets/manikantanrnair/malayalam-hand-written-letters)

The datasets contain handwritten samples belonging to different Malayalam character categories. Additional samples from the supplementary dataset are mapped to their corresponding character classes and merged with the primary dataset.

### Dataset Statistics

| Description       | Details   |
| ----------------- | --------- |
| Total Images      | 5,506     |
| Number of Classes | 119       |
| Training Set      | 70%       |
| Validation Set    | 20%       |
| Testing Set       | 10%       |
| Image Resolution  | 64 × 64   |
| Image Format      | Grayscale |

The dataset is organized into class-labelled training, validation, and testing directories using the standard ImageFolder structure.

## 5. Data Preprocessing and Augmentation

The following preprocessing operations are applied to prepare the handwritten character images:

* **Alpha Channel Extraction:** Extracts the ink mask from the original RGBA images.
* **Binarization:** Converts the extracted ink mask into binary pixel values.
* **Grayscale Conversion:** Converts images into single-channel grayscale format.
* **Image Resizing:** Resizes all images to 64 × 64 pixels.
* **Normalization:** Rescales pixel values to the [0, 1] range.

### Data Augmentation

To improve the model's ability to handle variations in handwriting, the following augmentations are applied to the training set:

* **Morphological Transformations:** Random dilation and erosion to simulate variations in character stroke thickness.
* **Affine Transformations:** Random rotation, translation, scaling, and shear to introduce variations in character orientation and positioning.
* **Random Erasing:** Removes small rectangular image regions to improve robustness against incomplete or partially obscured strokes.

### Class Imbalance Handling

Since handwritten character datasets naturally contain unequal numbers of samples across different classes, a `WeightedRandomSampler` is used during training.

Sample weights are calculated using the square-root dampened inverse class frequency:

`Weight ∝ n_c^(-0.5)`

This helps improve the representation of underrepresented character classes during training without excessively oversampling rare classes.

## 6. Model Training

The model is trained end-to-end using the following configuration:

| Parameter               | Configuration    |
| ----------------------- | ---------------- |
| Input Resolution        | 64 × 64          |
| CNN Backbone Channels   | 32, 64, 128, 256 |
| KAN Architecture        | 256 → 128 → 119  |
| B-Spline Grid Size      | 8                |
| B-Spline Order          | 3 (Cubic)        |
| Dropout                 | 0.3              |
| Batch Size              | 128              |
| Optimizer               | AdamW            |
| Initial Learning Rate   | 0.001            |
| Weight Decay            | 0.0001           |
| Learning Rate Scheduler | OneCycleLR       |
| Label Smoothing         | 0.05             |
| Gradient Clipping       | 1.0              |
| Maximum Epochs          | 200              |
| Training Strategy       | Early Stopping   |

Additional training techniques include Automatic Mixed Precision (AMP) and gradient-norm clipping to improve computational efficiency and training stability.

The best model checkpoint is selected based on validation accuracy.

## 7. Ablation Study

A controlled ablation study was conducted to evaluate the contribution of KAN classification heads compared with a conventional linear baseline.

Six configurations were trained for 15 epochs each while keeping the CNN backbone, optimizer, data augmentation, and sampling strategy unchanged.

### Experimental Comparison

| Classification Head               | Validation Accuracy | Test Accuracy |
| --------------------------------- | ------------------: | ------------: |
| Linear Baseline                   |              11.79% |        10.70% |
| KAN (1 Layer, Grid 5)             |              14.64% |        16.82% |
| KAN (1 Layer, Grid 8)             |              14.07% |        14.53% |
| KAN (2 Layers, Grid 5)            |              14.92% |        17.58% |
| KAN (2 Layers, Grid 8)            |          **15.87%** |    **17.89%** |
| KAN (2 Layers, Grid 8, Hidden 64) |              13.21% |        15.14% |

The results demonstrate that all tested KAN configurations outperform the linear baseline under the same short training budget.

The two-layer KAN with grid size 8 achieved the highest validation and test accuracy among the evaluated configurations and was selected for full training.

## 8. Final Model Performance

The selected CNN+KAN configuration was trained for up to 200 epochs with early stopping.

The model achieved the following results on the held-out test dataset:

| Evaluation Metric           |                     Result |
| --------------------------- | -------------------------: |
| Number of Character Classes |                        119 |
| Best Validation Accuracy    |                     82.60% |
| Test Accuracy               |                 **82.72%** |
| Total Parameters            | Approximately 0.69 Million |
| Best Validation Epoch       |                        130 |
| Early Stopping Epoch        |                        155 |

The final model demonstrates the effectiveness of combining convolutional feature extraction with learnable spline-based classification for isolated Malayalam handwritten character recognition.

## 9. Technologies and Libraries

* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch
* **Model Architecture:** CNN, KAN
* **Numerical Computing:** NumPy
* **Data Processing:** Pandas
* **Visualization:** Matplotlib
* **Image Processing:** PIL, Torchvision

## 10. Future Scope

Although the proposed model achieves promising results in isolated Malayalam character recognition, several improvements can be explored:

* Extending recognition from isolated characters to complete Malayalam words and text lines.
* Integrating the model into a complete Malayalam OCR pipeline.
* Improving generalization through larger and more balanced handwritten datasets.
* Exploring KAN pruning and symbolic interpretability to understand learned character representations.
* Investigating additional regularization and augmentation techniques to improve recognition of rare and visually similar characters.
* Developing applications for Malayalam document digitization and handwritten archival preservation.

## 11. Authors

**M. R. Lekshmi Priya**, Angel George, Athulya Mannambath, and Aswin V. S.

School of Digital Science
Kerala University of Digital Sciences, Innovation and Technology
Thiruvananthapuram, India.

**Research Paper:** *Malayalam Handwritten Character Recognition Model using Kolmogorov Arnold Networks* (August 2026).

---

*This project explores the application of Kolmogorov-Arnold Networks to handwritten Malayalam character recognition, with a focus on efficient feature extraction and improved classification of complex character patterns.*
