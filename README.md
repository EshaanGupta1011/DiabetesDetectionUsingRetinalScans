# DiabetesDetectionUsingRetinalScans

## Overview
Diabetic retinopathy is a leading cause of blindness in adults. Early detection via automated analysis of retinal fundus images can enable timely intervention. This project builds a multi-class classification pipeline that grades diabetic retinopathy severity (5 levels: `class_0` to `class_4`) on the Messidor dataset using custom preprocessing and an ensemble of five binary classifiers.

---

## 🗂 Dataset
- **Source:** [Messidor](http://www.adcis.net/en/third-party/messidor/)  
- **Total images:** 1,700  
- **Classes:**  
  - `class_0` – No retinopathy  
  - `class_1` – Mild  
  - `class_2` – Moderate  
  - `class_3` – Severe  
  - `class_4` – Proliferative  

Images were organized into folders `class_0/`, `class_1/`, …, `class_4/` according to the CSV labels provided.

---

## 🔬 Preprocessing Pipeline
To counteract the variability and low contrast of raw fundus scans, each image undergoes:
1. **Gaussian Blurring**  
   Smooths noise while preserving edges.  
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**  
   Enhances local contrast to reveal microaneurysms and hemorrhages.  
3. **Gamma Correction**  
   Adjusts brightness non‑linearly for improved feature visibility.  
4. **Canny Edge Detection**  
   Extracts prominent vessel and lesion boundaries.  
5. **Morphological Operations**  
   — Dilation & erosion to refine edges and remove isolated artifacts.

Each step is implemented in OpenCV and packaged into a reusable `preprocess_image()` function.

---

## ⚙️ Model Architecture & Ensemble Strategy
Rather than a single 5‑way classifier, we train **five binary CNNs** (one per class) using TensorFlow/Keras:

- **Model₀** detects `class_0` vs. rest  
- **Model₁** detects `class_1` vs. rest  
- …  
- **Model₄** detects `class_4` vs. rest  

During inference, a test image is passed through all five models; the class whose model outputs the highest softmax confidence is selected as the final prediction.

---

## 🚀 Training Details
- **Hardware:** Single GPU with 4 GB VRAM  
- **Batch size:** 16 (limited by memory)  
- **Epochs:** 20 per model  

> **Note:** Due to resource constraints, batch size and number of epochs were kept modest. Despite this, the ensemble approach provided robust class separation under limited compute.

![20051020_45110_0100_PP](https://github.com/user-attachments/assets/c8e54e0f-044c-42b5-a83f-7018f38da0c5)
Above is an example of pre processed class 4 image.
