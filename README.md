## Automatic License Plate Recognition (ALPR):

This project implements a complete **Automatic License Plate Recognition (ALPR)** system — capable of detecting, segmenting, and recognizing vehicle license plates from images.  
It combines **Deep Learning (YOLO models)** for license plate detection and **Machine Learning (SVM with HOG features)** for character recognition.

The system was developed as part of *Challenge 1* in the Vision & Learning course to explore how computer vision and learning methods can be applied to real-world problems such as vehicle monitoring, traffic control, and parking management.

---

## 🧩 Pipeline Structure

The ALPR pipeline consists of four main stages:

1. **Data Exploration and Image Acquisition**
   - Analysis of provided (`real_plates.zip`) and custom-acquired datasets.
   - Evaluation of color, brightness, and viewpoint distribution.
   - Establishment of an image acquisition protocol for consistent data collection.

2. **Detection / Localization**
   - Comparison of different models:
     - `YOLOv5` and `YOLOv8` (trained on COCO dataset)
     - `LP-Detection Model` (YOLOv8-based, trained for license plates)
     - A traditional ML approach using morphological and gradient-based operations.
   - Final detection and cropping of license plates from full car images.

3. **Character Segmentation**
   - Perspective correction using four-point transformation.
   - Segmentation with contour-based detection and watershed methods.
   - Preprocessing (black-hat, contrast enhancement) and postprocessing (Gaussian blur, Otsu thresholding) to improve mask quality.

4. **Recognition**
   - Feature extraction with **Histogram of Oriented Gradients (HOG)**.
   - Training two **SVM classifiers**:
     - One for digits (0–9)
     - One for alphabet characters (A–Z)
   - Prediction and reconstruction of full license plates.

---

## ⚙️ Technologies Used

| Component | Method / Library |
|------------|------------------|
| Object Detection | YOLOv5 / YOLOv8 (Ultralytics) |
| Plate Detection | MKgoud License Plate Recognizer (YOLOv8-based) |
| Image Processing | OpenCV, scikit-image |
| Feature Extraction | HOG, LBP, Block Binary Pixel Sum |
| Classifiers | scikit-learn (SVM, KNN, MLP) |
| Visualization | Matplotlib, seaborn |
| Environment | Python 3.10+, Jupyter Notebooks, LaTeX report |

---

## 📊 Datasets

- **real_plates.zip:** Provided dataset with car images (frontal and lateral views).  
- **example_fonts.zip:** Synthetic dataset of Spanish plate fonts, containing:
  - `digitsIms.pkl` and `alphabetIms.pkl` (cropped characters)
  - Corresponding labels (`digitsLabels`, `alphabetLabels`)
- **Captured Images:** Additional dataset collected under defined protocol:
  - Consistent brightness and angles
  - Different car colors for higher diversity

---

## 🧪 Experiments and Evaluation

1. **YOLOv8 on COCO cars** – AP/AR metrics for various IoU thresholds.  
2. **Confidence threshold effect** – Analysis of duplicate detections vs. missing objects.  
3. **Binary classifier comparison** – SVM, KNN, MLP using HOG/LBP/BBPS features.  
4. **SVM per-class recall** – Visualization of recognition accuracy per character/digit.  

Evaluation metrics include:
- **Precision**, **Recall**, **F1 Score**, **AUC**, and **Confidence Intervals (CIs)**.

---

## 🧠 Results Summary

- YOLOv8 showed the best generalization for car detection, though struggled with small objects.  
- The **LP-Detection Model** (fine-tuned YOLOv8) gave the most accurate plate localizations.  
- SVM classifiers achieved **perfect recall on digits** and **near-perfect recall on letters**, though limited by dataset size.  
- Misclassifications mainly occurred in visually similar letters (e.g., ‘J’ vs. ‘Y’).  
- Combined pipeline successfully reconstructed most real license plates.

---

## 🚀 Future Work

- Expand and diversify training data for SVM classifiers.  
- Apply **data augmentation** to improve model generalization.    
- Integrate watershed segmentation into the complete pipeline.  
- Explore deep learning–based recognition models (CNNs) for improved robustness.
