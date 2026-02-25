# 🏭 VisionSpec QC – Visual Quality Control System

## 📌 Project Overview

**VisionSpec QC** is an AI-powered Visual Quality Control System designed for real-time defect detection in Printed Circuit Boards (PCBs) on a production assembly line.

The system classifies each captured image as:

- ✅ **PASS**
- ❌ **DEFECT**

Additionally, it provides visual interpretability using **Grad-CAM**, highlighting the exact defect region to ensure transparency and reliability in industrial deployment.

---
## Live Demo Link:
  [https://visionspec-qc-visual-quality-control-fbow6bzwr2cjkqmhcpev2v.streamlit.app/]

## 🎯 Problem Statement

In PCB manufacturing, 100% inspection is mandatory to detect soldering defects. Manual inspection is:

- Time-consuming  
- Error-prone  
- Expensive  

This project builds a real-time AI inspection system capable of:

- High-accuracy classification  
- Low-latency inference  
- Defect localization  
- Production simulation  

---

## 🏗️ System Architecture

### 🔹 Core Stack

- TensorFlow / Keras  
- OpenCV  
- MobileNetV2 (Transfer Learning)  
- Grad-CAM (Explainable AI)  

---

### 🔹 Model Pipeline

Input Image (224x224)
↓
MobileNetV2 (Pretrained – Frozen Base)
↓
Global Average Pooling
↓
Dense (128) + Dropout
↓
Sigmoid Output (Binary Classification)
↓
PASS / DEFECT


---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (ImageNet Pretrained)
- **Custom Head:**
  - GlobalAveragePooling2D
  - Dense (128, ReLU)
  - Dropout (0.5)
  - Dense (1, Sigmoid)

✔ Functional API implementation  
✔ Optimized for speed (low latency)  

---

## 📊 Training Results

| Metric | Value |
|--------|--------|
| Training Accuracy | ~99% |
| Validation Accuracy | ~98% |
| Test Accuracy | ~98%+ |
| Precision | High |
| Recall | High |
| F1-Score | Strong |

Learning curves indicate:

- Minimal overfitting  
- Stable convergence  
- Strong generalization  

---

## 🔥 Grad-CAM (Explainable AI)

Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented to visualize:

- Where the model focuses  
- Whether it detects actual defect regions  
- Transparency in decision-making  

### Why It Matters:

In industrial AI systems, explainability is critical.  
Grad-CAM ensures the model is not relying on background noise but actual soldering defects.

✔ Heatmap overlays generated  
✔ Defect localization validated  
✔ Black-box problem mitigated  

---

## 🎥 Real-Time Inference Engine

The system includes a production simulation using:

- Webcam live feed  
- Frame-by-frame classification  
- Real-time overlay of:
  - Label (PASS / DEFECT)  
  - Confidence score  
  - Inference latency (ms)  

### Additional Production Features:

- Automatic saving of defective frames  
- Timestamped defect logging  
- Synthetic conveyor-belt video simulation  

---

## ⚡ Inference Performance

- Average Latency: ~15–30 ms per frame  
- Optimized for real-time deployment  
- Suitable for edge-device conversion (TensorFlow Lite)  

---

## 📂 Project Structure

<img width="313" height="185" alt="image" src="https://github.com/user-attachments/assets/f4e55020-b23d-45b9-9b3a-bd637cd314d0" />


---

## 🚀 Key Features

✔ Transfer Learning with MobileNetV2  
✔ Data Augmentation  
✔ High-Accuracy Binary Classification  
✔ Grad-CAM Explainability  
✔ Real-Time Inference  
✔ Defect Image Auto-Saving  
✔ Production Video Simulation  

---

## 🏭 Industrial Relevance

This system simulates:

- PCB assembly line inspection  
- Automated visual QC  
- Smart manufacturing (Industry 4.0)  

---

## 🧑‍💻 Author

**Aniket Dombale**  
AI & Machine Learning Intern  

VisionSpec QC – Production ML


