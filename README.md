# 🧠 Hand Gesture & Body Language Analysis System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Solutions-orange)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-ML-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)

## 📖 Overview
This project is an AI-powered system designed to analyze body language in **video interviews**. By tracking hand gestures and movement dynamics, the system assesses whether a candidate appears **Confident (Open)**, **Defensive (Clasped)**, or **Nervous (High Energy)**.

The system utilizes **MediaPipe** for real-time hand landmark extraction and a **Random Forest Classifier** for accurate gesture recognition, enhanced by a custom logic layer for two-hand interaction analysis.

---

## 📂 The Dataset
The model was trained on a custom dataset collected and preprocessed specifically for professional interview scenarios.

### 1. Gesture Classes:
The dataset consists of **4 main classes**:
* **🖐️ Open Hand:** Fingers extended (Indicates openness, honesty, and confidence).
* **✊ Closed Hand:** Fingers curled/Fist (Indicates tension or emphasis).
* **☝️ Pointing:** Index finger extended (Indicates authority or direction).
* **🤝 Clasped Hands:** Hands interlocked (Indicates self-control, anxiety, or "waiting" posture).

### 2. Data Engineering:
* **Input:** Raw images processed via MediaPipe to extract **21 (x, y, z) landmarks** per hand.
* **Normalization:** All coordinates are normalized relative to the **Wrist Position** and scaled by **Hand Size**. This ensures the model is invariant to distance (camera zoom) and position on the screen.

---

## ⚙️ Methodology & Logic
The system operates using a **Hybrid Approach** to ensure maximum accuracy:

1.  **Machine Learning Layer:**
    * A Random Forest model classifies the pose of each hand individually based on landmark geometry.

2.  **Heuristic Logic Layer (The "Smart" Rule):**
    * Calculates the **Euclidean Distance** between the two wrists.
    * **Rule:** If the distance is `< 100px` (adjustable), the system overrides the ML prediction and classifies the gesture as **"Clasped Hands"**. This solves occlusion issues where MediaPipe struggles to see individual fingers.

3.  **Behavioral Analysis (Movement):**
    * Tracks the wrist trajectory over a sliding window of **30 frames**.
    * Calculates the average speed (pixels/frame) to classify the subject's energy level:
        * **Stable/Calm:** Low movement score.
        * **High Energy/Nervous:** High movement score.

---

## 🏆 Model Results
We trained and compared multiple algorithms using Scikit-Learn (80% Train / 20% Test split).

### 📊 Performance Comparison:

| Model | Accuracy | Analysis |
| :--- | :---: | :--- |
| **Random Forest (Selected)** 🥇 | **98.5%** | Robust to noise, handles non-linear relationships best. |


---

## 🚀 Features
* **Real-time Detection:** High-FPS inference on live webcam feeds.
* **Video File Analysis:** Process MP4 files and generate a comprehensive summary report.
* **Dual Hand Support:** Detects and correlates data from both hands simultaneously.
* **Movement Tracking:** Quantifies hand movement to detect fidgeting or nervousness.
* **Robust Normalization:** Works accurately whether the subject is close to or far from the camera.

---

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/YourUsername/Hand-Gesture-Analysis.git](https://github.com/YourUsername/Hand-Gesture-Analysis.git)
cd Hand-Gesture-Analysis
