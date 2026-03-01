# 🚗💤 Real-Time Driver Drowsiness Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green?logo=google&logoColor=white)](https://mediapipe.dev)
[![OpenCV](https://img.shields.io/badge/OpenCV-Latest-red?logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A real-time computer vision system that detects driver drowsiness and yawning using a hybrid approach — combining **Eye Aspect Ratio (EAR)** analysis with a **CNN deep learning model** — and triggers an audio alarm to prevent accidents.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Development Steps](#-development-steps)
- [Dependencies](#-dependencies)

---

## 🔍 Overview

Driver fatigue is one of the leading causes of road accidents worldwide. This project provides a **non-intrusive, real-time solution** that monitors a driver's face through a webcam and detects signs of drowsiness or yawning. When fatigue is detected, the system immediately triggers a loud alarm to alert the driver.

The system uses a **dual-layer detection strategy**:
1. **EAR (Eye Aspect Ratio)** — geometric analysis of eye openness via MediaPipe face landmarks
2. **CNN Model** — a pre-trained deep learning model (`driver_drowsiness_cnn.h5`) for image-level classification

Results from both are combined using **temporal smoothing** and **majority voting** for robust, low-false-positive detection.

---

## ✨ Features

| Feature | Description |
|---|---|
| 👁️ **EAR Drowsiness Detection** | Detects prolonged eye closure using Eye Aspect Ratio geometry |
| 🤖 **CNN Classification** | Deep learning model classifies driver state from full-face images |
| 😮 **Yawn Detection** | Identifies yawning via CNN mouth prediction with frame-count thresholding |
| 🔊 **Audio Alarm** | Plays a looping emergency alarm when drowsiness or yawning is confirmed |
| 📊 **Real-Time HUD** | Live overlay showing EAR value, closure duration, confidence %, and status |
| 🧠 **Temporal Smoothing** | EAR history window + prediction majority voting reduces false alarms |
| ⚡ **Low Latency** | MediaPipe face mesh runs efficiently for real-time performance |

---

## 🔧 How It Works

```
Webcam Frame
     │
     ▼
MediaPipe Face Mesh ──► Extract 468 3D landmarks
     │
     ├──► LEFT_EYE / RIGHT_EYE landmarks ──► Compute EAR
     │         │
     │         ▼
     │    Smooth over 7-frame window
     │         │
     │         ▼
     │    EAR < 0.25 for ≥ 2 seconds? ──► DROWSY
     │
     └──► Resize frame to 224×224 ──► CNN Model
               │
               ▼
          Majority vote (10-frame window)
               │
               ▼
          Label == 2 + Confidence > 60%
          for ≥ 5 frames? ──► YAWNING
               │
               ▼
         Trigger Alarm 🔊 (if DROWSY or YAWNING)
```

### Detection States

| Status | Condition |
|---|---|
| 🟢 **ALERT** | EAR above threshold, no yawning detected |
| 🔴 **DROWSY** | Eyes closed (EAR < 0.25) for more than 2 consecutive seconds |
| 🟠 **YAWNING** | CNN predicts yawn class with ≥ 60% confidence for ≥ 5 frames |

---

## 📁 Project Structure

```
Alp project 2/
│
├── app.py                          # 🚀 Main application — run this to start detection
│
├── driver_drowsiness_cnn.h5        # 🤖 Pre-trained CNN model weights
├── freesound_community-emergency-alarm-with-reverb-29431.wav  # 🔊 Alarm audio
│
├── requirements.txt                # 📦 Python dependencies
│
├── step1_webcam_test.py            # 🔬 Dev step 1: Webcam feed test
├── step2_face_mesh.py              # 🔬 Dev step 2: MediaPipe face mesh overlay
├── step3_eye_mouth_landmarks.py    # 🔬 Dev step 3: Eye & mouth landmark extraction
├── step4_roi_extraction.py         # 🔬 Dev step 4: Region of Interest (ROI) cropping
│
└── README.md                       # 📖 This file
```

> **Note:** The `Real-time-driver-drowsiness-detection/` folder is excluded from version control (see `.gitignore`).

---

## 🚀 Installation

### Prerequisites

- Python **3.8 or higher**
- A working **webcam**
- macOS / Linux / Windows

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd "Alp project 2"

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
opencv-python
mediapipe==0.10.14
tensorflow>=2.16.1
keras>=3.0.0
numpy
pygame
scipy
```

> **⚠️ Note:** `pygame` and `scipy` are required by `app.py` but may not be listed in `requirements.txt`. Install manually if needed:
> ```bash
> pip install pygame scipy
> ```

---

## ▶️ Usage

### Run the Drowsiness Detector

```bash
python app.py
```

- A window titled **"Real-Time Driver Drowsiness Detection"** will open showing your webcam feed.
- The HUD will display:
  - `Status`: ALERT / DROWSY / YAWNING (color-coded)
  - `EAR`: Current smoothed Eye Aspect Ratio
  - `Eyes Closed`: Duration (seconds) eyes have been closed
  - `Confidence`: CNN model prediction confidence
  - `Yawns`: Yawn event count
- Press **`Q`** to quit.

### Development / Exploration Scripts

Run these step-by-step to understand the pipeline:

| Script | Description | Run Command |
|---|---|---|
| `step1_webcam_test.py` | Verify webcam access | `python step1_webcam_test.py` |
| `step2_face_mesh.py` | View MediaPipe face mesh overlay | `python step2_face_mesh.py` |
| `step3_eye_mouth_landmarks.py` | Visualize eye & mouth landmarks | `python step3_eye_mouth_landmarks.py` |
| `step4_roi_extraction.py` | See extracted eye/mouth ROI crops | `python step4_roi_extraction.py` |

Press **`ESC`** to exit any of the step scripts.

---

## ⚙️ Configuration

Key parameters in `app.py` you can tune:

| Parameter | Default | Description |
|---|---|---|
| `EAR_THRESHOLD` | `0.25` | Eye Aspect Ratio below which eyes are considered closed |
| `DROWSY_TIME` | `2.0` | Seconds eyes must be closed to trigger DROWSY alert |
| `CNN_CONF_THRESHOLD` | `0.6` | Minimum CNN confidence to count a yawn prediction |
| `EAR_WINDOW_SIZE` | `7` | Frames to average for EAR smoothing |
| `PRED_WINDOW` | `10` | Frames for majority-vote on CNN predictions |
| `YAWN_THRESH` | `5` | Consecutive yawn-frames needed to confirm yawning |
| `NO_YAWN_THRESH` | `10` | Consecutive non-yawn frames needed to reset yawn state |

---

## 🛠️ Development Steps

This project was built incrementally through 4 development steps:

1. **Step 1 — Webcam Access**: Verified live video capture using OpenCV
2. **Step 2 — Face Mesh**: Integrated MediaPipe FaceMesh for 468-point landmark detection
3. **Step 3 — Landmark Identification**: Isolated specific eye and mouth landmark indices
4. **Step 4 — ROI Extraction**: Cropped Region of Interest bounding boxes for eyes and mouth
5. **Final — `app.py`**: Combined EAR analysis, CNN model inference, majority voting, alarm system, and real-time HUD

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| `opencv-python` | Latest | Webcam capture & frame rendering |
| `mediapipe` | 0.10.14 | Face mesh landmark detection |
| `tensorflow` | ≥ 2.16.1 | Deep learning backend |
| `keras` | ≥ 3.0.0 | CNN model loading |
| `numpy` | Latest | Numerical computations |
| `scipy` | Latest | Euclidean distance for EAR |
| `pygame` | Latest | Audio alarm playback |

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Built with ❤️ using Python, OpenCV, MediaPipe & TensorFlow</sub>
</div>
