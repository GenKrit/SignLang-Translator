#  Sign Language Translator using Computer Vision and Deep Learning

> A comprehensive, real-time Sign Language Translator developed as a Final Year Capstone Project. Built with Python, MediaPipe, and TensorFlow, it recognizes static hand gestures (A–Z and 0–9) and translates them into spoken English text using a lightweight machine learning model and TTS.

---

## 📸 Preview Section


* GUI Screenshot:
* <img width="1825" height="1029" alt="Screenshot 2025-07-07 224250" src="https://github.com/user-attachments/assets/5c1b0611-eb5f-4aba-8198-cb5fdb235562" />




* Live Detection:
* <img width="373" height="373" alt="Screenshot 2025-07-04 074226" src="https://github.com/user-attachments/assets/42692250-dd20-4feb-b420-365dd0d06061" />
*<img width="936 height="702" alt="Screenshot 2025-07-05 082703" src="https://github.com/user-attachments/assets/6bfbe3c6-0230-48bb-b1a9-4ae3aa4838c8" />



* Confusion Matrix: 

---

## Abstract

This project aims to create an accessible and real-time tool that bridges the communication gap between sign language users and non-signers. It uses computer vision to detect hand landmarks and a machine learning model to classify static gestures. These gestures are converted into text and then to speech, facilitating smoother interactions.

---

##  Project Structure

```
Sign-Language-Recognition/
|
|│-- app2_words_core.py               # Core real-time inference logic
|│-- gui_app-gTTS.py                  # GUI application with TTS output
|│-- model/
|│   ├-- keypoint_classifier/
|│       ├-- keypoint_classifier.py
|│       ├-- keypoint_classifier.h5
|│       ├-- keypoint_classifier.tflite
|│       └-- keypoint_classifier_label.csv
|│-- keypoint_classification.ipynb   # Training notebook
|│-- utils/
|│   └-- cvfpscalc.py             # FPS calculator utility
|│-- requirements.txt
|│-- README.md
|│-- .gitignore
```

---

##  Key Features

| Feature                     | Description                                              |
| --------------------------- | -------------------------------------------------------- |
| Static Gesture Recognition  | Supports all 26 letters (A-Z) and 10 digits (0–9)        |
| Real-Time Feedback          | <65ms latency end-to-end, tested on mid-range system     |
| GUI Support                 | PyQt5 GUI with live preview and sentence construction    |
| Text-to-Speech (TTS)        | Uses gTTS to speak built sentence aloud                  |
| Gesture Stability Mechanism | Prevents jitter and false positives                      |
| Keyboard Controls           | Space (add letter), Enter (add word), Backspace (delete) |

---

## 🔍 Technologies Used

| Category         | Tools/Libraries                |
| ---------------- | ------------------------------ |
| Programming Lang | Python 3.9 (Anaconda)          |
| Vision           | MediaPipe Hands (Google)       |
| ML Framework     | TensorFlow 2.16 + Keras        |
| GUI              | PyQt5                          |
| TTS              | gTTS (Google Text-to-Speech)   |
| Editor           | PyCharm (After VS Code issues) |
| OS               | Windows 10/11                  |

---

## 🎨 GUI Overview

* Live webcam feed with gesture overlay
* Predicted letter display
* Word buffer display
* Sentence construction preview
* Speak button for final output
* Reset/Clear options
* gTTS toggle for automatic playback

---

## 🌐 Dataset Preparation

* 36-class classification (A–Z, 0–9)
* Used MediaPipe to extract (x, y) landmarks of 21 hand points
* Each sample has 42 features + 1 label column
* Data collected using `app-10+.py` into CSV format

```csv
label,x1,y1,x2,y2,...,x21,y21
0,0.63,0.45,0.61,0.42,...,0.70,0.39
```

* Augmented with rotation and scaling (optional)

---

## 🔧 Model Training

**Notebook**: `keypoint_classification.ipynb`

* **Input**: 42 (x, y) hand keypoint values
* **Output**: One of 36 classes
* **Architecture**:

```plaintext
Input (42 nodes)
└➜ Dropout(0.2)
 └➜ Dense(20, ReLU)
     └➜ Dropout(0.4)
         └➜ Dense(10, ReLU)
             └➜ Dense(36, Softmax)
```

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* EarlyStopping with patience=20
* Saved as `.h5` and `.tflite` with quantization

---

## 🔮 Evaluation Metrics

| Metric             | Result              |
| ------------------ | ------------------- |
| Accuracy (overall) | \~71%               |
| Inference Time     | \~3 ms (model only) |
| Pipeline Latency   | \~65 ms total       |
| Model Size         | \~50 KB (.h5)       |
| Class Count        | 36                  |

* Confusion matrix and per-class accuracy available in thesis

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/Genkrit/SignLang-translator.git
cd SignLang-Translator
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

##  Run the App

### Option 1: GUI Mode (recommended)

```bash
python gui_app-gTTS.py
```

### Option 2: CLI Detection (no GUI)

```bash
python app2_words_core.py
```

### Model Training (optional)

```bash
jupyter notebook keypoint_classification_EN.ipynb
```

---

##  Keyboard Shortcuts in GUI

| Key       | Function                     |
| --------- | ---------------------------- |
| SPACE     | Add predicted letter to word |
| ENTER     | Add word to sentence         |
| BACKSPACE | Delete last letter           |
| CTRL+C    | Quit                         |

---


##  License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

## 🚀 Roadmap / Future Work

* ✅ Add dynamic gesture support
* ✅ Integrate ASL grammar rules
* ✅ Add multilingual support (e.g. Hindi, Tamil)
* ✅ Improve model accuracy with larger dataset
* ✅ Mobile app using TFLite (Android)

---

## 🔗 Useful Links

* [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
* [TensorFlow](https://www.tensorflow.org/)
* [PyQt5 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
* [gTTS Docs](https://gtts.readthedocs.io/en/latest/)

---

