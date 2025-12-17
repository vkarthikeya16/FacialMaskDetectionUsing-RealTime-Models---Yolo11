
# 🛡️ MaskGuard — Real-Time Face Mask Detection System (YOLOv11)

MaskGuard is a real-time face mask compliance monitoring system built using YOLOv11.  
It detects and classifies faces into **Mask**, **No Mask**, and **Incorrect Mask** categories from a live webcam feed, providing visual overlays, alerts, and automatic logging.

This project was developed as part of a **Master’s-level Data Science / Deep Learning project**, with a strong focus on **class imbalance handling** and **real-time deployment**.

---

## 🚀 Features

- Real-time 3-class face mask detection
- YOLOv11-based lightweight model
- Minority-class optimized training
- Webcam-based live inference
- Visual bounding boxes with labels
- CSV logging of mask violations
- Simple GUI using Tkinter

---

## 🗂️ Project Structure

```
.
├── maskguard_gui.py              # Main application (GUI + inference)
├── best3.pt                     # Trained YOLOv11 model weights
├── A00051441_DeepLearning.ipynb  # Training & experimentation notebook
├── A00051441_DeepLearning.pdf   # Final project report
├── maskguard_violations_log.csv # Generated at runtime
└── README.md
```

---

## 🧩 Class Labels

| Class ID | Class Name |
|--------|------------|
| 0 | Mask |
| 1 | No Mask |
| 2 | Incorrect Mask |

---

## ⚙️ Requirements

- Python 3.9 – 3.11 (recommended: 3.10)
- Webcam
- Windows/Linux/macOS

### Python dependencies
```
pip install ultralytics opencv-python pillow numpy
```

---

## 📥 Installation

1. Clone the repository
```
git clone https://github.com/your-username/maskguard.git
cd maskguard
```

2. Place the trained model file `best3.pt` in the project directory

3. Install dependencies
```
pip install ultralytics opencv-python pillow
```

---

## ▶️ Running the Application

Run the GUI-based real-time detection:

```
python maskguard_gui.py
```

### At runtime:
- Webcam opens automatically
- Faces are detected and classified
- Bounding boxes are color-coded
- Mask violations are logged to CSV

---

## 📊 Output Logging

The system automatically creates:
```
maskguard_violations_log.csv
```

Logged fields include timestamp, number of faces, violations, and compliance percentage.

---

## 🧪 Model Training

- Training details are available in:
  - `A00051441_DeepLearning.ipynb`
  - `A00051441_DeepLearning.pdf`
- Model trained using:
  - YOLOv11n
  - Extended training (150 epochs)
  - Conservative class weighting
  - Heavy data augmentation

---

## 🔐 Ethics & Privacy

This system does **not perform identity recognition**.  
It is intended only for **mask compliance monitoring**.  
Users must comply with local privacy and data protection regulations.

---

Gradio Link : https://huggingface.co/spaces/Karthikeya1610/FaceMaskDetectionYolo11

## 📄 License

This project is intended for **academic and research use only**.
