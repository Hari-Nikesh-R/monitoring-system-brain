# 🐄 Cattle Health Monitoring System

## 📌 Project Overview

The Cattle Health Monitoring System is an AI-based solution designed to monitor cattle behavior using computer vision. The system uses a fixed camera setup to continuously observe cows and analyze their movement patterns to detect potential health issues.

---

## 🎯 Objectives

- Automate cattle monitoring using camera systems
- Reduce manual effort for farmers
- Detect abnormal behavior early
- Improve livestock health management

---

## 🏗️ System Architecture

```
Camera
   ↓
Video Capture (OpenCV)
   ↓
Cow Detection (YOLO)
   ↓
Cow Tracking (DeepSort/ByteTrack)
   ↓
Movement Analysis
   ↓
Health Monitoring
   ↓
Database Storage
   ↓
Dashboard / Alerts
```

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- OpenCV (video processing)
- Ultralytics YOLO (object detection)
- NumPy (data handling)
- DeepSort / ByteTrack (tracking)
- Streamlit (dashboard)

### Hardware
- Camera (CCTV / Webcam)
- Computer (Laptop/Desktop)

---

## ⚙️ Step-by-Step Implementation

### Step 1: Setup Environment

Install required libraries:

```bash
pip install opencv-python
pip install ultralytics
pip install numpy
pip install streamlit
```

---

### Step 2: Capture Video

```python
import cv2

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    cv2.imshow("Cattle Monitoring", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
```

---

### Step 3: Cow Detection using YOLO

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Cow Detection", annotated)

    if cv2.waitKey(1) == 27:
        break
```

---

### Step 4: Cow Tracking

- Assign unique IDs to cows
- Track across frames

Example Output:
```
Cow ID 1
Cow ID 2
```

---

### Step 5: Movement Tracking

Track position changes:

```python
if distance < threshold:
    cow_is_idle = True
```

---

### Step 6: Health Monitoring Logic

#### Rules:

- If cow not moving for long → Possible illness
- If abnormal movement → Possible injury
- If isolated → Possible issue

---

### Step 7: Data Storage

Store data in:

- SQLite
- CSV file

Example:

| Cow ID | Movement | Status |
|--------|----------|--------|
| 1      | Active   | Healthy|
| 2      | Idle     | Alert  |

---

### Step 8: Alert System

Example:

```
⚠ Cow 2 inactive for long time
```

---

### Step 9: Dashboard (Optional)

Use Streamlit to display:

- Live feed
- Cow status
- Alerts

Run:

```bash
streamlit run app.py
```

---

## 📊 Expected Output

- Real-time cow detection
- Unique cow tracking
- Movement analysis
- Health alerts

---

## 🚀 Future Enhancements

- Face recognition for cows
- Mobile app notifications
- Temperature sensors
- Disease prediction using ML

---

## ✅ Conclusion

This project provides an automated solution for monitoring cattle using AI and computer vision. It helps in early detection of health issues and reduces manual effort, making livestock management more efficient.

