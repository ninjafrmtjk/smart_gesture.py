# smart_gesture

# 🖐️ Hand Mouse Control

> Control your computer with hand gestures — no mouse needed.

Built with **MediaPipe** + **OpenCV**, this project tracks your hand through a webcam in real time and translates gestures into mouse actions.

---

## ✨ Features

- 🖱️ **Move cursor** — point with your index finger
- 👆 **Left click** — pinch thumb and index finger
- ✌️ **Double click** — pinch twice quickly
- 🤞 **Right click** — pinch thumb and middle finger
- 🖐️ **Scroll** — raise all four fingers, move hand up/down
- 📸 **Screenshot** — make a fist

---

## 🚀 Quick Start

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install opencv-python mediapipe pyautogui numpy

# Navigate to project folder
cd C:\...(project location)

# Run
python main.py
```

> On first run, the hand landmark model (~11 MB) downloads automatically.
> Press **Q** to quit.

---

## 📦 Requirements

- Python 3.11
- Webcam

---

## 📁 Files

```
main.py       — gesture controller
util.py       — geometry helpers (angle, distance)
```

---

## 📄 License

MIT
