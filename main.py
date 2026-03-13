"""
Smart Gesture — hand gesture mouse controller.

Gestures:
  Move cursor     : index fingertip position
  Left click      : thumb + index pinch  (quick double pinch = double click)
  Right click     : thumb + middle finger pinch
  Scroll          : all 4 fingers up, move hand up/down
  Screenshot      : closed fist (all fingers down)

Press 'q' to quit.
"""

import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # remove default 0.1s delay for responsiveness

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Hand skeleton connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # pinky
    (0,17),(5,9),(9,13),(13,17),      # palm
]

# ── Tunable constants ────────────────────────────────────────────────────────
SMOOTH_FACTOR       = 0.35   # cursor EMA smoothing  (0 = no move, 1 = raw)
PINCH_THRESH        = 0.05   # normalised distance to register a pinch
CLICK_COOLDOWN      = 0.35   # seconds between consecutive clicks
DOUBLE_CLICK_WINDOW = 0.45   # two pinches within this window → double click
SCROLL_DEADZONE     = 0.025  # min y-delta before scrolling kicks in
SCROLL_SPEED        = 80     # pyautogui scroll units per tick
SCREENSHOT_COOLDOWN = 2.0    # seconds between screenshots


def ensure_model():
    """Download the hand landmarker model if not already present."""
    if not MODEL_PATH.exists():
        print(f"Downloading hand landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


class GestureController:
    """Maps MediaPipe hand landmarks to mouse / keyboard actions."""

    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()

        self.cap = cv2.VideoCapture(0)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        # Cursor smoothing state
        self.cursor_x: float = 0.0
        self.cursor_y: float = 0.0

        # Click state
        self.left_pinch_active  = False
        self.right_pinch_active = False
        self.last_left_click    = 0.0
        self.last_right_click   = 0.0
        self.click_times: list  = []

        # Scroll state
        self.scroll_ref_y: float | None = None

        # Screenshot state
        self.last_screenshot = 0.0

    # ── Drawing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_landmarks(frame, landmarks):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 5, (0, 150, 255), 2)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pinch(landmarks, a: int, b: int) -> float:
        dx = landmarks[a].x - landmarks[b].x
        dy = landmarks[a].y - landmarks[b].y
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def _fingers_up(landmarks) -> list[int]:
        """Returns [index, middle, ring, pinky] extended flags (1 = up)."""
        return [
            1 if landmarks[tip].y < landmarks[tip - 2].y else 0
            for tip in [8, 12, 16, 20]
        ]

    @staticmethod
    def _label(frame, text: str, y: int, color=(0, 255, 0)):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # ── Gesture handlers ─────────────────────────────────────────────────────

    def _move_cursor(self, landmarks):
        tip = landmarks[8]
        tx  = tip.x * self.screen_w
        ty  = tip.y * self.screen_h
        self.cursor_x += (tx - self.cursor_x) * SMOOTH_FACTOR
        self.cursor_y += (ty - self.cursor_y) * SMOOTH_FACTOR
        pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y))

    def _handle_left_click(self, landmarks, frame) -> bool:
        pinching = self._pinch(landmarks, 4, 8) < PINCH_THRESH
        now = time.time()

        if pinching and not self.left_pinch_active:
            self.left_pinch_active = True
            if now - self.last_left_click < CLICK_COOLDOWN:
                return True
            self.last_left_click = now
            self.click_times = [t for t in self.click_times if now - t < DOUBLE_CLICK_WINDOW]
            self.click_times.append(now)

            if len(self.click_times) >= 2:
                pyautogui.doubleClick()
                self.click_times = []
                self._label(frame, "Double Click", 50)
            else:
                pyautogui.click()
                self._label(frame, "Click", 50)

        elif not pinching:
            self.left_pinch_active = False

        return pinching

    def _handle_right_click(self, landmarks, frame):
        pinching = self._pinch(landmarks, 4, 12) < PINCH_THRESH
        now = time.time()

        if pinching and not self.right_pinch_active:
            self.right_pinch_active = True
            if now - self.last_right_click > CLICK_COOLDOWN:
                self.last_right_click = now
                pyautogui.rightClick()
                self._label(frame, "Right Click", 50, (255, 100, 0))
        elif not pinching:
            self.right_pinch_active = False

    def _handle_scroll(self, landmarks, fingers: list[int], frame):
        if sum(fingers) == 4:
            tip_y = landmarks[8].y
            if self.scroll_ref_y is None:
                self.scroll_ref_y = tip_y
            dy = tip_y - self.scroll_ref_y
            if abs(dy) > SCROLL_DEADZONE:
                direction = -1 if dy > 0 else 1
                pyautogui.scroll(direction * SCROLL_SPEED)
                text, color = ("Scroll Down", (0, 0, 255)) if dy > 0 else ("Scroll Up", (0, 255, 0))
                self._label(frame, text, 90, color)
                self.scroll_ref_y = tip_y
        else:
            self.scroll_ref_y = None

    def _handle_screenshot(self, fingers: list[int], frame):
        if sum(fingers) == 0:
            now = time.time()
            if now - self.last_screenshot > SCREENSHOT_COOLDOWN:
                self.last_screenshot = now
                pyautogui.screenshot(f"screenshot_{int(now)}.png")
                self._label(frame, "Screenshot!", 130, (255, 255, 0))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        if not self.cap.isOpened():
            print("Error: cannot open camera.")
            return

        print("Smart Gesture running — press 'q' to quit.")
        start_ms = int(time.time() * 1000)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera stream ended.")
                break

            frame      = cv2.flip(frame, 1)
            timestamp  = int(time.time() * 1000) - start_ms
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result     = self.detector.detect_for_video(mp_image, timestamp)

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                self._draw_landmarks(frame, landmarks)

                fingers       = self._fingers_up(landmarks)
                left_pinching = self._handle_left_click(landmarks, frame)

                if not left_pinching:
                    self._handle_right_click(landmarks, frame)

                if not left_pinching:
                    self._move_cursor(landmarks)

                self._handle_scroll(landmarks, fingers, frame)
                self._handle_screenshot(fingers, frame)

            cv2.imshow("Smart Gesture", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ensure_model()
    GestureController().run()
