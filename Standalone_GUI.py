import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame
)
INPUT_INDEX = 3
TARGET_SIZE = 640  # 640x640 display
FPS_LIMIT = 30     # keep resource usage low (increase if you want smoother)

def center_crop_square(frame: np.ndarray) -> np.ndarray:
    """Crop the largest possible square from the center of the frame."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Stream")
        self.setMinimumSize(760, 820)

        self.cap = None
        self.streaming = False
        self.last_frame_time = 0.0

        # --- UI ---
        self.title = QLabel("Webcam")
        self.title.setObjectName("Title")

        self.video = QLabel()
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setObjectName("Video")
        self.video.setText("Press Start to stream")
        self.video.setMinimumSize(TARGET_SIZE, TARGET_SIZE)
        self.video.setMaximumSize(TARGET_SIZE, TARGET_SIZE)

        self.btn = QPushButton("Start")
        self.btn.setObjectName("PrimaryButton")
        self.btn.clicked.connect(self.toggle_stream)

        self.status = QLabel("Idle")
        self.status.setObjectName("Status")

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(12)
        card_layout.addWidget(self.title)
        card_layout.addWidget(self.video, alignment=Qt.AlignCenter)

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn)
        bottom.addStretch(1)
        bottom.addWidget(self.status)

        main = QVBoxLayout(self)
        main.setContentsMargins(22, 22, 22, 22)
        main.setSpacing(14)
        main.addWidget(card)
        main.addLayout(bottom)

        # Timer for grabbing frames (Qt-friendly, low overhead)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background: #0B0F14;
                color: #E7EEF7;
                font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial;
                font-size: 14px;
            }
            #Card {
                background: #0F1620;
                border: 1px solid rgba(231, 238, 247, 0.08);
                border-radius: 16px;
            }
            #Title {
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            #Video {
                background: #070A0E;
                border: 1px solid rgba(231, 238, 247, 0.08);
                border-radius: 14px;
            }
            #PrimaryButton {
                background: #2B74FF;
                border: none;
                padding: 10px 14px;
                border-radius: 12px;
                font-weight: 700;
                min-width: 120px;
            }
            #PrimaryButton:hover { background: #3A82FF; }
            #PrimaryButton:pressed { background: #1E5FE0; }
            #Status {
                color: rgba(231, 238, 247, 0.75);
            }
        """)

    def toggle_stream(self):
        if not self.streaming:
            self.start_stream()
        else:
            self.stop_stream()

    def start_stream(self):
        backend = 0
        if sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION

        self.cap = None
        working_index = None

        # Try camera indices 0..10
        for idx in range(0, 11):
            cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
            if cap.isOpened():
                self.cap = cap
                working_index = idx
                print(f"Using camera index {idx}")
                break
            cap.release()

        if self.cap is None:
            self.status.setText("Camera not found")
            self.video.setText("Could not open webcam (0–10)")
            return

        # Ask for a small-ish capture size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.streaming = True
        self.btn.setText("Stop")
        self.status.setText(f"Streaming (Camera {working_index})")
        self.video.setText("")
        self.last_frame_time = 0.0

        self.timer.start(5)

    def stop_stream(self):
        self.timer.stop()
        self.streaming = False
        self.btn.setText("Start")
        self.status.setText("Idle")
        self.video.setText("Press Start to stream")
        self.video.setPixmap(QPixmap())  # clear image

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_frame(self):
        if not self.streaming or self.cap is None:
            return

        # FPS limiting (keeps CPU low)
        now = time.time()
        if self.last_frame_time and (now - self.last_frame_time) < (1.0 / FPS_LIMIT):
            return
        self.last_frame_time = now

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status.setText("Frame read failed")
            return

        # Convert to 640x640: center crop square, then resize
        frame = center_crop_square(frame)
        frame = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Display
        self.video.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        # Ensure camera is released cleanly
        self.stop_stream()
        event.accept()

def main():
    app = QApplication(sys.argv)
    w = CameraApp()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
