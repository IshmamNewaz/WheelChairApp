
import sys
import time
import threading
import logging
import os
from dataclasses import dataclass
from collections import Counter

import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame, QMainWindow, QTabWidget, QStackedWidget, QTextEdit
)

try:
    from rplidar import RPLidar
    ON_LIDAR = True
except ImportError:
    ON_LIDAR = False

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except (ImportError, RuntimeError):
    ON_PI = False
    class _DummyGPIO:
        BCM = OUT = HIGH = LOW = None
        def setwarnings(self, *_): pass
        def setmode(self, *_): pass
        def setup(self, *_ , **__): pass
        def output(self, *_): pass
        def cleanup(self): pass
    GPIO = _DummyGPIO()


TARGET_SIZE = 640
FPS_LIMIT = 30

PORT = '/dev/ttyUSB0'
STOP_MIN = 300
STOP_MAX_INDOOR = 600
CLOSE_MIN_INDOOR = 600
CLOSE_MAX_INDOOR = 1000
MESSAGE_INTERVAL = 1

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
PIN_R, PIN_Y, PIN_G = 16, 21, 12
GPIO.setup(PIN_R, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PIN_Y, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PIN_G, GPIO.OUT, initial=GPIO.LOW)

def set_led_state(state_char: str):
    if ON_PI:
        GPIO.output(PIN_R, GPIO.HIGH if state_char == "R" else GPIO.LOW)
        GPIO.output(PIN_Y, GPIO.HIGH if state_char == "Y" else GPIO.LOW)
        GPIO.output(PIN_G, GPIO.HIGH if state_char == "G" else GPIO.LOW)

def get_direction(angle_deg: float) -> str:
    angle_deg %= 360
    if angle_deg <= 45 or angle_deg > 315: return "front"
    elif 45 < angle_deg <= 135: return "right"
    elif 135 < angle_deg <= 225: return "back"
    else: return "left"

def center_crop_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0+side, x0:x0+side]

@dataclass
class Thresholds:
    stop_min: int
    stop_max: int
    close_min: int
    close_max: int

@dataclass
class LidarUpdate:
    min_dist: float | None = None
    min_angle: float | None = None
    max_dist: float | None = None
    max_angle: float | None = None
    zone: str = "clear"
    zone_angle: float | None = None
    direction: str | None = None
    thresholds: Thresholds | None = None
    status: str = "OK"

class LidarWorker(QObject):
    update_signal = Signal(LidarUpdate)

    def __init__(self, thresholds_ref: dict):
        super().__init__()
        self.thresholds_ref = thresholds_ref
        self.running = False
        self.last_msg_time = 0
        self.last_zone = None
        self.last_led_state = None

    def run(self):
        self.running = True
        self._set_led("G")

        if not ON_LIDAR:
            while self.running:
                self.update_signal.emit(LidarUpdate(status="Error: rplidar library not found."))
                time.sleep(2)
            return

        while self.running:
            lidar = None
            try:
                lidar = RPLidar(PORT)
                for scan in lidar.iter_scans():
                    if not self.running: break
                    
                    valid_pairs = [(dist, ang) for _, ang, dist in scan if dist > 0]
                    upd = LidarUpdate(thresholds=self.thresholds_ref["t"])

                    if valid_pairs:
                        min_dist, min_angle = min(valid_pairs, key=lambda x: x[0])
                        max_dist, max_angle = max(valid_pairs, key=lambda x: x[0])
                        upd.min_dist, upd.min_angle = float(min_dist), float(min_angle)
                        upd.max_dist, upd.max_angle = float(max_dist), float(max_angle)

                    zone, closest_angle_for_zone = "clear", None
                    t = self.thresholds_ref["t"]
                    if valid_pairs:
                        stop_cand = [p for p in valid_pairs if t.stop_min <= p[0] < t.stop_max]
                        close_cand = [p for p in valid_pairs if t.close_min <= p[0] < t.close_max]
                        if stop_cand:
                            _, closest_angle_for_zone = min(stop_cand, key=lambda x: x[0])
                            zone = "stop"
                        elif close_cand:
                            _, closest_angle_for_zone = min(close_cand, key=lambda x: x[0])
                            zone = "close"
                    
                    upd.zone = zone
                    upd.zone_angle = closest_angle_for_zone
                    if closest_angle_for_zone is not None:
                        upd.direction = get_direction(closest_angle_for_zone)

                    desired_led = "G"
                    if zone == "stop": desired_led = "R"
                    elif zone == "close": desired_led = "Y"
                    self._set_led(desired_led)
                    
                    self.update_signal.emit(upd)
            except Exception as e:
                self.update_signal.emit(LidarUpdate(status=f"Error: {e} (restarting)"))
                time.sleep(1)
            finally:
                if lidar:
                    try: lidar.stop()
                    except: pass
                    try: lidar.disconnect()
                    except: pass
        self._set_led("G")

    def stop(self):
        self.running = False

    def _set_led(self, state_char: str):
        if state_char != self.last_led_state:
            set_led_state(state_char)
            self.last_led_state = state_char

class LidarTab(QWidget):
    def __init__(self):
        super().__init__()
        self.thresholds_ref = {"t": Thresholds(STOP_MIN, STOP_MAX_INDOOR, CLOSE_MIN_INDOOR, CLOSE_MAX_INDOOR)}
        
        self.worker = LidarWorker(self.thresholds_ref)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.update_signal.connect(self.update_ui)
        
        self.stack = QStackedWidget()
        self.mode_picker = self._create_mode_picker()
        self.dashboard = self._create_dashboard()
        self.stack.addWidget(self.mode_picker)
        self.stack.addWidget(self.dashboard)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.stack)
        self.setLayout(layout)

    def _create_mode_picker(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addStretch(1)
        layout.addWidget(QLabel("Choose Mode"))
        layout.addWidget(QLabel("Select environment before starting the LiDAR."))
        
        indoor_btn = QPushButton("Indoor")
        indoor_btn.clicked.connect(lambda: self.start_dashboard("Indoor"))
        outdoor_btn = QPushButton("Outdoor")
        outdoor_btn.clicked.connect(lambda: self.start_dashboard("Outdoor"))
        
        layout.addWidget(indoor_btn)
        layout.addWidget(outdoor_btn)
        layout.addStretch(1)
        return widget

    def _create_dashboard(self):
        widget = QWidget()
        
        self.status = QLabel("Starting...")
        self.min_line = QLabel("—")
        self.max_line = QLabel("—")
        self.zone_line = QLabel("CLEAR")
        self.thresh_line = QLabel(self._format_thresholds())
        
        self.btn_indoor = QPushButton("Indoor")
        self.btn_indoor.clicked.connect(self.set_indoor)
        self.btn_outdoor = QPushButton("Outdoor")
        self.btn_outdoor.clicked.connect(self.set_outdoor)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Min (nearest)"))
        left_layout.addWidget(self.min_line)
        left_layout.addWidget(QLabel("Max (farthest)"))
        left_layout.addWidget(self.max_line)
        left_layout.addWidget(QLabel("Zone"))
        left_layout.addWidget(self.zone_line)
        left_layout.addWidget(QLabel("Thresholds"))
        left_layout.addWidget(self.thresh_line)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Mode"))
        right_layout.addWidget(self.btn_indoor)
        right_layout.addWidget(self.btn_outdoor)
        right_layout.addStretch(1)

        main_layout = QHBoxLayout(widget)
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)
        
        return widget
    
    def start_dashboard(self, mode):
        if mode == "Outdoor": self.set_outdoor()
        else: self.set_indoor()
        self.thread.start()
        self.stack.setCurrentWidget(self.dashboard)

    def _format_thresholds(self):
        t = self.thresholds_ref["t"]
        return f"STOP: {t.stop_min}-{t.stop_max} mm | CLOSE: {t.close_min}-{t.close_max} mm"

    def set_indoor(self):
        self.thresholds_ref["t"] = Thresholds(STOP_MIN, STOP_MAX_INDOOR, CLOSE_MIN_INDOOR, CLOSE_MAX_INDOOR)
        self.thresh_line.setText(self._format_thresholds())
        self.btn_indoor.setStyleSheet("font-weight: bold;")
        self.btn_outdoor.setStyleSheet("")

    def set_outdoor(self):
        self.thresholds_ref["t"] = Thresholds(STOP_MIN, STOP_MAX_INDOOR * 2, CLOSE_MIN_INDOOR * 2, CLOSE_MAX_INDOOR * 2)
        self.thresh_line.setText(self._format_thresholds())
        self.btn_outdoor.setStyleSheet("font-weight: bold;")
        self.btn_indoor.setStyleSheet("")

    def update_ui(self, upd: LidarUpdate):
        self.status.setText("OK" if upd.status == "OK" else upd.status)
        if upd.min_dist is not None: self.min_line.setText(f"{upd.min_dist:.0f} mm @ {upd.min_angle:.1f}°")
        else: self.min_line.setText("—")
        if upd.max_dist is not None: self.max_line.setText(f"{upd.max_dist:.0f} mm @ {upd.max_angle:.1f}°")
        else: self.max_line.setText("—")
        
        d = upd.direction or "unknown"
        if upd.zone == "stop": self.zone_line.setText(f"STOP ({d})")
        elif upd.zone == "close": self.zone_line.setText(f"CLOSE ({d})")
        else: self.zone_line.setText("CLEAR")
        
        self.thresh_line.setText(self._format_thresholds())
    
    def stop_worker(self):
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.cap, self.streaming, self.last_frame_time = None, False, 0.0

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

        card = QFrame(); card.setObjectName("Card")
        card_layout = QVBoxLayout(card); card_layout.setContentsMargins(18,18,18,18); card_layout.setSpacing(12)
        card_layout.addWidget(self.title); card_layout.addWidget(self.video, alignment=Qt.AlignCenter)
        bottom = QHBoxLayout(); bottom.addWidget(self.btn); bottom.addStretch(1); bottom.addWidget(self.status)
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(22,22,22,22); main_layout.setSpacing(14)
        main_layout.addWidget(card); main_layout.addLayout(bottom)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet("""
            #Card { background: #0F1620; border: 1px solid rgba(231,238,247,0.08); border-radius: 16px; }
            #Title { font-size: 20px; font-weight: 700; }
            #Video { background: #070A0E; border: 1px solid rgba(231,238,247,0.08); border-radius: 14px; }
            #PrimaryButton { background: #2B74FF; border: none; padding: 10px 14px; border-radius: 12px; font-weight: 700; min-width: 120px; }
            #PrimaryButton:hover { background: #3A82FF; } #PrimaryButton:pressed { background: #1E5FE0; }
            #Status { color: rgba(231,238,247,0.75); }
        """)

    def toggle_stream(self):
        if not self.streaming: self.start_stream()
        else: self.stop_stream()

    def start_stream(self):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_AVFOUNDATION if sys.platform=="darwin" else 0
        self.cap = None
        for idx in range(11):
            cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)
            if cap.isOpened():
                self.cap, working_index = cap, idx
                print(f"Using camera index {idx}")
                break
            cap.release()
        
        if self.cap is None:
            self.status.setText("Camera not found"); self.video.setText("Could not open webcam (0-10)")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        
        self.streaming, self.last_frame_time = True, 0.0
        self.btn.setText("Stop"); self.status.setText(f"Streaming (Cam {working_index})"); self.video.setText("")
        self.timer.start(5)

    def stop_stream(self):
        self.timer.stop(); self.streaming = False
        self.btn.setText("Start"); self.status.setText("Idle"); self.video.setText("Press Start to stream")
        self.video.setPixmap(QPixmap())
        if self.cap: self.cap.release(); self.cap = None

    def update_frame(self):
        if not self.streaming or not self.cap: return
        now = time.time()
        if self.last_frame_time and (now-self.last_frame_time) < (1.0/FPS_LIMIT): return
        self.last_frame_time = now

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status.setText("Frame read failed"); return
        
        frame = center_crop_square(frame)
        frame = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()

def stopDefaultLogging():
    for name in logging.root.manager.loggerDict:
        t_logger = logging.getLogger(name)
        t_logger.setLevel(logging.CRITICAL)

def talk(text):
    speak = pyttsx3.init()
    voices = speak.getProperty('voices')
    try:
     speak.setProperty('voice', voices[3].id)
    except IndexError:
     pass
    speak.setProperty('rate', 150)
    speak.say(text)
    speak.runAndWait()

def imgCapture(numberOfImg, captureSource):
    imgSrc = []
    folder_path = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i in range(numberOfImg):
        ret, frame = captureSource.read()
        if not ret: continue
        image_path = os.path.join(folder_path, f"image_{i}.jpg")
        cv2.imwrite(image_path, frame)
        imgSrc.append(image_path)
    return imgSrc

def blackScreenDetection(frame):
    if frame is None: return 0
    imgHeight = frame.shape[0]
    top_half = frame[0:imgHeight//2, :]
    return 1 if cv2.mean(top_half)[0] < 1 else 0

def decisionBatch(imagePaths, imageSize, confThres, model):
    batch_results = []
    for path in imagePaths:
        inputSource = cv2.imread(path)
        if blackScreenDetection(inputSource) == 0 and inputSource is not None:
            results = model(source=inputSource, imgsz=imageSize, conf=confThres, stream=False, show=False)
            for r in results:
                if r.boxes.cls.numel() == 0:
                    batch_results.append("unknown")
                else:
                    for i, cls in enumerate(r.boxes.cls):
                        batch_results.append(r.names[int(cls)])
        else:
            batch_results.append("Black Screen!")
    return batch_results

def most_common_result(batch_results):
    if not batch_results: return "unknown"
    return Counter(batch_results).most_common(1)[0][0]

class BatchDetectWorker(QObject):
    result_signal = Signal(list, str)

    def __init__(self, camera_cap):
        super().__init__()
        self.running = False
        self.camera_cap = camera_cap
        self.model = YOLO("last.pt")
        self.last_common_result = None
        self.last_confirm_time = 0.0

    def run(self):
        self.running = True
        stopDefaultLogging()
        while self.running:
            if not self.camera_cap or not self.camera_cap.isOpened():
                time.sleep(1)
                continue

            images = imgCapture(5, self.camera_cap)
            if not images:
                time.sleep(0.5)
                continue
                
            batch_results = decisionBatch(images, 640, 0.4, self.model)
            common_result = most_common_result(batch_results)
            self.result_signal.emit(batch_results, common_result)
            
            current_time = time.time()
            if common_result != self.last_common_result:
                self.last_common_result = common_result
                self.last_confirm_time = 0.0

            if self.last_common_result not in [None, 'unknown'] and current_time - self.last_confirm_time >= 10:
                talk(f"{self.last_common_result} Ahead!")
                self.last_confirm_time = current_time
            time.sleep(0.1)

    def stop(self):
        self.running = False

class BatchDetectTab(QWidget):
    def __init__(self, camera_app):
        super().__init__()
        self.camera_app = camera_app
        self.worker = None
        self.thread = None

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.common_result_label = QLabel("Most Common: N/A")
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self.log_view)
        layout.addWidget(self.common_result_label)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

    def start_detection(self):
        if not self.camera_app.cap:
            self.log_view.append("Camera not started. Please start the webcam first.")
            return

        self.worker = BatchDetectWorker(self.camera_app.cap)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result_signal.connect(self.update_results)
        self.thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_view.clear()
        self.log_view.append("Detection started...")

    def stop_detection(self):
        if self.worker: self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_view.append("Detection stopped.")

    def update_results(self, batch_results, common_result):
        self.log_view.append(f"Batch: {batch_results}")
        self.common_result_label.setText(f"Most Common: {common_result}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combined GUI")
        self.setGeometry(100, 100, 800, 600)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.lidar_tab = LidarTab()
        self.webcam_tab = CameraApp()
        self.batch_detect_tab = BatchDetectTab(self.webcam_tab)

        self.tabs.addTab(self.lidar_tab, "LIDAR")
        self.tabs.addTab(self.webcam_tab, "Webcam")
        self.tabs.addTab(self.batch_detect_tab, "Batch Detect")
        
        self.setStyleSheet("""
            QWidget { background: #0B0F14; color: #E7EEF7; font-family: -apple-system, Segoe UI, Roboto; font-size: 14px; }
        """)

    def closeEvent(self, event):
        self.lidar_tab.stop_worker()
        self.webcam_tab.stop_stream()
        if self.batch_detect_tab.worker:
            self.batch_detect_tab.stop_detection()
        event.accept()
        
if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    finally:
        if ON_PI: GPIO.cleanup()
