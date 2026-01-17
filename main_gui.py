import sys

import os

import subprocess

import time

import threading

import queue
from collections import deque, Counter

from dataclasses import dataclass
from typing import Callable



import cv2

import numpy as np



from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject

from PySide6.QtGui import QImage, QPixmap

from PySide6.QtWidgets import (

    QApplication, QWidget, QLabel, QPushButton,

    QVBoxLayout, QHBoxLayout, QFrame, QMainWindow, QCheckBox

)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False



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



try:

    import pyttsx3

    TTS_AVAILABLE = True

except Exception:

    pyttsx3 = None

    TTS_AVAILABLE = False


def find_camera_index(start=0, end=10):
    for i in range(start, end + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None
    
TARGET_SIZE = 640

FPS_LIMIT = 30
STANDALONE_INPUT_INDEX = find_camera_index(0, 10)
CAMERA_INDEX_RANGE = range(0, 11)



PORT = '/dev/ttyUSB0'

STOP_MIN = 300

STOP_MAX_INDOOR = 600

CLOSE_MIN_INDOOR = 600

CLOSE_MAX_INDOOR = 1000

MESSAGE_INTERVAL = 1

SPEAK_REPEAT_SECONDS = 60

SPEAK_RATE = 150



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

    if angle_deg <= 45 or angle_deg > 315:

        return "front"

    elif 45 < angle_deg <= 135:

        return "right"

    elif 135 < angle_deg <= 225:

        return "back"

    else:

        return "left"



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

                    if not self.running:

                        break

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

                    if zone == "stop":

                        desired_led = "R"

                    elif zone == "close":

                        desired_led = "Y"

                    self._set_led(desired_led)



                    self.update_signal.emit(upd)

            except Exception as e:

                self.update_signal.emit(LidarUpdate(status=f"Error: {e} (restarting)"))

                time.sleep(1)

            finally:

                if lidar:

                    try:

                        lidar.stop()

                    except Exception:

                        pass

                    try:

                        lidar.disconnect()

                    except Exception:

                        pass

        self._set_led("G")



    def stop(self):

        self.running = False



    def _set_led(self, state_char: str):

        if state_char != self.last_led_state:

            set_led_state(state_char)

            self.last_led_state = state_char



class TTSWorker(threading.Thread):

    def __init__(self, tts_q: queue.Queue, stop_event: threading.Event):

        super().__init__(daemon=True)

        self.tts_q = tts_q

        self.stop_event = stop_event

        self.engine = None



    def run(self):

        if not TTS_AVAILABLE:

            return



        try:

            self.engine = pyttsx3.init()

            self.engine.setProperty('rate', SPEAK_RATE)

        except Exception:

            self.engine = None

            return



        while not self.stop_event.is_set():

            try:

                msg = self.tts_q.get(timeout=0.2)

            except queue.Empty:

                continue



            if msg is None:

                continue



            try:

                self.engine.say(msg)

                self.engine.runAndWait()

            except Exception:

                pass



        try:

            if self.engine is not None:

                self.engine.stop()

        except Exception:

            pass



class SpeechGate:

    def __init__(self, tts_q: queue.Queue):

        self.tts_q = tts_q

        self.last_key: tuple[str, str] | None = None

        self.last_spoken_time: float = 0.0



    def _phrase(self, zone: str, direction: str) -> str | None:

        if zone == "close":

            return f"Obstacle close at {direction}, careful!"

        if zone == "stop":

            return f"Obstacle in proximity at {direction}, stop!"

        return None



    def consider(self, zone: str, direction: str | None):

        if direction is None:

            return



        if zone not in ("close", "stop"):

            self.last_key = None

            self.last_spoken_time = 0.0

            return



        now = time.time()

        key = (zone, direction)



        if self.last_key != key:

            phrase = self._phrase(zone, direction)

            if phrase:

                self.tts_q.put(phrase)

                self.last_key = key

                self.last_spoken_time = now

            return



        if now - self.last_spoken_time >= SPEAK_REPEAT_SECONDS:

            phrase = self._phrase(zone, direction)

            if phrase:

                self.tts_q.put(phrase)

                self.last_spoken_time = now



class CameraApp(QWidget):

    def __init__(
        self,
        title: str = "Webcam",
        camera_indices: list[int] | None = None,
        fixed_index: int | None = None,
        display_size: tuple[int, int] = (TARGET_SIZE, 300),
        auto_start: bool = False,
        show_controls: bool = True,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ):

        super().__init__()

        self.cap, self.streaming, self.last_frame_time = None, False, 0.0
        self._camera_indices = camera_indices
        self._fixed_index = fixed_index
        self._display_width, self._display_height = display_size
        self._on_start = on_start
        self._on_stop = on_stop
        self._frame_lock = threading.Lock()
        self._frame_queue = deque(maxlen=20)
        self._latest_frame = None



        self.title = QLabel(title)

        self.title.setObjectName("Title")

        self.video = QLabel()

        self.video.setAlignment(Qt.AlignCenter)

        self.video.setObjectName("Video")

        self.video.setText("Press Start to stream")

        self.video.setMinimumSize(self._display_width, self._display_height)

        self.video.setMaximumSize(self._display_width, self._display_height)



        self.btn = QPushButton("Front Cam")

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

        if not show_controls:
            self.btn.setVisible(False)
            self.status.setVisible(False)

        if auto_start:
            QTimer.singleShot(0, self.start_stream)



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

        if not self.streaming:

            self.start_stream()

        else:

            self.stop_stream()



    def start_stream(self):

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_AVFOUNDATION if sys.platform=="darwin" else 0

        self.cap = None

        working_index = None
        if self._fixed_index is not None:
            indices = [self._fixed_index]
        else:
            indices = self._camera_indices if self._camera_indices is not None else list(CAMERA_INDEX_RANGE)

        for idx in indices:

            cap = cv2.VideoCapture(idx, backend) if backend != 0 else cv2.VideoCapture(idx)

            if cap.isOpened():

                self.cap, working_index = cap, idx

                print(f"Using camera index {idx}")

                break

            cap.release()

        

        if self.cap is None:

            self.status.setText("Camera not found")

            self.video.setText("Could not open webcam (0-10)")

            return



        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        except Exception:

            pass

        

        self.streaming, self.last_frame_time = True, 0.0

        self.btn.setText("Stop")

        if working_index is None:
            self.status.setText("Streaming")
        else:
            self.status.setText(f"Streaming (Cam {working_index})")

        self.video.setText("")

        self.timer.start(5)

        if self._on_start:
            self._on_start()



    def stop_stream(self):

        self.timer.stop()

        self.streaming = False

        self.btn.setText("Front Cam")

        self.status.setText("Idle")

        self.video.setText("Press Start to stream")

        self.video.setPixmap(QPixmap())

        if self.cap:

            self.cap.release()

            self.cap = None

        if self._on_stop:
            self._on_stop()



    def update_frame(self):

        if not self.streaming or not self.cap:

            return

        now = time.time()

        if self.last_frame_time and (now-self.last_frame_time) < (1.0/FPS_LIMIT):

            return

        self.last_frame_time = now



        ok, frame = self.cap.read()

        if not ok or frame is None:

            self.status.setText("Frame read failed")

            return

        

        frame = center_crop_square(frame)
        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._frame_queue.append(frame.copy())

        frame = cv2.resize(frame, (self._display_width, self._display_height), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape

        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.video.setPixmap(QPixmap.fromImage(qimg))

    def get_recent_frames(self, count: int) -> list[np.ndarray]:
        with self._frame_lock:
            if not self._frame_queue:
                return []
            return list(self._frame_queue)[-count:]



    def closeEvent(self, event):

        self.stop_stream()

        event.accept()


class BatchDetectWorker(threading.Thread):
    def __init__(self, camera_app: CameraApp, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.camera_app = camera_app
        self.stop_event = stop_event
        self.model = None
        self.img_size = 640
        self.confidence = 0.4
        self.batch_count = 5

    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("Ultralytics not available; batch detection disabled.")
            return False
        weights_path = os.path.join(os.path.dirname(__file__), "last.pt")
        if not os.path.exists(weights_path):
            print(f"Model weights not found at {weights_path}; batch detection disabled.")
            return False
        self.model = YOLO(weights_path)
        return True

    def _detect_frame(self, frame: np.ndarray) -> str:
        results = self.model(source=frame, imgsz=self.img_size, conf=self.confidence, stream=False, show=False)
        for r in results:
            if r.boxes.cls.numel() == 0:
                return "unknown"
            cls = int(r.boxes.cls[0])
            return r.names.get(cls, "unknown")
        return "unknown"

    def run(self):
        if not self._load_model():
            return

        last_common = None
        last_confirm_time = 0.0

        while not self.stop_event.is_set():
            frames = self.camera_app.get_recent_frames(self.batch_count)
            if len(frames) < self.batch_count:
                time.sleep(0.5)
                continue

            batch_results = [self._detect_frame(f) for f in frames]
            for i, result in enumerate(batch_results):
                print(f"Result of image {i+1}: {result}")

            common_result = Counter(batch_results).most_common(1)[0][0]
            print(f"Most common result: {common_result}\n")

            current_time = time.time()
            if common_result != last_common:
                last_common = common_result
                last_confirm_time = 0.0

            if last_common is not None and last_common != "unknown" and current_time - last_confirm_time >= 10:
                print(f"\n{last_common} is confirmed")
                last_confirm_time = current_time

            time.sleep(0.5)



class CombinedView(QWidget):

    def __init__(self):

        super().__init__()

        self.thresholds_ref = {"t": Thresholds(STOP_MIN, STOP_MAX_INDOOR, CLOSE_MIN_INDOOR, CLOSE_MAX_INDOOR)}

        self.lidar_worker = LidarWorker(self.thresholds_ref)

        self.lidar_thread = QThread()

        self.lidar_worker.moveToThread(self.lidar_thread)

        self.lidar_thread.started.connect(self.lidar_worker.run)

        self.lidar_worker.update_signal.connect(self.update_lidar_ui)

        self.tts_q = queue.Queue()

        self.tts_stop_event = threading.Event()

        self.tts_worker = TTSWorker(self.tts_q, self.tts_stop_event)

        self.tts_worker.start()

        self.speech_gate = SpeechGate(self.tts_q)


        self._build_ui()

        self.lidar_thread.start()

        self.detect_stop_event = threading.Event()
        self.detect_worker = BatchDetectWorker(self.camera_app, self.detect_stop_event)
        self.detect_worker.start()



    def _build_ui(self):

        layout = QVBoxLayout(self)

        header = QHBoxLayout()

        title = QLabel("LiDAR Monitor")

        title.setObjectName("Title")

        self.lidar_status = QLabel("Starting...")

        self.lidar_status.setObjectName("Status")

        header.addWidget(title)

        header.addStretch(1)

        header.addWidget(self.lidar_status)

        controls = QHBoxLayout()

        controls.addWidget(QLabel("Mode"))

        self.btn_indoor = QPushButton("Indoor")

        self.btn_indoor.clicked.connect(self.set_indoor)

        self.btn_outdoor = QPushButton("Outdoor")

        self.btn_outdoor.clicked.connect(self.set_outdoor)

        self.speech_toggle = QCheckBox("Speech")

        self.speech_toggle.setChecked(True)

        self.speech_toggle.setEnabled(TTS_AVAILABLE)

        self.secondary_toggle_btn = QPushButton("Rear Cam")

        self.secondary_toggle_btn.setObjectName("PrimaryButton")

        self.secondary_toggle_btn.clicked.connect(self.toggle_secondary_stream)

        self.rear_toggle_btn = QPushButton("Front Cam")

        self.rear_toggle_btn.setObjectName("PrimaryButton")

        self.rear_toggle_btn.clicked.connect(self.toggle_rear_stream)

        controls.addWidget(self.btn_indoor)

        controls.addWidget(self.btn_outdoor)

        controls.addStretch(1)

        controls.addWidget(self.speech_toggle)

        controls.addWidget(self.secondary_toggle_btn)

        controls.addWidget(self.rear_toggle_btn)

        self.standalone_camera_app = CameraApp(
            title="Rear Camera",
            fixed_index=STANDALONE_INPUT_INDEX,
            display_size=(TARGET_SIZE, 300),
            auto_start=False,
            show_controls=False,
        )

        secondary_indices = [i for i in CAMERA_INDEX_RANGE if i != STANDALONE_INPUT_INDEX]
        self.camera_app = CameraApp(
            title="Image Detection",
            camera_indices=secondary_indices,
            display_size=(TARGET_SIZE, 300),
            auto_start=True,
            show_controls=False,
        )

        self.camera_app.setVisible(False)
        self.standalone_camera_app.setVisible(False)

        lidar_dashboard = QFrame()

        lidar_layout = QVBoxLayout(lidar_dashboard)

        self.min_line = QLabel("—")

        self.max_line = QLabel("—")

        self.zone_line = QLabel("CLEAR")

        self.thresh_line = QLabel(self._format_thresholds())

        left_layout = QVBoxLayout()

        left_layout.addWidget(QLabel("Min (nearest)"))

        left_layout.addWidget(self.min_line)

        left_layout.addWidget(QLabel("Max (farthest)"))

        left_layout.addWidget(self.max_line)

        left_layout.addWidget(QLabel("Zone"))

        left_layout.addWidget(self.zone_line)

        left_layout.addWidget(QLabel("Thresholds"))

        left_layout.addWidget(self.thresh_line)

        main_lidar_layout = QHBoxLayout()

        main_lidar_layout.addLayout(left_layout, stretch=1)

        lidar_layout.addLayout(main_lidar_layout)

        layout.addLayout(header)

        layout.addLayout(controls)

        camera_row = QHBoxLayout()
        camera_row.addWidget(self.standalone_camera_app)
        camera_row.addWidget(self.camera_app)

        layout.addLayout(camera_row)

        layout.addWidget(lidar_dashboard)

        self.set_indoor()



    def toggle_secondary_stream(self):

        if not self.camera_app.streaming:
            self.camera_app.start_stream()

        if not self.camera_app.streaming:
            self.camera_app.setVisible(False)
            self.secondary_toggle_btn.setText("Rear Cam")
            return

        if self.camera_app.isVisible():
            self.camera_app.setVisible(False)
            self.secondary_toggle_btn.setText("Rear Cam")
        else:
            self.camera_app.setVisible(True)
            self.secondary_toggle_btn.setText("Hide")


    def toggle_rear_stream(self):

        if not self.standalone_camera_app.streaming:

            self.standalone_camera_app.setVisible(True)

            self.standalone_camera_app.start_stream()

            if not self.standalone_camera_app.streaming:

                self.standalone_camera_app.setVisible(False)

                return

            self.rear_toggle_btn.setText("Stop")

        else:

            self.standalone_camera_app.stop_stream()

            self.standalone_camera_app.setVisible(False)

            self.rear_toggle_btn.setText("Front Cam")



    def _start_aux_process(self):

        return



    def _stop_aux_process(self):

        return



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



    def update_lidar_ui(self, upd: LidarUpdate):

        self.lidar_status.setText("OK" if upd.status == "OK" else upd.status)

        if upd.min_dist is not None:

            self.min_line.setText(f"{upd.min_dist:.0f} mm @ {upd.min_angle:.1f}°")

        else:

            self.min_line.setText("—")

        if upd.max_dist is not None:

            self.max_line.setText(f"{upd.max_dist:.0f} mm @ {upd.max_angle:.1f}°")

        else:

            self.max_line.setText("—")

        d = upd.direction or "unknown"

        if upd.zone == "stop":

            self.zone_line.setText(f"STOP ({d})")

        elif upd.zone == "close":

            self.zone_line.setText(f"CLOSE ({d})")

        else:

            self.zone_line.setText("CLEAR")

        if self.speech_toggle.isChecked():

            self.speech_gate.consider(upd.zone, upd.direction)

        self.thresh_line.setText(self._format_thresholds())

    

    def stop_workers(self):

        self.lidar_worker.stop()

        self.lidar_thread.quit()

        self.lidar_thread.wait()

        self.tts_stop_event.set()

        self.detect_stop_event.set()

        if hasattr(self, 'camera_app'):

            self.camera_app.stop_stream()

        if hasattr(self, 'standalone_camera_app'):

            self.standalone_camera_app.stop_stream()





class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Combined GUI")

        self.setGeometry(100, 100, 1400, 900)

        self.main_view = CombinedView()

        self.setCentralWidget(self.main_view)

        self.batch_process = None

        self.setStyleSheet("""

            QWidget { background: #0B0F14; color: #E7EEF7; font-family: -apple-system, Segoe UI, Roboto; font-size: 14px; }

        """)



    def closeEvent(self, event):

        self.main_view.stop_workers()

        event.accept()



if __name__ == "__main__":

    try:

        app = QApplication(sys.argv)

        window = MainWindow()

        window.show()

        sys.exit(app.exec())

    finally:

        if ON_PI:

            GPIO.cleanup()


