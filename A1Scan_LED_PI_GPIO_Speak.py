import time
import threading
import queue
from dataclasses import dataclass

from rplidar import RPLidar

# ----------------------------
# OPTIONAL: Raspberry Pi GPIO (fallback-friendly)
# ----------------------------
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except Exception:
    ON_PI = False
    class _DummyGPIO:
        BCM = OUT = HIGH = LOW = None
        def setwarnings(self, *_): pass
        def setmode(self, *_): pass
        def setup(self, *_ , **__): pass
        def output(self, *_): pass
        def cleanup(self): pass
    GPIO = _DummyGPIO()

import tkinter as tk
from tkinter import ttk

# ----------------------------
# Text-to-speech (pyttsx3)
# ----------------------------
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    pyttsx3 = None
    TTS_AVAILABLE = False

# ----------------------------
# Config
# ----------------------------
PORT = '/dev/ttyUSB0'  # RPLidar port

STOP_MIN = 300
STOP_MAX_INDOOR = 600
CLOSE_MIN_INDOOR = 600
CLOSE_MAX_INDOOR = 1000

MESSAGE_INTERVAL = 1

# Speak behavior
SPEAK_REPEAT_SECONDS = 60   # repeat if same zone+direction persists
SPEAK_RATE = 150            # pyttsx3 rate

# ----------------------------
# GPIO settings (BCM numbering)
# ----------------------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

PIN_R = 16  # Red LED
PIN_Y = 21  # Yellow LED
PIN_G = 12  # Green LED

GPIO.setup(PIN_R, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PIN_Y, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PIN_G, GPIO.OUT, initial=GPIO.LOW)


def set_led_state(state_char: str):
    GPIO.output(PIN_R, GPIO.HIGH if state_char == "R" else GPIO.LOW)
    GPIO.output(PIN_Y, GPIO.HIGH if state_char == "Y" else GPIO.LOW)
    GPIO.output(PIN_G, GPIO.HIGH if state_char == "G" else GPIO.LOW)


def get_direction(angle_deg: float) -> str:
    angle_deg = angle_deg % 360
    if angle_deg <= 45 or angle_deg > 315:
        return "front"
    elif 45 < angle_deg <= 135:
        return "right"
    elif 135 < angle_deg <= 225:
        return "back"
    else:
        return "left"


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
    zone: str = "clear"             # "stop" | "close" | "clear"
    zone_angle: float | None = None
    direction: str | None = None
    thresholds: Thresholds | None = None
    status: str = "OK"


class LidarWorker(threading.Thread):
    def __init__(self, updates_q: queue.Queue, thresholds_ref: dict, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.updates_q = updates_q
        self.thresholds_ref = thresholds_ref  # dict holding current thresholds
        self.stop_event = stop_event

        self.last_msg_time = 0
        self.last_zone = None
        self.last_led_state = None

    def run(self):
        self._set_led("G")

        while not self.stop_event.is_set():
            lidar = None
            try:
                lidar = RPLidar(PORT)

                for scan in lidar.iter_scans():
                    if self.stop_event.is_set():
                        break

                    distances = []
                    angles = []

                    for (_, angle, distance) in scan:
                        if distance > 0:
                            distances.append(distance)
                            angles.append(angle)

                    valid_pairs = [(d, a) for d, a in zip(distances, angles) if d > 0]

                    upd = LidarUpdate()
                    upd.thresholds = self.thresholds_ref["t"]

                    if valid_pairs:
                        min_dist, min_angle = min(valid_pairs, key=lambda x: x[0])
                        max_dist, max_angle = max(valid_pairs, key=lambda x: x[0])
                        upd.min_dist, upd.min_angle = float(min_dist), float(min_angle)
                        upd.max_dist, upd.max_angle = float(max_dist), float(max_angle)

                    zone = "clear"
                    closest_angle_for_zone = None

                    t = self.thresholds_ref["t"]
                    if valid_pairs:
                        stop_candidates = [(d, a) for d, a in valid_pairs if t.stop_min <= d < t.stop_max]
                        close_candidates = [(d, a) for d, a in valid_pairs if t.close_min <= d < t.close_max]

                        if stop_candidates:
                            _, closest_angle_for_zone = min(stop_candidates, key=lambda x: x[0])
                            zone = "stop"
                        elif close_candidates:
                            _, closest_angle_for_zone = min(close_candidates, key=lambda x: x[0])
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

                    now = time.time()
                    if zone != "clear" and closest_angle_for_zone is not None:
                        if (now - self.last_msg_time >= MESSAGE_INTERVAL) or (zone != self.last_zone):
                            self.last_msg_time = now
                            self.last_zone = zone
                    else:
                        self.last_zone = None

                    self._push_update(upd)

            except Exception as e:
                self._push_update(LidarUpdate(status=f"Error: {e} (restarting)"))
                time.sleep(1)

            finally:
                if lidar is not None:
                    try: lidar.stop()
                    except Exception: pass
                    try: lidar.disconnect()
                    except Exception: pass

        self._set_led("G")

    def _push_update(self, upd: LidarUpdate):
        try:
            while self.updates_q.qsize() > 5:
                self.updates_q.get_nowait()
        except Exception:
            pass
        self.updates_q.put(upd)

    def _set_led(self, state_char: str):
        if state_char != self.last_led_state:
            set_led_state(state_char)
            self.last_led_state = state_char


class TTSWorker(threading.Thread):
    """Speaks queued messages without blocking the GUI."""
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
                # If TTS fails mid-run, just skip and continue
                pass

        # best effort shutdown
        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception:
            pass


class SpeechGate:
    """
    Only speaks for stop/close.
    - If zone+direction changes: speak immediately
    - If unchanged: speak again every SPEAK_REPEAT_SECONDS
    - Never speak for clear
    """
    def __init__(self, tts_q: queue.Queue):
        self.tts_q = tts_q
        self.last_key: tuple[str, str] | None = None   # (zone, direction)
        self.last_spoken_by_key: dict[tuple[str, str], float] = {}

    def _phrase(self, zone: str, direction: str) -> str | None:
        if zone == "close":
            return f"Obstacle close at {direction}, careful!"
        if zone == "stop":
            return f"Obstacle in proximity at {direction}, stop!"
        return None

    def consider(self, zone: str, direction: str | None):
        if zone not in ("close", "stop"):
            self.last_key = None
            return

        if direction is None:
            direction = "front"

        now = time.time()
        key = (zone, direction)
        last_time = self.last_spoken_by_key.get(key, 0.0)

        if last_time and (now - last_time) < SPEAK_REPEAT_SECONDS:
            return

        phrase = self._phrase(zone, direction)
        if phrase:
            self.tts_q.put(phrase)
            self.last_key = key
            self.last_spoken_by_key[key] = now


class ModePicker(ttk.Frame):
    """First screen: choose Indoor / Outdoor."""
    def __init__(self, master, on_pick):
        super().__init__(master, padding=24)
        self.on_pick = on_pick
        self.pack(fill="both", expand=True)

        ttk.Label(self, text="Choose Mode", style="Title.TLabel").pack(anchor="w")
        ttk.Label(self, text="Select environment before starting the LiDAR.", style="Small.TLabel").pack(anchor="w", pady=(6, 18))

        btns = ttk.Frame(self)
        btns.pack(fill="x")

        ttk.Button(btns, text="Indoor", style="Accent.TButton", command=lambda: self.on_pick("Indoor")).pack(fill="x", pady=(0, 10))
        ttk.Button(btns, text="Outdoor", style="TButton", command=lambda: self.on_pick("Outdoor")).pack(fill="x")


class Dashboard(ttk.Frame):
    """Second screen: live data."""
    def __init__(self, master, thresholds_ref, start_mode: str):
        super().__init__(master, padding=16)
        self.thresholds_ref = thresholds_ref
        self.mode = tk.StringVar(value=start_mode)
        self.status = tk.StringVar(value="Starting…")

        self.min_line = tk.StringVar(value="—")
        self.max_line = tk.StringVar(value="—")
        self.zone_line = tk.StringVar(value="CLEAR")
        self.thresh_line = tk.StringVar(value=self._format_thresholds())

        self.pack(fill="both", expand=True)
        self._build_ui()

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x")

        ttk.Label(header, text="LiDAR Monitor", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status, style="Small.TLabel").pack(side="right")

        ttk.Separator(self).pack(fill="x", pady=(12, 12))

        content = ttk.Frame(self)
        content.pack(fill="both", expand=True)

        left = ttk.Frame(content)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(content)
        right.pack(side="right", fill="y", padx=(16, 0))

        ttk.Label(left, text="Min (nearest)").pack(anchor="w")
        ttk.Label(left, textvariable=self.min_line).pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Max (farthest)").pack(anchor="w")
        ttk.Label(left, textvariable=self.max_line).pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Zone").pack(anchor="w")
        ttk.Label(left, textvariable=self.zone_line, font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))

        ttk.Label(left, text="Thresholds").pack(anchor="w")
        ttk.Label(left, textvariable=self.thresh_line).pack(anchor="w")

        ttk.Label(right, text="Mode").pack(anchor="w", pady=(0, 8))
        self.btn_indoor = ttk.Button(right, text="Indoor", command=self.set_indoor)
        self.btn_outdoor = ttk.Button(right, text="Outdoor", command=self.set_outdoor)
        self.btn_indoor.pack(fill="x", pady=(0, 10))
        self.btn_outdoor.pack(fill="x")

        # highlight current
        if self.mode.get() == "Indoor":
            self.btn_indoor.configure(style="Accent.TButton")
        else:
            self.btn_outdoor.configure(style="Accent.TButton")

    def _format_thresholds(self):
        t = self.thresholds_ref["t"]
        return f"STOP: {t.stop_min}-{t.stop_max} mm | CLOSE: {t.close_min}-{t.close_max} mm"

    def set_indoor(self):
        self.mode.set("Indoor")
        self.thresholds_ref["t"] = Thresholds(
            stop_min=STOP_MIN,
            stop_max=STOP_MAX_INDOOR,
            close_min=CLOSE_MIN_INDOOR,
            close_max=CLOSE_MAX_INDOOR
        )
        self.thresh_line.set(self._format_thresholds())
        self.btn_indoor.configure(style="Accent.TButton")
        self.btn_outdoor.configure(style="TButton")

    def set_outdoor(self):
        self.mode.set("Outdoor")
        self.thresholds_ref["t"] = Thresholds(
            stop_min=STOP_MIN,
            stop_max=STOP_MAX_INDOOR * 2,
            close_min=CLOSE_MIN_INDOOR * 2,
            close_max=CLOSE_MAX_INDOOR * 2
        )
        self.thresh_line.set(self._format_thresholds())
        self.btn_outdoor.configure(style="Accent.TButton")
        self.btn_indoor.configure(style="TButton")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LiDAR Monitor")
        self.geometry("560x320")
        self.minsize(520, 300)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 11))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 11), padding=10)
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=10)

        self.updates_q = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = None

        # TTS
        self.tts_q = queue.Queue()
        self.tts_worker = TTSWorker(self.tts_q, self.stop_event)
        self.tts_worker.start()
        self.speech_gate = SpeechGate(self.tts_q)

        # thresholds reference used by worker thread
        self.thresholds_ref = {
            "t": Thresholds(
                stop_min=STOP_MIN,
                stop_max=STOP_MAX_INDOOR,
                close_min=CLOSE_MIN_INDOOR,
                close_max=CLOSE_MAX_INDOOR
            )
        }

        self.current_view = None
        self.dashboard = None

        # First screen: mode picker
        self.show_mode_picker()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_mode_picker(self):
        if self.current_view is not None:
            self.current_view.destroy()
        self.current_view = ModePicker(self, on_pick=self.start_dashboard)

    def start_dashboard(self, mode: str):
        # Set initial thresholds based on chosen mode BEFORE starting lidar
        if mode == "Outdoor":
            self.thresholds_ref["t"] = Thresholds(
                stop_min=STOP_MIN,
                stop_max=STOP_MAX_INDOOR * 2,
                close_min=CLOSE_MIN_INDOOR * 2,
                close_max=CLOSE_MAX_INDOOR * 2
            )
        else:
            self.thresholds_ref["t"] = Thresholds(
                stop_min=STOP_MIN,
                stop_max=STOP_MAX_INDOOR,
                close_min=CLOSE_MIN_INDOOR,
                close_max=CLOSE_MAX_INDOOR
            )

        if self.current_view is not None:
            self.current_view.destroy()

        self.dashboard = Dashboard(self, thresholds_ref=self.thresholds_ref, start_mode=mode)
        self.current_view = self.dashboard

        # Start worker ONLY after mode chosen
        set_led_state("G")
        self.worker = LidarWorker(self.updates_q, self.thresholds_ref, self.stop_event)
        self.worker.start()

        self.after(50, self._poll_updates)

    def _poll_updates(self):
        if self.stop_event.is_set():
            return

        latest = None
        try:
            while True:
                latest = self.updates_q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None and self.dashboard is not None:
            self.dashboard.status.set("OK" if latest.status == "OK" else latest.status)

            if latest.min_dist is not None:
                self.dashboard.min_line.set(f"{latest.min_dist:.0f} mm @ {latest.min_angle:.1f}°")
            else:
                self.dashboard.min_line.set("—")

            if latest.max_dist is not None:
                self.dashboard.max_line.set(f"{latest.max_dist:.0f} mm @ {latest.max_angle:.1f}°")
            else:
                self.dashboard.max_line.set("—")

            # UI text (includes direction)
            if latest.zone == "stop":
                d = latest.direction or "unknown"
                self.dashboard.zone_line.set(f"STOP ({d})")
            elif latest.zone == "close":
                d = latest.direction or "unknown"
                self.dashboard.zone_line.set(f"CLOSE ({d})")
            else:
                self.dashboard.zone_line.set("CLEAR")

            # Speak logic (only stop/close; repeats every 1 min if unchanged; immediate on change)
            self.speech_gate.consider(latest.zone, latest.direction)

            # Keep thresholds text synced (in case mode buttons used)
            self.dashboard.thresh_line.set(self.dashboard._format_thresholds())

        self.after(60, self._poll_updates)

    def on_close(self):
        try:
            self.stop_event.set()
            time.sleep(0.15)
        except Exception:
            pass
        finally:
            try:
                # Let TTS worker exit cleanly
                try:
                    self.tts_q.put_nowait(None)
                except Exception:
                    pass
                set_led_state("G")
            except Exception:
                pass
            try:
                GPIO.cleanup()
            except Exception:
                pass
            self.destroy()


if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            set_led_state("G")
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
