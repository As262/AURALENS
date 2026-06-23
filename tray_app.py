"""System-tray entry point.

Runs the pystray icon on the main thread and owns the lifecycle of the child
processes (eye tracker, overlay, settings, calibration). Keep this module's
top-level imports light (no cv2/mediapipe/pystray) because `spawn` re-imports it
in every child; the heavy imports live inside the target wrappers below.
"""
import multiprocessing as mp
import threading
import time

import config
from config import SharedState


# --- picklable process targets (heavy imports happen lazily inside) -------- #
def _eye_target(stop_event, shared, calibration):
    from eye_tracking import EyeTracker
    EyeTracker(shared, calibration).run(stop_event)


def _overlay_target(stop_event, shared):
    from aura_overlay import run as run_overlay
    run_overlay(stop_event, shared)


def _settings_target(shared):
    from settings import run as run_settings
    run_settings(shared)


def _calibration_target(stop_event, shared):
    from calibration import run_calibration
    run_calibration(stop_event, shared)


class TrayApp:
    def __init__(self):
        self.shared = SharedState()
        self.cfg = config.load()
        config.push_to_shared(self.cfg, self.shared)

        self.quit_event = mp.Event()
        self._lock = threading.RLock()
        self._busy = False  # recalibration in progress

        self.eye_proc = None
        self.eye_stop = None
        self.overlay_proc = None
        self.overlay_stop = None
        self.settings_proc = None
        self.icon = None

    # ---- process lifecycle ------------------------------------------- #
    def start_eye(self):
        with self._lock:
            if self.eye_proc and self.eye_proc.is_alive():
                return
            self.eye_stop = mp.Event()
            calib = self.cfg.get("calibration")
            self.eye_proc = mp.Process(target=_eye_target,
                                       args=(self.eye_stop, self.shared, calib),
                                       name="eye-tracking", daemon=True)
            self.eye_proc.start()

    def stop_eye(self):
        with self._lock:
            proc = self.eye_proc
            stop = self.eye_stop
            self.eye_proc = None
        if proc:
            try:
                stop.set()
            except Exception:
                pass
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

    def start_overlay(self):
        with self._lock:
            if self.overlay_proc and self.overlay_proc.is_alive():
                return
            self.overlay_stop = mp.Event()
            self.overlay_proc = mp.Process(target=_overlay_target,
                                           args=(self.overlay_stop, self.shared),
                                           name="overlay", daemon=True)
            self.overlay_proc.start()

    def stop_overlay(self):
        with self._lock:
            proc = self.overlay_proc
            stop = self.overlay_stop
            self.overlay_proc = None
        if proc:
            try:
                stop.set()
            except Exception:
                pass
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

    # ---- persistence -------------------------------------------------- #
    def _persist(self):
        config.pull_from_shared(self.shared, self.cfg)
        config.save(self.cfg)

    # ---- menu actions ------------------------------------------------- #
    def on_toggle_tracking(self, icon, item):
        self.shared.tracking_enabled.value = 0 if self.shared.tracking_enabled.value else 1
        self._persist()

    def on_toggle_light(self, icon, item):
        self.shared.light_on.value = 0 if self.shared.light_on.value else 1
        self._persist()

    def on_toggle_full(self, icon, item):
        self.shared.light_mode.value = 0 if self.shared.light_mode.value else 1
        self._persist()

    def on_settings(self, icon, item):
        with self._lock:
            if self.settings_proc and self.settings_proc.is_alive():
                return
            self.settings_proc = mp.Process(target=_settings_target,
                                            args=(self.shared,), name="settings", daemon=True)
            self.settings_proc.start()

    def on_recalibrate(self, icon, item):
        threading.Thread(target=self._do_recalibrate, daemon=True).start()

    def _do_recalibrate(self):
        with self._lock:
            if self._busy:
                return
            self._busy = True
        try:
            self.shared.calibrating.value = 1  # overlay hides cursor / stops lifting
            self.stop_eye()
            time.sleep(0.4)  # let DirectShow fully release the webcam
            stop_event = mp.Event()
            proc = mp.Process(target=_calibration_target, args=(stop_event, self.shared),
                              name="calibration", daemon=True)
            proc.start()
            proc.join()
            time.sleep(0.3)
            self.cfg = config.load()  # reload the freshly fitted model
        finally:
            self.shared.calibrating.value = 0
            if not self.quit_event.is_set():
                self.start_eye()
            with self._lock:
                self._busy = False

    def on_quit(self, icon, item):
        self.quit_event.set()
        self.stop_eye()
        self.stop_overlay()
        with self._lock:
            if self.settings_proc and self.settings_proc.is_alive():
                self.settings_proc.terminate()
        try:
            icon.stop()
        except Exception:
            pass

    # ---- watcher: settings -> recalibrate request --------------------- #
    def _watcher(self):
        while not self.quit_event.is_set():
            try:
                if self.shared.recalibrate_request.value:
                    self.shared.recalibrate_request.value = 0
                    self._do_recalibrate()
            except Exception:
                pass
            time.sleep(0.4)

    # ---- run ---------------------------------------------------------- #
    def _make_icon_image(self):
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 20, 60, 44), fill=(238, 238, 238, 255))   # sclera
        d.ellipse((23, 20, 41, 44), fill=(45, 120, 220, 255))   # iris
        d.ellipse((28, 27, 36, 37), fill=(15, 15, 15, 255))     # pupil
        return img

    def run(self):
        import pystray
        from pystray import MenuItem as item, Menu

        self.start_eye()
        self.start_overlay()
        threading.Thread(target=self._watcher, daemon=True).start()

        menu = Menu(
            item("Eye Tracking", self.on_toggle_tracking,
                 checked=lambda i: bool(self.shared.tracking_enabled.value)),
            item("Key Light", self.on_toggle_light,
                 checked=lambda i: bool(self.shared.light_on.value)),
            item("Full-panel Light", self.on_toggle_full,
                 checked=lambda i: bool(self.shared.light_mode.value)),
            Menu.SEPARATOR,
            item("Settings…", self.on_settings),
            item("Recalibrate", self.on_recalibrate),
            Menu.SEPARATOR,
            item("Quit", self.on_quit),
        )
        self.icon = pystray.Icon("eye_tracker", self._make_icon_image(),
                                 "Eye Tracker", menu)
        self.icon.run()


def main():
    app = TrayApp()
    app.run()


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
