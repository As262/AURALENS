"""Persistent configuration + the shared-memory state bundle.

The SharedState holds every live-tunable value as an mp.Value. It is created
ONCE by the tray process and passed to every child; children must never create
their own SharedState or they'd get unshared memory. The calibration model is
NOT kept here (it is structured data that changes rarely) - it lives in
config.json and is reloaded by restarting the eye process.
"""
import json
import multiprocessing as mp
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

# Scalar tunables (the keys that map onto SharedState mp.Values)
DEFAULTS = {
    "tracking_enabled": 1,     # int flag
    "light_on": 0,             # int flag
    "light_mode": 0,           # 0 = glow border, 1 = full panel
    "brightness": 0.85,        # 0.0 - 1.0 (key-light intensity)
    "temp_k": 4500.0,          # 2000 - 6500 Kelvin
    "overlay_opacity": 0.85,   # feedback (gaze cursor) window alpha
    "sensitivity_x": 6.5,      # heuristic-fallback sensitivity only
    "sensitivity_y": 9.0,
    "dwell_enabled": 0,        # dwell-to-click (hold gaze still to click)
    "dwell_time": 1.0,         # seconds of holding still before a click
}

_INT_KEYS = ("tracking_enabled", "light_on", "light_mode", "dwell_enabled")


def load():
    """Load config.json, falling back to defaults for missing keys.
    Returns a dict that also carries the (optional) 'calibration' block."""
    cfg = dict(DEFAULTS)
    cfg["calibration"] = None
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k in DEFAULTS:
                if k in data:
                    cfg[k] = data[k]
            if data.get("calibration"):
                cfg["calibration"] = data["calibration"]
        except Exception:
            pass
    return cfg


def save(cfg):
    """Persist the scalar tunables + calibration model to config.json."""
    out = {k: cfg.get(k, DEFAULTS[k]) for k in DEFAULTS}
    out["calibration"] = cfg.get("calibration")
    try:
        CONFIG_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        pass


class SharedState:
    """Bundle of mp.Value shared between the tray and all child processes."""

    def __init__(self):
        self.gaze_x = mp.Value('d', 0.5)
        self.gaze_y = mp.Value('d', 0.5)
        # click_state: 0=none, 1=left, 2=right, 3=blink-freeze
        self.click_state = mp.Value('i', 0)

        self.tracking_enabled = mp.Value('i', 1)
        self.light_on = mp.Value('i', 0)
        self.light_mode = mp.Value('i', 0)
        self.brightness = mp.Value('d', 0.85)
        self.temp_k = mp.Value('d', 4500.0)
        self.overlay_opacity = mp.Value('d', 0.85)
        self.sensitivity_x = mp.Value('d', 6.5)
        self.sensitivity_y = mp.Value('d', 9.0)

        # Dwell-to-click
        self.dwell_enabled = mp.Value('i', 0)
        self.dwell_time = mp.Value('d', 1.0)
        self.dwell_progress = mp.Value('d', 0.0)   # runtime only (eye -> overlay)

        # Settings -> tray request flag (watched by the tray)
        self.recalibrate_request = mp.Value('i', 0)
        # Set by the tray while the calibration window is up, so the overlay
        # hides its cursor and stops raising itself over the calibration screen.
        self.calibrating = mp.Value('i', 0)


def push_to_shared(cfg, shared):
    """Copy scalar config values into the shared mp.Values."""
    shared.tracking_enabled.value = int(cfg["tracking_enabled"])
    shared.light_on.value = int(cfg["light_on"])
    shared.light_mode.value = int(cfg["light_mode"])
    shared.brightness.value = float(cfg["brightness"])
    shared.temp_k.value = float(cfg["temp_k"])
    shared.overlay_opacity.value = float(cfg["overlay_opacity"])
    shared.sensitivity_x.value = float(cfg["sensitivity_x"])
    shared.sensitivity_y.value = float(cfg["sensitivity_y"])
    shared.dwell_enabled.value = int(cfg["dwell_enabled"])
    shared.dwell_time.value = float(cfg["dwell_time"])


def pull_from_shared(shared, cfg):
    """Copy the live shared values back into a config dict (for persistence).
    Leaves the 'calibration' block untouched."""
    cfg["tracking_enabled"] = int(shared.tracking_enabled.value)
    cfg["light_on"] = int(shared.light_on.value)
    cfg["light_mode"] = int(shared.light_mode.value)
    cfg["brightness"] = float(shared.brightness.value)
    cfg["temp_k"] = float(shared.temp_k.value)
    cfg["overlay_opacity"] = float(shared.overlay_opacity.value)
    cfg["sensitivity_x"] = float(shared.sensitivity_x.value)
    cfg["sensitivity_y"] = float(shared.sensitivity_y.value)
    cfg["dwell_enabled"] = int(shared.dwell_enabled.value)
    cfg["dwell_time"] = float(shared.dwell_time.value)
    return cfg
