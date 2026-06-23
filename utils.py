"""Shared utilities used across processes.

Deliberately lightweight: imports only stdlib so the tray / settings / overlay
processes can import this without pulling in OpenCV or MediaPipe. The gaze
feature extraction here MUST stay byte-identical to what calibration and the
live tracker use, otherwise every prediction is offset.
"""
import ctypes
import math

# MediaPipe FaceMesh landmark indices (shared by tracker + calibration)
LEFT_EYE = [33, 133, 159, 145, 158, 144]
RIGHT_EYE = [362, 263, 386, 374, 385, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Blend weights: how much the iris-within-eye signal vs head pose contribute to
# the gaze proxy. Used by BOTH calibration and the live tracker so the fitted
# polynomial matches the live feature.
IRIS_WEIGHT = 0.45
HEAD_WEIGHT_X = 0.55
HEAD_WEIGHT_Y = 0.65


# --------------------------------------------------------------------------- #
# DPI awareness
# --------------------------------------------------------------------------- #
def enable_dpi_awareness():
    """Make this process DPI-aware so screen metrics, Tk geometry and
    SetCursorPos all agree on *physical* pixels. Must be called before any
    window is created or screen size is queried. Safe to call once per process;
    a second call (already-set) just fails silently.
    """
    try:
        # PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Colour temperature (Tanner Helland approximation), clamped to a sane range
# --------------------------------------------------------------------------- #
def kelvin_to_rgb(kelvin):
    """Convert a colour temperature in Kelvin to an (r, g, b) tuple (0-255)."""
    kelvin = max(1000.0, min(40000.0, float(kelvin)))
    temp = kelvin / 100.0

    if temp <= 66:
        red = 255.0
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)

    if temp <= 66:
        green = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)

    if temp >= 66:
        blue = 255.0
    elif temp <= 19:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307

    def clamp(v):
        return int(max(0, min(255, round(v))))

    return clamp(red), clamp(green), clamp(blue)


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def scale_rgb(rgb, frac):
    """Scale an rgb tuple toward black (frac 0..1) for a dimmer shade."""
    frac = max(0.0, min(1.0, frac))
    return tuple(int(max(0, min(255, c * frac))) for c in rgb)


# --------------------------------------------------------------------------- #
# One-Euro filter (adaptive smoothing for the gaze cursor)
# --------------------------------------------------------------------------- #
class _LowPass:
    __slots__ = ("s",)

    def __init__(self):
        self.s = None

    def __call__(self, x, alpha):
        if self.s is None:
            self.s = x
        else:
            self.s = alpha * x + (1.0 - alpha) * self.s
        return self.s

    def reset(self):
        self.s = None


class OneEuroFilter:
    """1-D One-Euro filter. Heavy smoothing when still, low lag when moving."""

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t = None
        self._last_x = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
        else:
            dt = 1.0 / 60.0
        self._last_t = t

        dx = 0.0 if self._last_x is None else (x - self._last_x) / dt
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        out = self._x(x, self._alpha(cutoff, dt))
        self._last_x = x
        return out

    def reset(self):
        self._x.reset()
        self._dx.reset()
        self._last_t = None
        self._last_x = None


class OneEuroFilter2D:
    def __init__(self, **kwargs):
        self.fx = OneEuroFilter(**kwargs)
        self.fy = OneEuroFilter(**kwargs)

    def __call__(self, x, y, t):
        return self.fx(x, t), self.fy(y, t)

    def reset(self):
        self.fx.reset()
        self.fy.reset()


# --------------------------------------------------------------------------- #
# Gaze feature extraction (shared by calibration and the live tracker)
# --------------------------------------------------------------------------- #
def _mean_xy(landmarks, indices):
    x = 0.0
    y = 0.0
    for idx in indices:
        x += landmarks[idx].x
        y += landmarks[idx].y
    inv = 1.0 / len(indices)
    return x * inv, y * inv


def _iris_center(landmarks, iris_indices, fallback_indices):
    if not landmarks or max(iris_indices) >= len(landmarks):
        return _mean_xy(landmarks, fallback_indices)
    return _mean_xy(landmarks, iris_indices)


def eye_aspect_ratio(landmarks, indices):
    """Vertical/horizontal ratio for blink detection (lower = more closed)."""
    p1 = landmarks[indices[0]]
    p4 = landmarks[indices[1]]
    p2 = landmarks[indices[2]]
    p6 = landmarks[indices[3]]
    p3 = landmarks[indices[4]]
    p5 = landmarks[indices[5]]

    def dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    vert = dist(p2, p6) + dist(p3, p5)
    horiz = dist(p1, p4)
    if horiz <= 1e-6:
        return 0.0
    return vert / (2.0 * horiz)


def extract_gaze_features(landmarks):
    """Return (iris_x, iris_y, head_x, head_y, ear) from face landmarks.

    iris_x/iris_y: iris offset within the eye, normalised by eye width/height.
    head_x/head_y: nose tip position in normalised image coords.
    ear: average eye-aspect-ratio (for blink detection).
    """
    nose = landmarks[1]
    head_x = nose.x
    head_y = nose.y

    left_iris = _iris_center(landmarks, LEFT_IRIS, LEFT_EYE)
    right_iris = _iris_center(landmarks, RIGHT_IRIS, RIGHT_EYE)

    left_corners = (landmarks[33].x, landmarks[133].x)
    right_corners = (landmarks[362].x, landmarks[263].x)
    left_vert = ((landmarks[159].y + landmarks[158].y) * 0.5,
                 (landmarks[145].y + landmarks[144].y) * 0.5)
    right_vert = ((landmarks[386].y + landmarks[385].y) * 0.5,
                  (landmarks[374].y + landmarks[380].y) * 0.5)

    left_w = abs(left_corners[1] - left_corners[0])
    right_w = abs(right_corners[1] - right_corners[0])
    left_h = abs(left_vert[1] - left_vert[0])
    right_h = abs(right_vert[1] - right_vert[0])

    left_mid_x = (left_corners[0] + left_corners[1]) * 0.5
    right_mid_x = (right_corners[0] + right_corners[1]) * 0.5
    left_mid_y = (left_vert[0] + left_vert[1]) * 0.5
    right_mid_y = (right_vert[0] + right_vert[1]) * 0.5

    iris_left_x = (left_iris[0] - left_mid_x) / max(left_w, 1e-6)
    iris_right_x = (right_iris[0] - right_mid_x) / max(right_w, 1e-6)
    iris_left_y = (left_iris[1] - left_mid_y) / max(left_h, 1e-6)
    iris_right_y = (right_iris[1] - right_mid_y) / max(right_h, 1e-6)

    iris_x = (iris_left_x + iris_right_x) * 0.5
    iris_y = (iris_left_y + iris_right_y) * 0.5

    ear_left = eye_aspect_ratio(landmarks, LEFT_EYE)
    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    ear = (ear_left + ear_right) * 0.5

    return iris_x, iris_y, head_x, head_y, ear


def blended_gaze(iris_x, iris_y, head_x, head_y):
    """Combine iris + head pose into a 2-D gaze proxy (gx, gy)."""
    gx = IRIS_WEIGHT * iris_x + HEAD_WEIGHT_X * head_x
    gy = IRIS_WEIGHT * iris_y + HEAD_WEIGHT_Y * head_y
    return gx, gy


def poly_features(gx, gy):
    """2nd-order polynomial basis used by the calibration mapping."""
    return [1.0, gx, gy, gx * gy, gx * gx, gy * gy]


def apply_calibration(gx, gy, model):
    """Map a (gx, gy) gaze proxy to screen coords (nx, ny) in [0, 1] using a
    fitted calibration model dict (coef_x, coef_y, mean, std)."""
    mean = model["mean"]
    std = model["std"]
    sgx = (gx - mean[0]) / (std[0] if std[0] else 1.0)
    sgy = (gy - mean[1]) / (std[1] if std[1] else 1.0)
    feats = poly_features(sgx, sgy)
    cx = model["coef_x"]
    cy = model["coef_y"]
    nx = sum(f * c for f, c in zip(feats, cx))
    ny = sum(f * c for f, c in zip(feats, cy))
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    return nx, ny
