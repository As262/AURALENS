"""Shared MediaPipe face-landmark processor + frame preprocessing.

cv2 and mediapipe are imported lazily (inside functions / __init__) so that
processes which never touch the camera (tray, settings, overlay) don't pay the
multi-second, multi-hundred-MB import cost when spawn re-imports their module.

Both the live tracker and the calibration step go through this module so their
features come from identically-preprocessed frames.
"""
import time
import urllib.request
from pathlib import Path

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model():
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "face_landmarker.task"
    if not model_path.exists():
        urllib.request.urlretrieve(_MODEL_URL, model_path)
    return model_path


def make_clahe():
    """Create the CLAHE object used for adaptive lighting normalisation."""
    import cv2
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(frame_bgr, clahe):
    """BGR camera frame -> CLAHE-normalised RGB image for MediaPipe.
    This preprocessing is part of the gaze feature; calibration and live
    tracking MUST both use it."""
    import cv2
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


class FaceProcessor:
    """Wraps either the MediaPipe `solutions` FaceMesh API or, when that's
    unavailable, the newer Tasks FaceLandmarker API. `process()` returns a list
    of landmarks (each with .x/.y/.z) or None."""

    def __init__(self):
        import mediapipe as mp
        self._mp = mp
        self._has_solutions = hasattr(mp, "solutions")
        self._face_mesh = None
        self._landmarker = None
        self._last_ts = -1   # monotonic timestamp guard for VIDEO mode

        if self._has_solutions:
            # solutions FaceMesh already does temporal tracking via process()
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,   # iris landmarks
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            base_options = mp_python.BaseOptions(model_asset_path=str(_ensure_model()))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,   # temporal tracking
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _next_ts(self):
        # VIDEO mode requires strictly increasing timestamps (ms).
        ts = int(time.monotonic() * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, rgb_image):
        if self._has_solutions:
            results = self._face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
            return None
        mp = self._mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect_for_video(mp_image, self._next_ts())
        if result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()
