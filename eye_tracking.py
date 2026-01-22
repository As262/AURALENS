import os
import time
import math
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp

from cursor_control import CursorController

os.environ.setdefault("OMP_NUM_THREADS", "1")

LEFT_EYE = [33, 133, 159, 145, 158, 144]
RIGHT_EYE = [362, 263, 386, 374, 385, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

HAS_SOLUTIONS = hasattr(mp, "solutions")
if not HAS_SOLUTIONS:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision


class EyeTracker:
    def __init__(self):
        self.max_fps = 60
        self.process_every = 1
        self.smoothing = 0.08
        self.cursor_update_interval = 0.025
        self.blink_threshold = 0.20
        self.short_blink_min = 0.05
        self.long_blink_threshold = 0.7
        self.click_debounce = 0.80
        self.double_blink_window = 0.45
        self.sensitivity_x = 2.5
        self.sensitivity_y = 2.0
        self.head_weight = 0.7
        self.iris_weight = 0.3

        self._cursor = CursorController()
        self._last_move_ts = 0.0
        self._last_click_ts = 0.0
        self._blink_start = None
        self._smoothed_x = None
        self._smoothed_y = None
        self._pending_short_blink_ts = None
        self._neutral_nx = None
        self._neutral_ny = None
        self._neutral_drift = 0.0
        self._is_blinking = False
        self._calibration_frames = 60
        self._frame_calibration_count = 0
        self._neutral_head_x = None
        self._neutral_head_y = None

    def run(self, stop_event):
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, self.max_fps)

        if not cap.isOpened():
            return

        face_mesh = None
        face_landmarker = None
        if HAS_SOLUTIONS:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        else:
            face_landmarker = self._create_face_landmarker()

        desired_dt = 1.0 / self.max_fps
        frame_count = 0

        try:
            while not stop_event.is_set():
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                if frame_count % self.process_every != 0:
                    self._sleep_if_needed(loop_start, desired_dt)
                    continue

                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=18)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if HAS_SOLUTIONS:
                    results = face_mesh.process(rgb)
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        self._update_cursor(landmarks)
                        self._update_blink(landmarks)
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = face_landmarker.detect(mp_image)
                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                        self._update_cursor(landmarks)
                        self._update_blink(landmarks)

                self._sleep_if_needed(loop_start, desired_dt)
        finally:
            if face_mesh is not None:
                face_mesh.close()
            if face_landmarker is not None:
                face_landmarker.close()
            cap.release()

    def _update_cursor(self, landmarks):
        if self._is_blinking:
            return

        nx, ny = self._gaze_from_landmarks(landmarks)

        screen_w, screen_h = self._cursor.screen_size
        target_x = nx * screen_w
        target_y = ny * screen_h

        if self._smoothed_x is None:
            self._smoothed_x = target_x
            self._smoothed_y = target_y
        else:
            self._smoothed_x = self.smoothing * target_x + (1 - self.smoothing) * self._smoothed_x
            self._smoothed_y = self.smoothing * target_y + (1 - self.smoothing) * self._smoothed_y

        now = time.time()
        if now - self._last_move_ts >= self.cursor_update_interval:
            self._cursor.move(self._smoothed_x, self._smoothed_y)
            self._last_move_ts = now

    def _update_blink(self, landmarks):
        ear_left = self._eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_right = self._eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_left + ear_right) * 0.5

        now = time.time()
        if self._pending_short_blink_ts and now - self._pending_short_blink_ts > self.double_blink_window:
            self._pending_short_blink_ts = None

        if ear < self.blink_threshold:
            self._is_blinking = True
            if self._blink_start is None:
                self._blink_start = now
        else:
            self._is_blinking = False
            if self._blink_start is not None:
                duration = now - self._blink_start
                self._blink_start = None
                if now - self._last_click_ts >= self.click_debounce:
                    if duration >= self.short_blink_min:
                        if duration >= self.long_blink_threshold:
                            self._cursor.right_click()
                            self._pending_short_blink_ts = None
                        else:
                            if self._pending_short_blink_ts and now - self._pending_short_blink_ts <= self.double_blink_window:
                                self._cursor.left_click()
                                self._pending_short_blink_ts = None
                            else:
                                self._pending_short_blink_ts = now
                        self._last_click_ts = now

    def _create_face_landmarker(self):
        model_path = self._ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _gaze_from_landmarks(self, landmarks):
        nose = landmarks[1]
        head_x = nose.x
        head_y = nose.y

        left_iris = self._iris_center(landmarks, LEFT_IRIS, LEFT_EYE)
        right_iris = self._iris_center(landmarks, RIGHT_IRIS, RIGHT_EYE)

        left_corners = self._eye_corners(landmarks, 33, 133)
        right_corners = self._eye_corners(landmarks, 362, 263)
        left_vert = self._eye_vertical_bounds(landmarks, 159, 158, 145, 144)
        right_vert = self._eye_vertical_bounds(landmarks, 386, 385, 374, 380)

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

        if self._frame_calibration_count < self._calibration_frames:
            if self._neutral_nx is None:
                self._neutral_nx = iris_x
                self._neutral_ny = iris_y
                self._neutral_head_x = head_x
                self._neutral_head_y = head_y
            else:
                n = self._frame_calibration_count
                self._neutral_nx = (self._neutral_nx * n + iris_x) / (n + 1)
                self._neutral_ny = (self._neutral_ny * n + iris_y) / (n + 1)
                self._neutral_head_x = (self._neutral_head_x * n + head_x) / (n + 1)
                self._neutral_head_y = (self._neutral_head_y * n + head_y) / (n + 1)
            self._frame_calibration_count += 1
            return 0.5, 0.5

        delta_iris_x = iris_x - self._neutral_nx
        delta_iris_y = iris_y - self._neutral_ny
        delta_head_x = head_x - self._neutral_head_x
        delta_head_y = head_y - self._neutral_head_y

        combined_x = self.iris_weight * delta_iris_x + self.head_weight * delta_head_x
        combined_y = self.iris_weight * delta_iris_y + self.head_weight * delta_head_y

        nx = 0.5 - combined_x * self.sensitivity_x
        ny = 0.5 + combined_y * self.sensitivity_y

        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return nx, ny

    @staticmethod
    def _ensure_model():
        models_dir = Path(__file__).resolve().parent / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "face_landmarker.task"
        if not model_path.exists():
            url = (
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
        return model_path

    @staticmethod
    def _eye_center(landmarks, indices):
        x = 0.0
        y = 0.0
        for idx in indices:
            x += landmarks[idx].x
            y += landmarks[idx].y
        inv = 1.0 / len(indices)
        return x * inv, y * inv

    @staticmethod
    def _iris_center(landmarks, iris_indices, fallback_indices):
        if not landmarks or max(iris_indices) >= len(landmarks):
            return EyeTracker._eye_center(landmarks, fallback_indices)
        return EyeTracker._eye_center(landmarks, iris_indices)

    @staticmethod
    def _eye_corners(landmarks, left_idx, right_idx):
        left = landmarks[left_idx]
        right = landmarks[right_idx]
        return left.x, right.x

    @staticmethod
    def _eye_vertical_bounds(landmarks, top_idx1, top_idx2, bot_idx1, bot_idx2):
        top = (landmarks[top_idx1].y + landmarks[top_idx2].y) * 0.5
        bottom = (landmarks[bot_idx1].y + landmarks[bot_idx2].y) * 0.5
        return top, bottom

    @staticmethod
    def _ratio(value, a, b):
        denom = b - a
        if abs(denom) < 1e-6:
            return 0.5
        return (value - a) / denom

    @staticmethod
    def _relative_pos(value, center, span):
        if span <= 1e-6:
            return 0.5
        return 0.5 + (value - center) / span

    @staticmethod
    def _eye_aspect_ratio(landmarks, indices):
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

    @staticmethod
    def _normalize(value, min_val, max_val):
        if value < min_val:
            value = min_val
        elif value > max_val:
            value = max_val
        return (value - min_val) / (max_val - min_val)

    @staticmethod
    def _sleep_if_needed(start_ts, desired_dt):
        elapsed = time.time() - start_ts
        if elapsed < desired_dt:
            time.sleep(desired_dt - elapsed)
