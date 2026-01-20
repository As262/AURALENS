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

HAS_SOLUTIONS = hasattr(mp, "solutions")
if not HAS_SOLUTIONS:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision


class EyeTracker:
    def __init__(self):
        self.max_fps = 24
        self.process_every = 2
        self.smoothing = 0.35
        self.cursor_update_interval = 0.03
        self.blink_threshold = 0.20
        self.short_blink_min = 0.05
        self.long_blink_threshold = 0.7
        self.click_debounce = 0.35

        self._cursor = CursorController()
        self._last_move_ts = 0.0
        self._last_click_ts = 0.0
        self._blink_start = None
        self._smoothed_x = None
        self._smoothed_y = None

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
        left = self._eye_center(landmarks, LEFT_EYE)
        right = self._eye_center(landmarks, RIGHT_EYE)
        gaze_x = (left[0] + right[0]) * 0.5
        gaze_y = (left[1] + right[1]) * 0.5

        nx = self._normalize(gaze_x, 0.2, 0.8)
        ny = self._normalize(gaze_y, 0.2, 0.8)

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
        if ear < self.blink_threshold:
            if self._blink_start is None:
                self._blink_start = now
        else:
            if self._blink_start is not None:
                duration = now - self._blink_start
                self._blink_start = None
                if now - self._last_click_ts >= self.click_debounce:
                    if duration >= self.short_blink_min:
                        if duration >= self.long_blink_threshold:
                            self._cursor.right_click()
                        else:
                            self._cursor.left_click()
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
