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
    def __init__(self, shared_x=None, shared_y=None, shared_click_state=None):
        self.max_fps = 60
        self.process_every = 1
        self.smoothing = 0.10  # Slightly less smoothing for stability
        self.cursor_update_interval = 0.012
        # Blink detection - tuned thresholds
        self.blink_threshold = 0.23
        self.short_blink_min = 0.05
        self.long_blink_threshold = 0.55
        self.click_debounce = 0.3
        self.double_blink_window = 0.55
        # IMPORTANT: Freeze cursor during blink AND for a period after
        self.blink_freeze_duration = 0.15  # Freeze cursor 150ms after blink ends
        # Sensitivity
        self.sensitivity_x = 6.5
        self.sensitivity_y = 9.0
        self.head_weight = 0.55
        self.iris_weight = 0.45
        self.head_weight_vertical = 0.65
        # Non-linear amplification
        self.amplification_power = 1.5
        self.min_amplification = 1.0
        self.max_amplification = 2.2
        # Velocity prediction (reduced to improve stability)
        self.velocity_weight = 0.15
        self.velocity_smoothing = 0.2
        # Dead zone - ignore tiny movements
        self.dead_zone = 0.008
        # Face tracking recovery
        self._frames_without_face = 0
        self._max_frames_without_face = 10

        self._cursor = CursorController()
        self._last_move_ts = 0.0
        self._last_click_ts = 0.0
        self._blink_start = None
        self._blink_end_ts = None  # Track when blink ended for freeze period
        self._smoothed_x = None
        self._smoothed_y = None
        self._frozen_x = None  # Position to hold during blink
        self._frozen_y = None
        self._pending_short_blink_ts = None
        self._blink_count = 0
        self._neutral_nx = None
        self._neutral_ny = None
        self._neutral_drift = 0.0
        self._is_blinking = False
        self._calibration_frames = 45
        self._frame_calibration_count = 0
        self._neutral_head_x = None
        self._neutral_head_y = None
        # Velocity tracking
        self._prev_nx = None
        self._prev_ny = None
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._last_gaze_time = None
        # Stability: track last valid gaze for continuity
        self._last_valid_nx = 0.5
        self._last_valid_ny = 0.5
        
        # Shared multiprocessing values for overlay feedback
        self._shared_x = shared_x
        self._shared_y = shared_y
        self._shared_click_state = shared_click_state
        
        # Initialize CLAHE for adaptive lighting enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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
                refine_landmarks=True,  # Enable iris tracking landmarks
                min_detection_confidence=0.3,  # Lower threshold for better detection
                min_tracking_confidence=0.3,
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

                # Adaptive lighting enhancement using LAB color space + CLAHE
                # This works much better than static brightness/contrast across different lighting conditions
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                cl = self._clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                # Convert back to RGB for MediaPipe (MediaPipe expects RGB)
                final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
                
                face_detected = False
                if HAS_SOLUTIONS:
                    results = face_mesh.process(final_img)
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        # IMPORTANT: Check blink FIRST before updating cursor
                        self._update_blink(landmarks)
                        self._update_cursor(landmarks)
                        face_detected = True
                        self._frames_without_face = 0
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=final_img)
                    result = face_landmarker.detect(mp_image)
                    if result.face_landmarks:
                        landmarks = result.face_landmarks[0]
                        # IMPORTANT: Check blink FIRST before updating cursor
                        self._update_blink(landmarks)
                        self._update_cursor(landmarks)
                        face_detected = True
                        self._frames_without_face = 0
                
                if not face_detected:
                    self._frames_without_face += 1
                    # Reset blink state if face lost for too long
                    if self._frames_without_face > self._max_frames_without_face:
                        self._is_blinking = False
                        self._blink_start = None

                self._sleep_if_needed(loop_start, desired_dt)
        finally:
            if face_mesh is not None:
                face_mesh.close()
            if face_landmarker is not None:
                face_landmarker.close()
            cap.release()

    def _update_cursor(self, landmarks):
        now = time.time()
        
        # Check if we're in blink freeze period (during blink OR shortly after)
        in_freeze_period = False
        if self._is_blinking:
            in_freeze_period = True
            # Save current position to freeze at
            if self._frozen_x is None and self._smoothed_x is not None:
                self._frozen_x = self._smoothed_x
                self._frozen_y = self._smoothed_y
        elif self._blink_end_ts is not None:
            # Still in post-blink freeze period
            if now - self._blink_end_ts < self.blink_freeze_duration:
                in_freeze_period = True
            else:
                # Freeze period ended, clear frozen position
                self._blink_end_ts = None
                self._frozen_x = None
                self._frozen_y = None
        
        # Update shared state regarding freeze
        if self._shared_click_state and in_freeze_period:
            self._shared_click_state.value = 3

        # During freeze, don't update cursor at all
        if in_freeze_period:
            return

        nx, ny = self._gaze_from_landmarks(landmarks)
        
        # Store last valid gaze
        self._last_valid_nx = nx
        self._last_valid_ny = ny

        screen_w, screen_h = self._cursor.screen_size
        target_x = nx * screen_w
        target_y = ny * screen_h

        if self._smoothed_x is None:
            self._smoothed_x = target_x
            self._smoothed_y = target_y
        else:
            self._smoothed_x = self.smoothing * target_x + (1 - self.smoothing) * self._smoothed_x
            self._smoothed_y = self.smoothing * target_y + (1 - self.smoothing) * self._smoothed_y

        if now - self._last_move_ts >= self.cursor_update_interval:
            self._cursor.move(self._smoothed_x, self._smoothed_y)
            self._last_move_ts = now
            
            # Update shared state for overlay
            if self._shared_x: self._shared_x.value = self._smoothed_x / screen_w if screen_w > 0 else 0.5
            if self._shared_y: self._shared_y.value = self._smoothed_y / screen_h if screen_h > 0 else 0.5
            if self._shared_click_state: 
                self._shared_click_state.value = 0

    def _update_blink(self, landmarks):
        ear_left = self._eye_aspect_ratio(landmarks, LEFT_EYE)
        ear_right = self._eye_aspect_ratio(landmarks, RIGHT_EYE)
        ear = (ear_left + ear_right) * 0.5

        now = time.time()
        
        # Reset pending blink if window expired
        if self._pending_short_blink_ts and now - self._pending_short_blink_ts > self.double_blink_window:
            self._pending_short_blink_ts = None
            self._blink_count = 0

        # Detect blink state
        if ear < self.blink_threshold:
            # Eyes closing/closed
            if not self._is_blinking:
                # Just started blinking
                self._is_blinking = True
                self._blink_start = now
        else:
            # Eyes open
            if self._is_blinking:
                # Blink just ended
                self._is_blinking = False
                self._blink_end_ts = now  # Start freeze period
                
                if self._blink_start is not None:
                    duration = now - self._blink_start
                    self._blink_start = None
                    
                    # Process blink action
                    if duration >= self.short_blink_min:
                        if duration >= self.long_blink_threshold:
                            # Long blink = right click
                            if now - self._last_click_ts >= self.click_debounce:
                                self._cursor.right_click()
                                self._last_click_ts = now
                                if self._shared_click_state: self._shared_click_state.value = 2  # Right click feedback
                            self._pending_short_blink_ts = None
                            self._blink_count = 0
                        else:
                            # Short blink - check for double blink
                            if self._pending_short_blink_ts is not None:
                                time_since_first = now - self._pending_short_blink_ts
                                if time_since_first <= self.double_blink_window:
                                    # Double blink = left click!
                                    if now - self._last_click_ts >= self.click_debounce:
                                        self._cursor.left_click()
                                        self._last_click_ts = now
                                        if self._shared_click_state: self._shared_click_state.value = 1  # Left click feedback
                                    self._pending_short_blink_ts = None
                                    self._blink_count = 0
                                else:
                                    # Too slow, treat as new first blink
                                    self._pending_short_blink_ts = now
                                    self._blink_count = 1
                            else:
                                # First blink, wait for second
                                self._pending_short_blink_ts = now
                                self._blink_count = 1

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

        # Apply dead zone to filter micro-movements
        if abs(delta_iris_x) < self.dead_zone:
            delta_iris_x = 0
        if abs(delta_iris_y) < self.dead_zone:
            delta_iris_y = 0
        if abs(delta_head_x) < self.dead_zone:
            delta_head_x = 0
        if abs(delta_head_y) < self.dead_zone:
            delta_head_y = 0

        # Horizontal: balanced iris + head
        combined_x = self.iris_weight * delta_iris_x + self.head_weight * delta_head_x
        # Vertical: stronger head influence (nodding is more natural for vertical)
        combined_y = self.iris_weight * delta_iris_y + self.head_weight_vertical * delta_head_y

        # Non-linear amplification - separate for X and Y
        dist_x = abs(combined_x)
        dist_y = abs(combined_y)
        
        # Amplification based on distance from center
        amp_x = self.min_amplification + (self.max_amplification - self.min_amplification) * min(1.0, dist_x * 5) ** self.amplification_power
        amp_y = self.min_amplification + (self.max_amplification - self.min_amplification) * min(1.0, dist_y * 6) ** self.amplification_power
        
        combined_x *= amp_x
        combined_y *= amp_y

        # Calculate base position
        base_nx = 0.5 - combined_x * self.sensitivity_x
        base_ny = 0.5 + combined_y * self.sensitivity_y

        # Velocity prediction - anticipate where eye is moving
        now = time.time()
        if self._prev_nx is not None and self._last_gaze_time is not None:
            dt = now - self._last_gaze_time
            if dt > 0 and dt < 0.1:  # Only predict if reasonable time delta
                instant_vx = (base_nx - self._prev_nx) / dt
                instant_vy = (base_ny - self._prev_ny) / dt
                # Smooth velocity
                self._velocity_x = self.velocity_smoothing * instant_vx + (1 - self.velocity_smoothing) * self._velocity_x
                self._velocity_y = self.velocity_smoothing * instant_vy + (1 - self.velocity_smoothing) * self._velocity_y
        
        self._prev_nx = base_nx
        self._prev_ny = base_ny
        self._last_gaze_time = now

        # Apply velocity prediction to anticipate movement
        predicted_offset = 0.016  # Predict ~16ms ahead
        nx = base_nx + self._velocity_x * predicted_offset * self.velocity_weight
        ny = base_ny + self._velocity_y * predicted_offset * self.velocity_weight

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
