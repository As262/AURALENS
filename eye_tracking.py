import os
import time

from cursor_control import CursorController
from utils import (
    enable_dpi_awareness,
    extract_gaze_features,
    blended_gaze,
    apply_calibration,
    OneEuroFilter2D,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")


class EyeTracker:
    """Gaze -> cursor. Uses a fitted polynomial calibration when one exists,
    otherwise falls back to the original neutral-recentering heuristic."""

    def __init__(self, shared=None, calibration=None):
        self.max_fps = 60
        self.process_every = 1
        self.cursor_update_interval = 0.012
        # Blink detection - tuned thresholds
        self.blink_threshold = 0.23
        self.short_blink_min = 0.05
        self.long_blink_threshold = 0.55
        self.click_debounce = 0.3
        self.double_blink_window = 0.55
        # Freeze cursor during blink AND for a period after
        self.blink_freeze_duration = 0.15
        # Heuristic-fallback sensitivity / blend
        self.sensitivity_x = 6.5
        self.sensitivity_y = 9.0
        self.iris_weight = 0.45
        self.head_weight = 0.55
        self.head_weight_vertical = 0.65
        self.amplification_power = 1.5
        self.min_amplification = 1.0
        self.max_amplification = 2.2
        self.dead_zone = 0.008
        # Face tracking recovery
        self._frames_without_face = 0
        self._max_frames_without_face = 10

        self._cursor = CursorController()
        self._filter = OneEuroFilter2D(min_cutoff=1.0, beta=0.01, d_cutoff=1.0)
        self._last_move_ts = 0.0
        self._last_click_ts = 0.0
        self._blink_start = None
        self._blink_end_ts = None
        self._pending_short_blink_ts = None
        self._blink_count = 0
        # Deferred click: fire 0.5s after blink ends for stable input
        self._pending_click_action = None
        self._pending_click_ts = None
        self._click_delay = 0.5
        self._is_blinking = False
        self._paused = False

        # Heuristic auto-calibration state (only used when uncalibrated)
        self._neutral_nx = None
        self._neutral_ny = None
        self._neutral_head_x = None
        self._neutral_head_y = None
        self._calibration_frames = 45
        self._frame_calibration_count = 0

        # Fitted calibration model (dict) or None
        self._calibration = calibration

        # Shared multiprocessing values
        self._shared = shared
        if shared is not None:
            self._shared_x = shared.gaze_x
            self._shared_y = shared.gaze_y
            self._shared_click_state = shared.click_state
            self._tracking_enabled = shared.tracking_enabled
            self._sens_x_val = shared.sensitivity_x
            self._sens_y_val = shared.sensitivity_y
        else:
            self._shared_x = self._shared_y = self._shared_click_state = None
            self._tracking_enabled = self._sens_x_val = self._sens_y_val = None

    def run(self, stop_event):
        enable_dpi_awareness()
        import cv2
        from face_mp import FaceProcessor, make_clahe, preprocess

        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        cap.set(cv2.CAP_PROP_FPS, self.max_fps)
        if not cap.isOpened():
            return

        clahe = make_clahe()
        processor = FaceProcessor()
        desired_dt = 1.0 / self.max_fps
        frame_count = 0

        try:
            while not stop_event.is_set():
                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Paused: keep the camera warm but do no work and don't move
                # the cursor. Reset the filter once so resume doesn't jump.
                if self._tracking_enabled is not None and self._tracking_enabled.value == 0:
                    if not self._paused:
                        self._paused = True
                        self._filter.reset()
                    self._sleep_if_needed(loop_start, desired_dt)
                    continue
                self._paused = False

                frame_count += 1
                if frame_count % self.process_every != 0:
                    self._sleep_if_needed(loop_start, desired_dt)
                    continue

                rgb = preprocess(frame, clahe)
                landmarks = processor.process(rgb)
                if landmarks is not None:
                    ix, iy, hx, hy, ear = extract_gaze_features(landmarks)
                    # Check blink FIRST, then update cursor
                    self._update_blink(ear)
                    self._update_cursor(ix, iy, hx, hy)
                    self._frames_without_face = 0
                else:
                    self._frames_without_face += 1
                    if self._frames_without_face > self._max_frames_without_face:
                        self._is_blinking = False
                        self._blink_start = None

                self._sleep_if_needed(loop_start, desired_dt)
        finally:
            processor.close()
            cap.release()

    # ------------------------------------------------------------------ #
    def _update_cursor(self, ix, iy, hx, hy):
        now = time.time()

        # Freeze the cursor during a blink and briefly after (and across the
        # deferred-click window) so the pointer doesn't lurch while the eyes
        # are shut.
        in_freeze = False
        if self._is_blinking:
            in_freeze = True
        elif self._pending_click_action is not None:
            in_freeze = True
        elif self._blink_end_ts is not None:
            if now - self._blink_end_ts < self.blink_freeze_duration:
                in_freeze = True
            else:
                self._blink_end_ts = None
                self._filter.reset()  # re-seed cleanly after the blink

        if in_freeze:
            if self._shared_click_state:
                self._shared_click_state.value = 3
            return

        nx, ny = self._gaze(ix, iy, hx, hy)
        fx, fy = self._filter(nx, ny, now)

        screen_w, screen_h = self._cursor.screen_size
        if now - self._last_move_ts >= self.cursor_update_interval:
            self._cursor.move(fx * screen_w, fy * screen_h)
            self._last_move_ts = now
            if self._shared_x:
                self._shared_x.value = fx
            if self._shared_y:
                self._shared_y.value = fy
            if self._shared_click_state:
                self._shared_click_state.value = 0

    def _gaze(self, ix, iy, hx, hy):
        if self._calibration is not None:
            gx, gy = blended_gaze(ix, iy, hx, hy)
            return apply_calibration(gx, gy, self._calibration)
        return self._gaze_heuristic(ix, iy, hx, hy)

    def _gaze_heuristic(self, iris_x, iris_y, head_x, head_y):
        # First N frames: average a neutral baseline, hold at centre.
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

        if abs(delta_iris_x) < self.dead_zone:
            delta_iris_x = 0
        if abs(delta_iris_y) < self.dead_zone:
            delta_iris_y = 0
        if abs(delta_head_x) < self.dead_zone:
            delta_head_x = 0
        if abs(delta_head_y) < self.dead_zone:
            delta_head_y = 0

        combined_x = self.iris_weight * delta_iris_x + self.head_weight * delta_head_x
        combined_y = self.iris_weight * delta_iris_y + self.head_weight_vertical * delta_head_y

        dist_x = abs(combined_x)
        dist_y = abs(combined_y)
        amp_x = self.min_amplification + (self.max_amplification - self.min_amplification) * min(1.0, dist_x * 5) ** self.amplification_power
        amp_y = self.min_amplification + (self.max_amplification - self.min_amplification) * min(1.0, dist_y * 6) ** self.amplification_power
        combined_x *= amp_x
        combined_y *= amp_y

        sens_x = self._sens_x_val.value if self._sens_x_val is not None else self.sensitivity_x
        sens_y = self._sens_y_val.value if self._sens_y_val is not None else self.sensitivity_y

        nx = 0.5 - combined_x * sens_x
        ny = 0.5 + combined_y * sens_y
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return nx, ny

    # ------------------------------------------------------------------ #
    def _update_blink(self, ear):
        now = time.time()

        # Fire a deferred click once its delay has elapsed
        if self._pending_click_action is not None and now - self._pending_click_ts >= self._click_delay:
            if now - self._last_click_ts >= self.click_debounce:
                if self._pending_click_action == 'left':
                    self._cursor.left_click()
                    if self._shared_click_state:
                        self._shared_click_state.value = 1
                elif self._pending_click_action == 'right':
                    self._cursor.right_click()
                    if self._shared_click_state:
                        self._shared_click_state.value = 2
                self._last_click_ts = now
            self._pending_click_action = None
            self._pending_click_ts = None

        # Reset pending blink if the double-blink window expired
        if self._pending_short_blink_ts and now - self._pending_short_blink_ts > self.double_blink_window:
            self._pending_short_blink_ts = None
            self._blink_count = 0

        if ear < self.blink_threshold:
            if not self._is_blinking:
                self._is_blinking = True
                self._blink_start = now
        else:
            if self._is_blinking:
                self._is_blinking = False
                self._blink_end_ts = now  # start freeze period

                if self._blink_start is not None:
                    duration = now - self._blink_start
                    self._blink_start = None

                    if duration >= self.short_blink_min:
                        if duration >= self.long_blink_threshold:
                            # Long blink = right click
                            self._pending_click_action = 'right'
                            self._pending_click_ts = now
                            self._pending_short_blink_ts = None
                            self._blink_count = 0
                        else:
                            # Short blink - check for double blink
                            if self._pending_short_blink_ts is not None:
                                time_since_first = now - self._pending_short_blink_ts
                                if time_since_first <= self.double_blink_window:
                                    # Double blink = left click
                                    self._pending_click_action = 'left'
                                    self._pending_click_ts = now
                                    self._pending_short_blink_ts = None
                                    self._blink_count = 0
                                else:
                                    self._pending_short_blink_ts = now
                                    self._blink_count = 1
                            else:
                                self._pending_short_blink_ts = now
                                self._blink_count = 1

    @staticmethod
    def _sleep_if_needed(start_ts, desired_dt):
        elapsed = time.time() - start_ts
        if elapsed < desired_dt:
            time.sleep(desired_dt - elapsed)
