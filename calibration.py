"""Standalone 9-point gaze calibration.

Runs as its own process (camera is single-owner, so the tray stops the live
tracker first). Shows a fullscreen grid of dots; for each dot it collects gaze
features, drops blink/no-face frames, takes the median, then fits a 2nd-order
polynomial mapping (ridge-regularised) from the gaze proxy to screen coords and
saves it to config.json. The live tracker reloads it on its next start.
"""
import time
import tkinter as tk

import config
from utils import enable_dpi_awareness, extract_gaze_features, blended_gaze

_BLINK_THRESHOLD = 0.23
_GRID = [0.1, 0.5, 0.9]          # 3x3 grid, corners pulled in for stability
_SETTLE = 0.6                     # seconds to let the eyes land on a dot
_CAPTURE_TIME = 2.0               # max seconds collecting per dot
_MIN_SAMPLES = 25                 # target good samples per dot
_RIDGE = 1e-3


class _Session:
    def __init__(self, root, canvas, cap, processor, clahe, np, sw, sh, shared, stop_event):
        self.root = root
        self.canvas = canvas
        self.cap = cap
        self.processor = processor
        self.clahe = clahe
        self.np = np
        self.sw = sw
        self.sh = sh
        self.shared = shared
        self.stop_event = stop_event

        self.targets = [(fx, fy) for fy in _GRID for fx in _GRID]
        self.index = -1
        self.collected = []   # median (gx, gy) per completed dot
        self.samples = []     # raw (gx, gy) for the current dot
        self.state = "intro"
        self.state_start = time.time()
        self.cancelled = False
        self.dot_r = 18

    # -- frame -> feature ------------------------------------------------ #
    def _read_feature(self):
        from face_mp import preprocess
        ret, frame = self.cap.read()
        if not ret:
            return None
        rgb = preprocess(frame, self.clahe)
        lm = self.processor.process(rgb)
        if lm is None:
            return None
        ix, iy, hx, hy, ear = extract_gaze_features(lm)
        if ear < _BLINK_THRESHOLD:   # drop blink frames
            return None
        return blended_gaze(ix, iy, hx, hy)

    # -- main tick ------------------------------------------------------- #
    def tick(self):
        if self.cancelled or (self.stop_event is not None and self.stop_event.is_set()):
            self.finish(save=False)
            return

        now = time.time()
        feat = self._read_feature()

        if self.state == "intro":
            self._draw_message("Eye-Tracking Calibration",
                               "Look directly at each dot until its ring fills.\n"
                               "Keep your head still and use the same lighting "
                               "you'll track under.\n\nStarting…")
            if now - self.state_start > 2.2:
                self._advance_dot()

        elif self.state == "settle":
            self._draw_dot(0.0)
            if now - self.state_start >= _SETTLE:
                self.state = "capture"
                self.state_start = now
                self.samples = []

        elif self.state == "capture":
            if feat is not None:
                self.samples.append(feat)
            progress = min(1.0, len(self.samples) / float(_MIN_SAMPLES))
            self._draw_dot(progress)
            done = len(self.samples) >= _MIN_SAMPLES or (now - self.state_start) > _CAPTURE_TIME
            if done:
                if len(self.samples) >= 5:
                    self.collected.append(self._median(self.samples))
                    self._advance_dot()
                else:
                    # Face lost / too many blinks - retry this dot
                    self.state = "settle"
                    self.state_start = now

        self.root.after(25, self.tick)

    def _advance_dot(self):
        self.index += 1
        if self.index >= len(self.targets):
            self.finish(save=True)
            return
        self.state = "settle"
        self.state_start = time.time()
        self.samples = []

    # -- drawing --------------------------------------------------------- #
    def _draw_message(self, title, body):
        self.canvas.delete("all")
        self.canvas.create_text(self.sw // 2, self.sh // 2 - 40, text=title,
                                fill="#ffffff", font=("Segoe UI", 30, "bold"))
        self.canvas.create_text(self.sw // 2, self.sh // 2 + 40, text=body,
                                fill="#bbbbbb", font=("Segoe UI", 16), justify="center")

    def _draw_dot(self, progress):
        self.canvas.delete("all")
        fx, fy = self.targets[self.index]
        cx, cy = fx * self.sw, fy * self.sh
        r = self.dot_r
        self.canvas.create_oval(cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8,
                                outline="#444444", width=2)
        fill = "#33dd55" if progress >= 1.0 else "#ffffff"
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill, outline="")
        if progress > 0:
            self.canvas.create_arc(cx - r - 8, cy - r - 8, cx + r + 8, cy + r + 8,
                                   start=90, extent=-359.999 * progress,
                                   style="arc", outline="#33dd55", width=5)
        self.canvas.create_text(self.sw // 2, self.sh - 40,
                                text=f"Dot {self.index + 1} / {len(self.targets)}    (Esc to cancel)",
                                fill="#888888", font=("Segoe UI", 14))

    # -- fitting --------------------------------------------------------- #
    def _median(self, samples):
        arr = self.np.array(samples)
        return float(self.np.median(arr[:, 0])), float(self.np.median(arr[:, 1]))

    def _ridge(self, A, y):
        np = self.np
        n = A.shape[1]
        reg = _RIDGE * np.eye(n)
        reg[0, 0] = 0.0  # don't penalise the intercept
        coef = np.linalg.solve(A.T @ A + reg, A.T @ y)
        return [float(c) for c in coef]

    def _fit_and_save(self):
        np = self.np
        pts = np.array(self.collected)
        mean = [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
        std = [float(pts[:, 0].std()), float(pts[:, 1].std())]
        std = [s if s > 1e-6 else 1.0 for s in std]

        rows = []
        for gx, gy in self.collected:
            sgx = (gx - mean[0]) / std[0]
            sgy = (gy - mean[1]) / std[1]
            rows.append([1.0, sgx, sgy, sgx * sgy, sgx * sgx, sgy * sgy])
        A = np.array(rows)
        tx = np.array([fx for (fx, fy) in self.targets])
        ty = np.array([fy for (fx, fy) in self.targets])

        model = {
            "coef_x": self._ridge(A, tx),
            "coef_y": self._ridge(A, ty),
            "mean": mean,
            "std": std,
            "neutral": mean,
            "under_light": bool(self.shared.light_on.value) if self.shared is not None else False,
        }
        cfg = config.load()
        cfg["calibration"] = model
        config.save(cfg)

    def finish(self, save):
        if save and len(self.collected) == len(self.targets):
            try:
                self._fit_and_save()
            except Exception:
                pass
        try:
            self.processor.close()
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def run_calibration(stop_event=None, shared=None):
    enable_dpi_awareness()
    import cv2
    import numpy as np
    from face_mp import FaceProcessor, make_clahe

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 60)

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    canvas = tk.Canvas(root, bg="black", highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    if not cap.isOpened():
        canvas.create_text(sw // 2, sh // 2, text="No camera available.\nClosing…",
                           fill="#ffffff", font=("Segoe UI", 24), justify="center")
        root.after(1800, root.destroy)
        root.mainloop()
        return

    processor = FaceProcessor()
    clahe = make_clahe()

    session = _Session(root, canvas, cap, processor, clahe, np, sw, sh, shared, stop_event)
    root.bind("<Escape>", lambda e: setattr(session, "cancelled", True))
    root.focus_force()
    root.after(150, session.tick)
    root.mainloop()


if __name__ == "__main__":
    run_calibration()
