"""Settings window (runs as its own short-lived process).

Sliders write the shared mp.Values live (instant preview in the overlay/tracker)
and persist to config.json. The Recalibrate button just raises a request flag
that the tray watches - the tray owns the camera-handoff sequence.
"""
import tkinter as tk

import config
from utils import enable_dpi_awareness, kelvin_to_rgb, rgb_to_hex

BG = "#1e1e1e"
FG = "#e0e0e0"
SUB = "#9a9a9a"
TROUGH = "#3a3a3a"
ACCENT = "#2d7d46"


def run(shared):
    enable_dpi_awareness()
    cfg = config.load()

    root = tk.Tk()
    root.title("Eye Tracker — Settings")
    root.configure(bg=BG)
    root.resizable(False, False)
    root.attributes("-topmost", True)

    def persist():
        config.pull_from_shared(shared, cfg)
        # Re-read the calibration block from disk so a slider save can't clobber
        # a model that was just fitted (e.g. via this window's Recalibrate).
        cfg["calibration"] = config.load().get("calibration")
        config.save(cfg)

    def header(text):
        tk.Label(root, text=text, bg=BG, fg=FG,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=16, pady=(14, 2))

    def make_scale(label, frm, to, res, getter, setter, fmt="{:.0f}"):
        row = tk.Frame(root, bg=BG)
        row.pack(fill="x", padx=16, pady=2)
        top = tk.Frame(row, bg=BG)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=BG, fg=FG, font=("Segoe UI", 10)).pack(side="left")
        val_lbl = tk.Label(top, text=fmt.format(getter()), bg=BG, fg=SUB,
                           font=("Segoe UI", 10))
        val_lbl.pack(side="right")

        def on_move(v):
            setter(float(v))
            val_lbl.config(text=fmt.format(float(v)))

        scale = tk.Scale(row, from_=frm, to=to, resolution=res, orient="horizontal",
                         showvalue=False, bg=BG, fg=FG, troughcolor=TROUGH,
                         highlightthickness=0, bd=0, sliderrelief="flat",
                         activebackground=ACCENT, command=on_move, length=300)
        scale.set(getter())
        scale.pack(fill="x")
        scale.bind("<ButtonRelease-1>", lambda e: persist())
        return val_lbl

    # ---- Key light --------------------------------------------------------
    header("Key Light")

    light_var = tk.IntVar(value=int(shared.light_on.value))

    def toggle_light():
        shared.light_on.value = int(light_var.get())
        persist()

    tk.Checkbutton(root, text="Enable key light", variable=light_var,
                   command=toggle_light, bg=BG, fg=FG, selectcolor=BG,
                   activebackground=BG, activeforeground=FG,
                   font=("Segoe UI", 10)).pack(anchor="w", padx=14)

    mode_var = tk.IntVar(value=int(shared.light_mode.value))

    def set_mode():
        shared.light_mode.value = int(mode_var.get())
        persist()

    mode_row = tk.Frame(root, bg=BG)
    mode_row.pack(anchor="w", padx=14)
    tk.Radiobutton(mode_row, text="Glow border", variable=mode_var, value=0,
                   command=set_mode, bg=BG, fg=FG, selectcolor=BG,
                   activebackground=BG, activeforeground=FG,
                   font=("Segoe UI", 10)).pack(side="left")
    tk.Radiobutton(mode_row, text="Full panel", variable=mode_var, value=1,
                   command=set_mode, bg=BG, fg=FG, selectcolor=BG,
                   activebackground=BG, activeforeground=FG,
                   font=("Segoe UI", 10)).pack(side="left", padx=12)

    make_scale("Brightness", 0, 100, 1,
               lambda: shared.brightness.value * 100.0,
               lambda v: setattr(shared.brightness, "value", v / 100.0),
               fmt="{:.0f}%")

    # Temperature with a live colour swatch
    temp_row = tk.Frame(root, bg=BG)
    temp_row.pack(fill="x", padx=16, pady=2)
    temp_top = tk.Frame(temp_row, bg=BG)
    temp_top.pack(fill="x")
    tk.Label(temp_top, text="Temperature", bg=BG, fg=FG,
             font=("Segoe UI", 10)).pack(side="left")
    swatch = tk.Label(temp_top, text="  ", bg=rgb_to_hex(kelvin_to_rgb(shared.temp_k.value)))
    swatch.pack(side="right", padx=(8, 0))
    temp_lbl = tk.Label(temp_top, text=f"{int(shared.temp_k.value)}K", bg=BG, fg=SUB,
                        font=("Segoe UI", 10))
    temp_lbl.pack(side="right")

    def on_temp(v):
        shared.temp_k.value = float(v)
        temp_lbl.config(text=f"{int(float(v))}K")
        swatch.config(bg=rgb_to_hex(kelvin_to_rgb(float(v))))

    temp_scale = tk.Scale(temp_row, from_=2000, to=6500, resolution=100,
                          orient="horizontal", showvalue=False, bg=BG, fg=FG,
                          troughcolor=TROUGH, highlightthickness=0, bd=0,
                          sliderrelief="flat", activebackground=ACCENT,
                          command=on_temp, length=300)
    temp_scale.set(shared.temp_k.value)
    temp_scale.pack(fill="x")
    temp_scale.bind("<ButtonRelease-1>", lambda e: persist())

    make_scale("Overlay opacity", 5, 100, 1,
               lambda: shared.overlay_opacity.value * 100.0,
               lambda v: setattr(shared.overlay_opacity, "value", v / 100.0),
               fmt="{:.0f}%")

    # ---- Tracking ---------------------------------------------------------
    header("Tracking")
    tk.Label(root, text="Sensitivity applies only in uncalibrated mode.",
             bg=BG, fg=SUB, font=("Segoe UI", 8)).pack(anchor="w", padx=16)

    make_scale("Sensitivity X", 1.0, 15.0, 0.5,
               lambda: shared.sensitivity_x.value,
               lambda v: setattr(shared.sensitivity_x, "value", v),
               fmt="{:.1f}")
    make_scale("Sensitivity Y", 1.0, 20.0, 0.5,
               lambda: shared.sensitivity_y.value,
               lambda v: setattr(shared.sensitivity_y, "value", v),
               fmt="{:.1f}")

    # ---- Clicking ---------------------------------------------------------
    header("Clicking")
    tk.Label(root, text="Blink clicks always work: double-blink = left, "
                        "long blink = right.", bg=BG, fg=SUB,
             font=("Segoe UI", 8)).pack(anchor="w", padx=16)

    dwell_var = tk.IntVar(value=int(shared.dwell_enabled.value))

    def toggle_dwell():
        shared.dwell_enabled.value = int(dwell_var.get())
        persist()

    tk.Checkbutton(root, text="Dwell click (hold gaze still to click)",
                   variable=dwell_var, command=toggle_dwell, bg=BG, fg=FG,
                   selectcolor=BG, activebackground=BG, activeforeground=FG,
                   font=("Segoe UI", 10)).pack(anchor="w", padx=14)

    make_scale("Dwell time", 0.5, 2.5, 0.1,
               lambda: shared.dwell_time.value,
               lambda v: setattr(shared.dwell_time, "value", v),
               fmt="{:.1f}s")

    # ---- Actions ----------------------------------------------------------
    status = tk.Label(root, text="", bg=BG, fg=ACCENT, font=("Segoe UI", 9))
    status.pack(pady=(8, 0))

    def recalibrate():
        shared.recalibrate_request.value = 1
        status.config(text="Calibration starting… follow the dots on screen.")

    btns = tk.Frame(root, bg=BG)
    btns.pack(fill="x", padx=16, pady=14)
    tk.Button(btns, text="Recalibrate", command=recalibrate, bg=ACCENT, fg="white",
              relief="flat", font=("Segoe UI", 10), padx=12, pady=4).pack(side="left")
    tk.Button(btns, text="Close", command=root.destroy, bg=TROUGH, fg=FG,
              relief="flat", font=("Segoe UI", 10), padx=12, pady=4).pack(side="right")

    root.bind("<Escape>", lambda e: root.destroy())
    root.mainloop()


if __name__ == "__main__":
    from config import SharedState
    run(SharedState())
