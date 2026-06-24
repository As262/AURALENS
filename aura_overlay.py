"""On-screen overlay: an Apple-style soft glow key light + a gaze-cursor dot.

* KEY LIGHT - a single full-screen layered window painted with a per-pixel-alpha
  bitmap (UpdateLayeredWindow, see glow.py): a soft, continuous, rounded glow
  ring with a transparent centre (border mode) or a warm full panel. Rebuilt
  only when on/mode/temperature/brightness change.
* GAZE DOT - a small window clipped to a CIRCLE (SetWindowRgn) that follows the
  cursor; grows / recolours for dwell progress and click / freeze feedback.

Both windows are click-through and topmost.
"""
import tkinter as tk

import glow
from utils import enable_dpi_awareness, kelvin_to_rgb

DOT_WIN = 72          # dot window size (room for the grown dwell radius)
DOT_BASE_R = 9
DOT_MAX_R = 30
FLASH_FRAMES = 12     # ~190ms click flash at 60fps

GREEN = (51, 221, 85)
CYAN = (0, 229, 255)


def _hex(rgb):
    return "#%02x%02x%02x" % rgb


def _lerp(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def run(stop_event, shared):
    enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    # ---- key light: one per-pixel-alpha layered window ----
    light = tk.Toplevel(root)
    light.overrideredirect(True)
    light.geometry(f"{sw}x{sh}+0+0")
    light.configure(bg="black")
    light.update_idletasks()
    glow.make_click_through(light.winfo_id(), layered=True)
    light_hwnd = light.winfo_id()
    blank_img = glow.blank(sw, sh)
    glow.push(light_hwnd, blank_img)            # start fully transparent

    # ---- gaze dot: shaped circular window ----
    dot = tk.Toplevel(root)
    dot.overrideredirect(True)
    dot.geometry(f"{DOT_WIN}x{DOT_WIN}+0+0")
    dot.attributes("-topmost", True)
    dot_canvas = tk.Canvas(dot, width=DOT_WIN, height=DOT_WIN,
                           highlightthickness=0, bd=0, bg=_hex(GREEN))
    dot_canvas.pack(fill="both", expand=True)
    dot.update_idletasks()
    glow.make_click_through(dot.winfo_id(), layered=True)
    dot_hwnd = dot.winfo_id()
    glow.set_alpha(dot_hwnd, 1.0)               # opaque; shaped by its region
    glow.set_circle_region(dot_hwnd, DOT_WIN, DOT_BASE_R)

    light_state = {"on": None, "mode": None, "temp": None, "br": None,
                   "base": None, "base_key": None}
    dot_state = {"radius": DOT_BASE_R, "flash": 0, "flash_color": "#ffffff"}

    def _refresh_light():
        on = bool(shared.light_on.value)
        mode = int(shared.light_mode.value)
        temp = int(shared.temp_k.value)
        br = round(max(0.0, min(1.0, shared.brightness.value)), 2)
        if (on, mode, temp, br) == (light_state["on"], light_state["mode"],
                                    light_state["temp"], light_state["br"]):
            return
        if not on:
            glow.push(light_hwnd, blank_img)
        else:
            if (mode, temp) != light_state["base_key"]:
                rgb = kelvin_to_rgb(temp)
                light_state["base"] = (glow.build_full(sw, sh, rgb) if mode == 1
                                       else glow.build_border(sw, sh, rgb))
                light_state["base_key"] = (mode, temp)
            glow.push(light_hwnd, light_state["base"], br)
        light_state.update(on=on, mode=mode, temp=temp, br=br)

    def _update():
        if stop_event.is_set():
            root.destroy()
            return

        _refresh_light()

        # ---- gaze dot ----
        if shared.calibrating.value or not shared.tracking_enabled.value:
            glow.move(dot_hwnd, -DOT_WIN * 3, -DOT_WIN * 3)   # park off-screen
            root.after(16, _update)
            return

        click = shared.click_state.value
        if click == 1:
            dot_state["flash"] = FLASH_FRAMES
            dot_state["flash_color"] = "#ffffff"
            shared.click_state.value = 0
        elif click == 2:
            dot_state["flash"] = FLASH_FRAMES
            dot_state["flash_color"] = "#3b82f6"
            shared.click_state.value = 0

        if dot_state["flash"] > 0:
            dot_canvas.configure(bg=dot_state["flash_color"])
            radius = DOT_MAX_R
            dot_state["flash"] -= 1
        elif click == 3:                              # blink-freeze
            dot_canvas.configure(bg="#ff3b30")
            radius = DOT_BASE_R
        else:
            progress = shared.dwell_progress.value
            radius = DOT_BASE_R + (DOT_MAX_R - DOT_BASE_R) * progress
            dot_canvas.configure(bg=_hex(_lerp(GREEN, CYAN, progress)))

        new_r = int(radius)
        if new_r != dot_state["radius"]:
            glow.set_circle_region(dot_hwnd, DOT_WIN, new_r)
            dot_state["radius"] = new_r

        sx = int(shared.gaze_x.value * sw)
        sy = int(shared.gaze_y.value * sh)
        glow.move(dot_hwnd, sx - DOT_WIN // 2, sy - DOT_WIN // 2)

        root.after(16, _update)

    root.after(100, _update)
    root.mainloop()
