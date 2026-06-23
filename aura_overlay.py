"""On-screen overlay: an Apple-style key light + the gaze-cursor feedback.

Two separate click-through, topmost, layered windows under one hidden Tk root:

* LIGHT window  - window alpha = brightness; draws a warm (Kelvin) glow border
  with a transparent, click-through centre, or a full-screen warm panel.
* FEEDBACK window - fixed alpha; draws the gaze cursor + status, kept above the
  light so a bright full panel never washes it out.

tkinter only offers ONE window-level alpha and ONE transparent colour, hence the
split. The transparent colour is pure black: the only Windows colour key that
reliably composites to the desktop on this hardware. Nothing we draw is pure
black - the warm Kelvin ramp and the glow bands (which floor at 45% brightness)
stay well clear of (0,0,0) - so nothing we want visible gets keyed out.
"""
import ctypes
import tkinter as tk

from utils import enable_dpi_awareness, kelvin_to_rgb, rgb_to_hex, scale_rgb

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008

SENTINEL = "#000000"          # transparent / click-through colour key (black)
BORDER_THICKNESS = 70
BORDER_BANDS = 6


def _make_click_through(hwnd):
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


def _make_layer_window(root, sw, sh, alpha):
    win = tk.Toplevel(root)
    win.overrideredirect(True)
    win.geometry(f"{sw}x{sh}+0+0")
    win.attributes("-topmost", True)
    win.configure(bg=SENTINEL)
    win.attributes("-transparentcolor", SENTINEL)
    win.attributes("-alpha", alpha)
    canvas = tk.Canvas(win, width=sw, height=sh, highlightthickness=0, bd=0, bg=SENTINEL)
    canvas.pack(fill="both", expand=True)
    return win, canvas


def _render_light(canvas, sw, sh, mode, temp_k):
    canvas.delete("all")
    rgb = kelvin_to_rgb(temp_k)
    if mode == 1:  # full panel
        canvas.create_rectangle(0, 0, sw, sh, fill=rgb_to_hex(rgb), outline="")
        return
    # Glow border: transparent base + graduated bands (bright edge -> dim inner)
    canvas.create_rectangle(0, 0, sw, sh, fill=SENTINEL, outline="")
    band_t = BORDER_THICKNESS / BORDER_BANDS
    for i in range(BORDER_BANDS):
        frac = 1.0 - 0.55 * (BORDER_BANDS - 1 - i) / (BORDER_BANDS - 1)
        col = rgb_to_hex(scale_rgb(rgb, frac))
        inset = (BORDER_BANDS - 1 - i) * band_t
        canvas.create_rectangle(0, inset, sw, inset + band_t, fill=col, outline="")
        canvas.create_rectangle(0, sh - inset - band_t, sw, sh - inset, fill=col, outline="")
        canvas.create_rectangle(inset, 0, inset + band_t, sh, fill=col, outline="")
        canvas.create_rectangle(sw - inset - band_t, 0, sw - inset, sh, fill=col, outline="")


def run(stop_event, shared):
    enable_dpi_awareness()
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # Light window first so the feedback window naturally stacks above it.
    light_win, light_canvas = _make_layer_window(root, screen_w, screen_h,
                                                 max(0.0, min(1.0, shared.brightness.value)))
    fb_win, fb_canvas = _make_layer_window(root, screen_w, screen_h,
                                           max(0.0, min(1.0, shared.overlay_opacity.value)))

    cursor_radius = 20
    dwell_ring = fb_canvas.create_arc(-100, -100, -50, -50, start=90, extent=0,
                                      style="arc", outline="#00E5FF", width=4,
                                      state="hidden")
    gaze_cursor = fb_canvas.create_oval(-100, -100, -50, -50, fill="#00FF00",
                                        outline="white", width=2)
    status_text = fb_canvas.create_text(screen_w // 2, screen_h - 15, text="",
                                        fill="white", font=("Arial", 12))

    for win in (light_win, fb_win):
        win.update_idletasks()
        _make_click_through(win.winfo_id())

    state = {"mode": None, "temp": None, "on": None}

    def _update():
        if stop_event.is_set():
            root.destroy()
            return

        # ---- key light ----
        light_on = bool(shared.light_on.value)
        mode = int(shared.light_mode.value)
        temp = int(shared.temp_k.value)
        if light_on:
            light_win.attributes("-alpha", max(0.0, min(1.0, shared.brightness.value)))
            if (mode, temp) != (state["mode"], state["temp"]) or not state["on"]:
                _render_light(light_canvas, screen_w, screen_h, mode, temp)
        else:
            light_win.attributes("-alpha", 0.0)
        state["mode"], state["temp"], state["on"] = mode, temp, light_on

        # While calibrating, hide the cursor and DON'T lift over the (separate
        # process) fullscreen calibration window.
        if shared.calibrating.value:
            fb_canvas.coords(gaze_cursor, -100, -100, -50, -50)
            fb_canvas.itemconfig(dwell_ring, state="hidden")
            fb_canvas.itemconfig(status_text, text="")
            root.after(16, _update)
            return

        # ---- gaze feedback ----
        fb_win.attributes("-alpha", max(0.05, min(1.0, shared.overlay_opacity.value)))
        gx = shared.gaze_x.value
        gy = shared.gaze_y.value
        sx = int(gx * screen_w)
        sy = int(gy * screen_h)
        fb_canvas.coords(gaze_cursor, sx - cursor_radius, sy - cursor_radius,
                         sx + cursor_radius, sy + cursor_radius)

        # Dwell progress ring around the cursor
        dp = shared.dwell_progress.value
        if dp > 0.02:
            rr = cursor_radius + 8
            fb_canvas.coords(dwell_ring, sx - rr, sy - rr, sx + rr, sy + rr)
            fb_canvas.itemconfig(dwell_ring, extent=-359.999 * dp, state="normal")
        else:
            fb_canvas.itemconfig(dwell_ring, state="hidden")

        click = shared.click_state.value
        if click == 1:
            fb_canvas.itemconfig(gaze_cursor, fill="white", outline="#00FF00", width=4)
            fb_canvas.itemconfig(status_text, text="LEFT CLICK")
            shared.click_state.value = 0
        elif click == 2:
            fb_canvas.itemconfig(gaze_cursor, fill="white", outline="#0000FF", width=4)
            fb_canvas.itemconfig(status_text, text="RIGHT CLICK")
            shared.click_state.value = 0
        elif click == 3:
            fb_canvas.itemconfig(gaze_cursor, fill="#FF0000", outline="white", width=2)
            fb_canvas.itemconfig(status_text, text="FROZEN (BLINK)")
        else:
            fb_canvas.itemconfig(gaze_cursor, fill="#00FF00", outline="white", width=2)
            fb_canvas.itemconfig(status_text, text="")

        # Keep the cursor window above the (possibly bright) light window.
        fb_win.lift()
        root.after(16, _update)

    root.after(100, _update)
    root.mainloop()
