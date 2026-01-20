import ctypes
import tkinter as tk

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008


def _make_click_through(hwnd):
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


def run(stop_event):
    root = tk.Tk()
    root.withdraw()

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    root.geometry(f"{screen_w}x{screen_h}+0+0")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.attributes("-transparentcolor", "black")
    root.attributes("-alpha", 0.08)

    canvas = tk.Canvas(root, width=screen_w, height=screen_h, highlightthickness=0, bd=0, bg="black")
    canvas.pack(fill="both", expand=True)

    glow_color = "#FFB347"
    thickness = 80

    canvas.create_rectangle(0, 0, screen_w, thickness, fill=glow_color, outline="")
    canvas.create_rectangle(0, screen_h - thickness, screen_w, screen_h, fill=glow_color, outline="")
    canvas.create_rectangle(0, 0, thickness, screen_h, fill=glow_color, outline="")
    canvas.create_rectangle(screen_w - thickness, 0, screen_w, screen_h, fill=glow_color, outline="")

    root.deiconify()
    root.update_idletasks()
    hwnd = root.winfo_id()
    _make_click_through(hwnd)

    def _poll_stop():
        if stop_event.is_set():
            root.destroy()
        else:
            root.after(500, _poll_stop)

    root.after(500, _poll_stop)
    root.mainloop()
