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


def run(stop_event, gaze_x=None, gaze_y=None, click_state=None):
    root = tk.Tk()
    root.withdraw()

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    root.geometry(f"{screen_w}x{screen_h}+0+0")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.attributes("-transparentcolor", "black")
    # Slightly more opaque to see the feedback clearly
    root.attributes("-alpha", 0.15)

    canvas = tk.Canvas(root, width=screen_w, height=screen_h, highlightthickness=0, bd=0, bg="black")
    canvas.pack(fill="both", expand=True)

    glow_color = "#FFB347"
    thickness = 30  # Thinner border, less intrusive

    canvas.create_rectangle(0, 0, screen_w, thickness, fill=glow_color, outline="")
    canvas.create_rectangle(0, screen_h - thickness, screen_w, screen_h, fill=glow_color, outline="")
    canvas.create_rectangle(0, 0, thickness, screen_h, fill=glow_color, outline="")
    canvas.create_rectangle(screen_w - thickness, 0, screen_w, screen_h, fill=glow_color, outline="")
    
    # Gaze cursor element
    cursor_radius = 20
    gaze_cursor = canvas.create_oval(-100, -100, -50, -50, fill="#00FF00", outline="white", width=2)
    
    # Text feedback at bottom center
    status_text = canvas.create_text(screen_w//2, screen_h - 15, text="Eye Tracking Active", fill="white", font=("Arial", 12))

    root.deiconify()
    root.update_idletasks()
    hwnd = root.winfo_id()
    _make_click_through(hwnd)
    
    # Feedback color map
    # 0 = Normal, 1 = Left Click (Green), 2 = Right Click (Blue), 3 = Frozen (Red)
    state_colors = {0: "#00FF00", 1: "#00FF00", 2: "#0000FF", 3: "#FF0000"}

    def _update_ui():
        if stop_event.is_set():
            root.destroy()
            return

        if gaze_x and gaze_y:
            # Get latest values from shared memory
            gx = gaze_x.value
            gy = gaze_y.value
            
            # Update cursor position on screen
            screen_x = int(gx * screen_w)
            screen_y = int(gy * screen_h)
            
            x1 = screen_x - cursor_radius
            y1 = screen_y - cursor_radius
            x2 = screen_x + cursor_radius
            y2 = screen_y + cursor_radius
            
            canvas.coords(gaze_cursor, x1, y1, x2, y2)
            
            # Update feedback color/state
            if click_state:
                state = click_state.value
                current_color = state_colors.get(state, "#00FF00")
                
                # Visual flare for clicks
                if state == 1:
                    canvas.itemconfig(gaze_cursor, fill="white", outline="#00FF00", width=4)
                    canvas.itemconfig(status_text, text="LEFT CLICK")
                    # Reset state after display to avoid stuck UI (logic handles debounce)
                    click_state.value = 0 
                elif state == 2:
                    canvas.itemconfig(gaze_cursor, fill="white", outline="#0000FF", width=4)
                    canvas.itemconfig(status_text, text="RIGHT CLICK")
                    click_state.value = 0
                elif state == 3:
                    # Blink Frozen - Red cursor
                    canvas.itemconfig(gaze_cursor, fill="#FF0000", outline="white", width=2)
                    canvas.itemconfig(status_text, text="FROZEN (BLINK)")
                else:
                    # Normal tracking - Green semi-transparent
                    canvas.itemconfig(gaze_cursor, fill=state_colors[0], outline="white", width=2)
                    canvas.itemconfig(status_text, text="")

        root.after(16, _update_ui)  # ~60 FPS update

    root.after(100, _update_ui)
    root.mainloop()
