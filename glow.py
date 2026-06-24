"""Per-pixel-alpha layered-window rendering (UpdateLayeredWindow) for the key
light, plus the small Win32 helpers shared with the overlay.

UpdateLayeredWindow takes a 32-bpp premultiplied-BGRA bitmap and composites it
to the desktop with true per-pixel alpha. Unlike a colour-key or a uniform
window alpha, this gives a soft rounded glow with a fully transparent centre and
works regardless of the GPU/tkinter quirks that broke the earlier approaches.
"""
import ctypes
from ctypes import wintypes

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
LWA_ALPHA = 0x2
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
ULW_ALPHA = 0x02
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte),
                ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte),
                ("AlphaFormat", ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


# 64-bit-safe prototypes (handles are pointer-sized, not int)
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL
user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HANDLE, wintypes.BOOL]
user32.SetWindowRgn.restype = ctypes.c_int
user32.SetLayeredWindowAttributes.argtypes = [wintypes.HWND, wintypes.COLORREF,
                                              ctypes.c_ubyte, wintypes.DWORD]
user32.SetLayeredWindowAttributes.restype = wintypes.BOOL
user32.UpdateLayeredWindow.argtypes = [wintypes.HWND, wintypes.HDC,
                                       ctypes.POINTER(wintypes.POINT),
                                       ctypes.POINTER(wintypes.SIZE), wintypes.HDC,
                                       ctypes.POINTER(wintypes.POINT), wintypes.COLORREF,
                                       ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
user32.UpdateLayeredWindow.restype = wintypes.BOOL
gdi32.CreateEllipticRgn.argtypes = [ctypes.c_int] * 4
gdi32.CreateEllipticRgn.restype = wintypes.HANDLE
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT,
                                   ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE,
                                   wintypes.DWORD]
gdi32.CreateDIBSection.restype = wintypes.HANDLE
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
gdi32.SelectObject.restype = wintypes.HANDLE
gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL


# --------------------------------------------------------------------------- #
# Small window helpers (used by the overlay for the gaze dot)
# --------------------------------------------------------------------------- #
def make_click_through(hwnd, layered=True):
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_TRANSPARENT | WS_EX_TOPMOST
    if layered:
        style |= WS_EX_LAYERED
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)


def set_alpha(hwnd, a01):
    a = max(0, min(255, int(a01 * 255)))
    user32.SetLayeredWindowAttributes(hwnd, 0, a, LWA_ALPHA)


def set_circle_region(hwnd, win_size, radius):
    c = win_size // 2
    radius = max(0, int(radius))
    rgn = gdi32.CreateEllipticRgn(c - radius, c - radius, c + radius, c + radius)
    user32.SetWindowRgn(hwnd, rgn, True)


def move(hwnd, x, y):
    user32.SetWindowPos(hwnd, HWND_TOPMOST, int(x), int(y), 0, 0,
                        SWP_NOSIZE | SWP_NOACTIVATE)


# --------------------------------------------------------------------------- #
# Glow image builders (Pillow)
# --------------------------------------------------------------------------- #
def build_border(w, h, rgb, ds=0.5, intensity=3.0):
    """Soft, continuous rounded-rectangle glow ring with a transparent centre."""
    W, H = max(1, int(w * ds)), max(1, int(h * ds))
    margin = int(52 * ds)
    radius = int(150 * ds)
    band = max(2, int(64 * ds))
    blur = max(1, int(56 * ds))

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([margin, margin, W - margin, H - margin],
                        radius=radius, outline=rgb + (255,), width=band)
    img = img.filter(ImageFilter.GaussianBlur(blur))
    # Boost intensity: brighten the alpha so the band core reads strong while the
    # blur keeps a soft falloff on both edges.
    a = img.split()[3].point(lambda p: min(255, int(p * intensity)))
    img.putalpha(a)
    if (W, H) != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    return img


def build_full(w, h, rgb):
    """Full-screen warm panel (a softly rounded filled rect)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=40, fill=rgb + (255,))
    return img


def blank(w, h):
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def _premultiplied_bgra(img, brightness):
    w, h = img.size
    arr = np.frombuffer(img.tobytes(), np.uint8).reshape(h, w, 4).astype(np.uint16)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    if brightness < 1.0:
        a = (a * brightness).astype(np.uint16)
    out = np.empty((h, w, 4), np.uint8)
    out[:, :, 0] = (b * a // 255).astype(np.uint8)
    out[:, :, 1] = (g * a // 255).astype(np.uint8)
    out[:, :, 2] = (r * a // 255).astype(np.uint8)
    out[:, :, 3] = a.astype(np.uint8)
    return out.tobytes()


def push(hwnd, img, brightness=1.0):
    """Composite a PIL RGBA image onto a layered window via UpdateLayeredWindow."""
    w, h = img.size
    data = _premultiplied_bgra(img, brightness)

    screen_dc = user32.GetDC(None)
    mem_dc = gdi32.CreateCompatibleDC(screen_dc)

    bmi = BITMAPINFOHEADER()
    bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.biWidth = w
    bmi.biHeight = -h          # top-down
    bmi.biPlanes = 1
    bmi.biBitCount = 32
    bmi.biCompression = 0      # BI_RGB

    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(screen_dc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
    ctypes.memmove(bits, data, len(data))
    old = gdi32.SelectObject(mem_dc, hbmp)

    size = wintypes.SIZE(w, h)
    src = wintypes.POINT(0, 0)
    dst = wintypes.POINT(0, 0)
    blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
    ok = user32.UpdateLayeredWindow(hwnd, screen_dc, ctypes.byref(dst),
                                    ctypes.byref(size), mem_dc, ctypes.byref(src),
                                    0, ctypes.byref(blend), ULW_ALPHA)

    gdi32.SelectObject(mem_dc, old)
    gdi32.DeleteObject(hbmp)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(None, screen_dc)
    return bool(ok)
