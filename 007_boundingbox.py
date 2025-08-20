# pip install dxcam opencv-python numpy pywin32
import time, threading
import numpy as np
import cv2, dxcam
import win32gui

import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle, SkDrawLine

# --- CONFIG ---------------------------------------------------------------
TARGET_TITLE_SUBSTR = None  # e.g. "Google Chrome" or "NIS-Elements"; set to None for full screen
DRAW_EVERY_N_FRAMES = 100   # update detections every 100th frame
GREEN_LOW_HSV  = (40,  80,  80)
GREEN_HIGH_HSV = (85, 255, 255)
INSET_PIXELS = 200  # pixels to subtract from each edge of the screen

# --- shared state for overlay callback ------------------------------------
_circles = []
_lock = threading.Lock()

def _find_hwnd_by_title_substr(s):
    if not s: return 0
    found = []
    def cb(h, _):
        if win32gui.IsWindowVisible(h) and s.lower() in win32gui.GetWindowText(h).lower():
            found.append(h)
    win32gui.EnumWindows(cb, None)
    return found[0] if found else 0

def _get_region():
    """Return region=(l,t,r,b) inset by INSET_PIXELS from each edge or None for full screen."""
    if TARGET_TITLE_SUBSTR:
        hwnd = _find_hwnd_by_title_substr(TARGET_TITLE_SUBSTR)
        if hwnd:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            return (left + INSET_PIXELS, top + INSET_PIXELS,
                    right - INSET_PIXELS, bottom - INSET_PIXELS)
    # For full screen, get screen dimensions and apply inset
    screen_width, screen_height = win32gui.GetSystemMetrics(0), win32gui.GetSystemMetrics(1)
    return (INSET_PIXELS, INSET_PIXELS,
            screen_width - INSET_PIXELS, screen_height - INSET_PIXELS)

def _capture_worker():
    cam = dxcam.create(output_color="BGR")  # full-screen or region
    frame_i = 0
    region = _get_region()

    while True:
        frame = cam.grab(region=region) if region else cam.grab()
        if frame is None:
            time.sleep(0.01); continue

        frame_i += 1
        if frame_i % DRAW_EVERY_N_FRAMES != 0:
            continue
        print(f"frame{frame_i}")
        # --- detect green blobs --------------------------------------------
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(GREEN_LOW_HSV), np.array(GREEN_HIGH_HSV))
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        circles = []
        for c in contours:
            if cv2.contourArea(c) < 10:  # ignore tiny noise
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(c)
            # Adjust coordinates to account for region offset
            circles.append(SkDrawCircle(Vector2D(int(cx + region[0]), int(cy + region[1])),
                                        max(3+5, int(radius+10)),
                                        RgbaColor(100, 100, 200, 255), 2))

        with _lock:
            _circles[:] = circles

def drawlist_callback():
    with _lock:
        drawlist = list(_circles) if _circles else []
    # Add lines to show the tracking region
    region = _get_region()
    if region:
        left, top, right, bottom = region
        # Horizontal lines (top and bottom)
        drawlist.append(SkDrawLine(Vector2D(left, top), Vector2D(right, top),
                                  RgbaColor(255, 0, 0, 255), 2))
        drawlist.append(SkDrawLine(Vector2D(left, bottom), Vector2D(right, bottom),
                                  RgbaColor(255, 0, 0, 255), 2))
        # Vertical lines (left and right)
        drawlist.append(SkDrawLine(Vector2D(left, top), Vector2D(left, bottom),
                                  RgbaColor(255, 0, 0, 255), 2))
        drawlist.append(SkDrawLine(Vector2D(right, top), Vector2D(right, bottom),
                                  RgbaColor(255, 0, 0, 255), 2))
    return drawlist

def main():
    # spawn capture thread
    threading.Thread(target=_capture_worker, daemon=True).start()

    # start overlay (topmost, click-through handled by overlay_lib)
    overlay = overlay_lib.Overlay(
        drawlistCallback=drawlist_callback,
        refreshTimeout=3  # redraw overlay ~every 1s (drawing cost is tiny)
    )
    overlay.spawn()  # usually blocks; if not, keep the process alive.

if __name__ == "__main__":
    main()