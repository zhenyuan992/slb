# pip install dxcam opencv-python numpy pywin32
import time, threading
import numpy as np
import cv2, dxcam
import win32gui

import overlay_lib
from overlay_lib import Vector2D, RgbaColor, SkDrawCircle

# --- CONFIG ---------------------------------------------------------------
TARGET_TITLE_SUBSTR = None  # e.g. "Google Chrome" or "NIS-Elements"; set to None for full screen
DRAW_EVERY_N_FRAMES = 100   # update detections every 100th frame
GREEN_LOW_HSV  = (40,  80,  80)
GREEN_HIGH_HSV = (85, 255, 255)

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
    """Return region=(l,t,r,b) or None for full screen."""
    if not TARGET_TITLE_SUBSTR:
        return None
    hwnd = _find_hwnd_by_title_substr(TARGET_TITLE_SUBSTR)
    return win32gui.GetWindowRect(hwnd) if hwnd else None

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
            circles.append(SkDrawCircle(Vector2D(int(cx), int(cy)),
                                        max(3+5, int(radius+10)),
                                        RgbaColor(100, 100, 200, 255), 2))

        with _lock:
            _circles[:] = circles

def drawlist_callback():
    with _lock:
        return list(_circles) if _circles else []
    # You can also add a FPS/debug circle here if desired.

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
