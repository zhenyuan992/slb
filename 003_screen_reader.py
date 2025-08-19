

## install 

#pip install dxcam pywin32

import dxcam
import win32gui
import time

# Find Chrome window handle
hwnd = win32gui.FindWindow("Chrome_WidgetWin_1", None)  # class name for Chrome


if hwnd == 0:
    raise RuntimeError("Chrome window not found. Make sure Chrome is open.")

# Get window rectangle (left, top, right, bottom)
rect = win32gui.GetWindowRect(hwnd)

# Create camera object
cam = dxcam.create(output_color="BGR")

frame_count = 0
while True:
    frame = cam.grab(region=rect)  # capture only Chrome window
    if frame is None:
        continue

    frame_count += 1
    if frame_count % 10 == 0:  # every 10th frame
        rgb_sum = frame.sum(axis=(0,1))
        print(f"Frame {frame_count}: RGB sum = {rgb_sum}")

    time.sleep(0.05)  # small sleep to ease CPU load
    if frame_count > 500:
        break

