"""CCD 十字线实时读数 - 点击刻线输数值标定，十字线实时读数"""

import cv2
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import scale_detector

WIN_W, WIN_H = 960, 540

marks = []            # [(px, py), value]
calib_frame_copy = None
is_vertical = True
scale_factor = 1.0
img_h, img_w = 0, 0
calibrated = False
reading = None
reading_status = ""
calib_mapping = None    # (a, b) 线性映射 value = a*px + b
records = []            # [(timestamp, reading), ...]


def to_orig(win_x, win_y):
    return (win_x / scale_factor, win_y / scale_factor)


def to_win(orig_x, orig_y):
    return (int(orig_x * scale_factor), int(orig_y * scale_factor))


def compute():
    global is_vertical, calib_mapping, calib_frame_copy

    if len(marks) < 2:
        return

    (x0, y0), _ = marks[0]
    (x1, y1), _ = marks[1]
    is_vertical = abs(x1 - x0) >= abs(y1 - y0)

    pixels = np.array([p[0] if is_vertical else p[1] for p, v in marks])
    values = np.array([v for p, v in marks])

    A = np.vstack([pixels, np.ones_like(pixels)]).T
    a, b = np.linalg.lstsq(A, values, rcond=None)[0]
    calib_mapping = (a, b)


def detect_and_read(frame, cx, cy):
    global calib_mapping, is_vertical, calib_frame_copy, reading_status

    if calib_mapping is None:
        reading_status = "no mapping"
        return None, None
    if calib_frame_copy is None:
        reading_status = "no calib frame"
        return None, None

    gray_calib = (calib_frame_copy if len(calib_frame_copy.shape) == 2
                  else cv2.cvtColor(calib_frame_copy, cv2.COLOR_BGR2GRAY))
    gray_live = (frame if len(frame.shape) == 2
                 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    axis = 0 if is_vertical else 1

    proj_calib = scale_detector.get_projection_signal(gray_calib, axis=axis)
    proj_live = scale_detector.get_projection_signal(gray_live, axis=axis)

    if proj_calib is None or proj_live is None:
        reading_status = "proj failed"
        return None, None
    if len(proj_calib) < 3 or len(proj_live) < 3:
        reading_status = f"proj too short: {len(proj_calib)}/{len(proj_live)}"
        return None, None

    max_shift = min(len(proj_calib), len(proj_live)) // 3
    min_len = min(len(proj_calib), len(proj_live))

    center = len(proj_calib) // 2
    half = min_len // 4
    c0 = max(0, center - half)
    c1 = min(len(proj_calib), center + half)
    template = proj_calib[c0:c1].astype(np.float64)
    template = template - np.mean(template)

    l0 = max(0, center - half - max_shift)
    l1 = min(len(proj_live), center + half + max_shift)
    signal = proj_live[l0:l1].astype(np.float64)
    signal = signal - np.mean(signal)

    if len(template) < 10 or len(signal) < len(template):
        reading_status = f"template too small: {len(template)}"
        return None, None

    corr = np.correlate(signal, template, mode='valid')
    if len(corr) == 0:
        reading_status = "corr empty"
        return None, None

    best_idx = int(np.argmax(corr))
    offset = (l0 + best_idx) - c0

    a, b = calib_mapping
    cross_pos = cx if is_vertical else cy
    cross_calib = cross_pos - offset
    reading_val = float(a * cross_calib + b)

    reading_status = f"offset={offset:.0f}px"
    return reading_val, offset


def list_cameras():
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i + cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append((i, w, h))
            cap.release()
        else:
            cap.release()
    return cameras


def switch_camera(cap, current_index, available_indices):
    """循环切换到下一个可用相机"""
    cap.release()
    if current_index not in available_indices:
        next_idx = available_indices[0]
    else:
        pos = available_indices.index(current_index)
        next_idx = available_indices[(pos + 1) % len(available_indices)]
    new_cap = try_open_camera(next_idx)
    if new_cap is not None:
        print(f"Switched to Camera {next_idx}")
        return new_cap, next_idx
    else:
        print(f"Camera {next_idx} unavailable")
        return cap, current_index


def try_open_camera(index, width=1280, height=720):
    configs = [
        (width, height, cv2.CAP_DSHOW),
        (width, height, cv2.CAP_ANY),
        (640, 480, cv2.CAP_DSHOW),
    ]
    for w, h, backend in configs:
        cap = cv2.VideoCapture(index + backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            return cap
        cap.release()
    return None


def draw_crosshair(img, orig_w, orig_h):
    """在原图上画十字线（原图坐标）"""
    if len(img.shape) == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display = img.copy()

    cx, cy = orig_w // 2, orig_h // 2
    cv2.line(display, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 2)
    cv2.line(display, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 2)
    cv2.circle(display, (cx, cy), 30, (0, 255, 0), 1)
    return display, cx, cy


def mouse_calib(event, x, y, flags, param):
    global marks, calibrated

    if calibrated:
        return

    if x < 0 or y < 0:
        return

    ox, oy = to_orig(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        val = input_dialog(ox, oy)
        if val is not None and val.strip():
            try:
                num = float(val)
                marks.append(((ox, oy), num))
                compute()
                print(f"[{len(marks)}] ({ox:.0f},{oy:.0f}) = {num}")
            except ValueError:
                print(f"Invalid: {val}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if marks:
            removed = marks.pop()
            print(f"undo: {removed[1]}")
            if len(marks) >= 2:
                compute()


def input_dialog(ox, oy):
    print(f"\nClicked ({ox:.0f}, {oy:.0f})")
    print("Enter value (Enter=confirm, empty=cancel): ", end="", flush=True)
    return sys.stdin.readline().strip()


def main():
    global marks, calibrated, reading, scale_factor, img_h, img_w
    global is_vertical, calib_mapping, calib_frame_copy

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--list", action="store_true")
    args, _ = parser.parse_known_args()

    cameras = list_cameras()
    if not cameras:
        print("No cameras found")
        return

    print("Cameras:")
    for idx, w, h in cameras:
        print(f"  Camera {idx}: {w}x{h}")

    if args.list:
        return

    camera_index = args.camera if args.camera is not None else cameras[-1][0]
    cam_indices = [c[0] for c in cameras]
    cap = try_open_camera(camera_index)
    if cap is None:
        print("Cannot open camera")
        return

    print(f"\nCamera {camera_index} opened")
    print("=" * 50)
    print("Live preview. Crosshair in center.")
    print("Press ENTER to freeze frame and calibrate.")
    print("=" * 50)
    print()

    cv2.namedWindow("CCD Crosshair", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCD Crosshair", WIN_W, WIN_H)

    mode = "preview"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            img_h, img_w = frame.shape[:2]
            scale_w = WIN_W / img_w
            scale_h = WIN_H / img_h
            scale_factor = min(scale_w, scale_h, 1.0)

            if mode == "preview":
                display, cx, cy = draw_crosshair(frame, img_w, img_h)
                cv2.putText(display, "ENTER:calibrate | TAB:switch cam | q:quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                canvas_w = max(1, int(img_w * scale_factor))
                canvas_h = max(1, int(img_h * scale_factor))
                if abs(scale_factor - 1.0) > 0.001:
                    display = cv2.resize(display, (canvas_w, canvas_h))

                cv2.imshow("CCD Crosshair", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == 9:  # Tab: switch camera
                    cap, camera_index = switch_camera(cap, camera_index, cam_indices)
                elif key == 13:  # Enter: freeze and calibrate
                    calib_frame = frame.copy()
                    calib_frame_copy = frame.copy()
                    cv2.setMouseCallback("CCD Crosshair", mouse_calib)
                    mode = "calib"
                    print("Frame frozen. Mark scale lines (LEFT click + type value).")
                continue

            elif mode == "calib":
                if calib_frame is None:
                    mode = "preview"
                    continue

                img_h, img_w = calib_frame.shape[:2]
                scale_w = WIN_W / img_w
                scale_h = WIN_H / img_h
                scale_factor = min(scale_w, scale_h, 1.0)

                display, cx, cy = draw_crosshair(calib_frame, img_w, img_h)

                colors = [(0, 255, 255), (0, 200, 255), (0, 255, 150), (255, 0, 255)]
                for i, ((px, py), val) in enumerate(marks):
                    color = colors[i % len(colors)]
                    xi, yi = int(px), int(py)
                    if is_vertical:
                        cv2.line(display, (xi, 0), (xi, img_h), color, 2)
                    else:
                        cv2.line(display, (0, yi), (img_w, yi), color, 2)
                    cv2.putText(display, str(val), (xi + 5, yi - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.circle(display, (xi, yi), 5, color, -1)

                if len(marks) < 2:
                    msg = f"Mark {2 - len(marks)} more line(s)"
                else:
                    msg = "ENTER to confirm | ESC to cancel"
                cv2.putText(display, msg, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.putText(display, "LEFT: mark + value | RIGHT: undo",
                            (10, img_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

                canvas_w = max(1, int(img_w * scale_factor))
                canvas_h = max(1, int(img_h * scale_factor))
                if abs(scale_factor - 1.0) > 0.001:
                    display = cv2.resize(display, (canvas_w, canvas_h))

                cv2.imshow("CCD Crosshair", display)

                key = cv2.waitKey(30) & 0xFF
                if key == 13 and len(marks) >= 2:  # Enter: confirm
                    calibrated = True
                    mode = "live"
                    cv2.setMouseCallback("CCD Crosshair", lambda *args: None)
                    print(f"Calibrated! {len(marks)} lines. Live reading mode.")
                elif key == 27:  # ESC: cancel
                    marks = []
                    reading = None
                    calibrated = False
                    calib_mapping = None
                    calib_frame_copy = None
                    mode = "preview"
                    cv2.setMouseCallback("CCD Crosshair", lambda *args: None)
                    print("Calibration cancelled. Back to preview.")
                continue

            elif mode == "live":
                display, cx, cy = draw_crosshair(frame, img_w, img_h)

                if len(marks) >= 2:
                    reading, offset = detect_and_read(frame, cx, cy)
                    if reading is not None:
                        abs_val = abs(reading)
                        if abs_val < 0.2:
                            color = (0, 255, 0)
                        elif abs_val < 1.0:
                            color = (0, 255, 255)
                        else:
                            color = (0, 0, 255)
                        cv2.putText(display, f"{reading:.3f}",
                                    (cx + 35, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        cv2.putText(display, f"No reading ({reading_status})",
                                    (cx + 35, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
                else:
                    cv2.putText(display, "Not calibrated",
                                (cx + 35, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)

                if reading is not None and offset is not None:
                    info = f"Offset={offset:.0f}px  Xhair=({cx},{cy})"
                    cv2.putText(display, info, (10, img_h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

                cv2.putText(display, "ENTER:recalibrate | SPACE:record | TAB:switch cam | q:quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(display, f"Records: {len(records)}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

                canvas_w = max(1, int(img_w * scale_factor))
                canvas_h = max(1, int(img_h * scale_factor))
                if abs(scale_factor - 1.0) > 0.001:
                    display = cv2.resize(display, (canvas_w, canvas_h))

                cv2.imshow("CCD Crosshair", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord(' '):  # Space: record
                    if reading is not None:
                        t = time.time()
                        records.append((t, reading))
                        print(f"[REC {len(records)}] {reading:.3f}")
                    else:
                        print("No reading to record")
                elif key == 9:  # Tab: switch camera
                    cap, camera_index = switch_camera(cap, camera_index, cam_indices)
                elif key == 13:  # Enter: recalibrate
                    for _ in range(5):
                        cap.read()
                    attempts = 0
                    calib_frame = None
                    while calib_frame is None and attempts < 50:
                        ret, f = cap.read()
                        if ret and f is not None and np.mean(f) > 10:
                            calib_frame = f.copy()
                            calib_frame_copy = f.copy()
                        attempts += 1
                    if calib_frame is not None:
                        mode = "calib"
                        calibrated = False
                        marks = []
                        reading = None
                        calib_mapping = None
                        cv2.setMouseCallback("CCD Crosshair", mouse_calib)
                        print("Frame frozen. Mark scale lines.")
                    else:
                        print("Failed to capture frame")
                continue

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # 输出记录的数据表格
        if records:
            print("\n" + "=" * 50)
            print("   Recorded Data")
            print("=" * 50)
            print(f"  {'No.':<6} {'Timestamp':<20} {'Reading (mm)':<15}")
            print("  " + "-" * 41)
            t0 = records[0][0] if records else 0
            for i, (ts, val) in enumerate(records, 1):
                dt = ts - t0
                print(f"  {i:<6} {dt:>8.2f} s{'':>8} {val:>10.3f}")
            print("=" * 50)
        else:
            print("No data recorded.")


if __name__ == "__main__":
    main()
