import mediapipe as mp
import time
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Tuple, Sequence
import cv2, mss
import numpy as np
from gaze   import Detector, Predictor
from models import FullModel
from utils  import get_config, clamp_value, plot_trajectory
from data_collection import apply_affine
import keyboard

SETTINGS, COLOURS, EYETRACKER, TF = get_config("src/config.ini")

def _weighted_avg(buf: Sequence[float], weights: np.ndarray) -> float:
    return float(np.sum(np.array(buf) * weights) / weights.sum())

def _setup_monitor(monitors: list[dict], idx: int) -> Tuple[dict, int, int]:
    mon = monitors[idx]
    w, h = mon["width"], mon["height"]
    monitor = {"top": mon["top"], "left": mon["left"], "width": w, "height": h}
    return monitor, w, h

# --- exclude cv2 window from screen capture (Windows 10+ only) ---
def _exclude_from_capture(window_title="Eye-tracker"):
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd:
            WDA_EXCLUDEFROMCAPTURE = 0x11  # requires Win10 2004+
            user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd),
                                            wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE))
            return True
    except Exception:
        pass
    return False

def _minimize_cv2_window(title="Eye-tracker"):
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        SW_MINIMIZE = 6
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            user32.ShowWindow(wintypes.HWND(hwnd), SW_MINIMIZE)
            return True
    except Exception:
        pass
    return False

def tracker():
    detector  = Detector(output_size=SETTINGS["image_size"])
    predictor = Predictor(
        FullModel,
        model_path = Path("src/trained_models/full/fine_tuned_eyetracking_model.pt"),
        cfg_json   = Path("src/trained_models/full/fine_tuned_eyetracking_config.json"),
        gpu        = 0,
    )
    screen_err = np.load("src/trained_models/full/errors.npy")
    
    # Load affine calibration if available
    calib_affine_path = Path("src/trained_models/full/calibration_affine.npy")
    calib_affine = None
    if calib_affine_path.exists():
        calib_affine = np.load(calib_affine_path)
        print(f"[+] Loaded affine calibration from {calib_affine_path}")
    else:
        print(f"[!] No affine calibration found at {calib_affine_path}")
        print(f"[!] Run calibration in data_collection.py to create calibration file")
    
    # Calibration state
    calibration_enabled = False  # Start with calibration disabled

    # smoothing buffers
    win_pos   = SETTINGS["avg_window_length"]
    win_err   = win_pos * 2
    track_x   = deque([0]*win_pos, maxlen=win_pos)
    track_y   = deque([0]*win_pos, maxlen=win_pos)
    track_err = deque([0]*win_err, maxlen=win_err)
    w_pos     = np.arange(1, win_pos+1)
    w_err     = np.arange(1, win_err+1)

    is_recording = False
    record_pending = False
    traj_points = []
    screenshot_img = None
    is_tracking_active = True  # Start with tracking active
    should_exit = False  # Flag to exit the program

    with mss.mss() as sct:
        monitor, scr_w, scr_h = _setup_monitor(sct.monitors, EYETRACKER["monitor_num"])

        def on_toggle():
            nonlocal record_pending
            record_pending = True
        keyboard.add_hotkey('r', on_toggle)
        
        def on_stop_tracking():
            nonlocal is_tracking_active
            is_tracking_active = False
            print("[*] Eye tracking stopped (Press S to start)")
        keyboard.add_hotkey('q', on_stop_tracking)
        
        def on_start_tracking():
            nonlocal is_tracking_active
            is_tracking_active = True
            print("[*] Eye tracking started")
        keyboard.add_hotkey('s', on_start_tracking)
        
        def on_exit():
            nonlocal should_exit
            should_exit = True
            print("[*] Exiting eye tracker...")
        keyboard.add_hotkey('esc', on_exit)
        
        def on_toggle_calibration():
            nonlocal calibration_enabled
            if calib_affine is not None:
                calibration_enabled = not calibration_enabled
                status = "enabled" if calibration_enabled else "disabled"
                print(f"[*] Affine calibration {status}")
            else:
                print("[!] No calibration available. Run calibration first.")
        keyboard.add_hotkey('c', on_toggle_calibration)

        writer = None
        if EYETRACKER["write_to_disk"]:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_sz = (int(scr_w*EYETRACKER["screen_capture_scale"]),
                      int(scr_h*EYETRACKER["screen_capture_scale"]))
            writer = cv2.VideoWriter(f"src/media/recordings/{dt_str}.mp4",
                                     fourcc,
                                     EYETRACKER["tracker_frame_rate"],
                                     out_sz)

        last = time.time()

        """Delete this part before sending code"""
        cv2.namedWindow("Eye-tracker", cv2.WINDOW_NORMAL)
        _minimize_cv2_window("Eye-tracker")  # start hidden
        #cv2.setWindowProperty(
        #    "Eye-tracker",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        """---------------------------"""

        while True:
            # Check for ESC key (cv2 window focus) or should_exit flag
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or should_exit:
                break

            if record_pending:
                record_pending = False
                if not is_recording:
                    is_recording = True
                    traj_points = []
                    screenshot_img = np.array(sct.grab(monitor))
                    print("[*] Started recording trajectory")
                else:
                    is_recording = False
                    out_path = plot_trajectory(screenshot_img, traj_points)
                    print(f"[+] Trajectory saved in {out_path}")


            # Only process tracking when active
            if is_tracking_active:
                now = time.time()
                if now - last < 1/EYETRACKER["tracker_frame_rate"]:
                    continue
                fps = 1 / (now - last)
                last = now

                l_eye, r_eye, face, face_al, head_pos, head_ang = detector.get_frame()
                x_pred, y_pred = predictor.predict(face_al, l_eye, r_eye,
                                                   head_pos, head_angle=head_ang)
                
                # Apply affine calibration if enabled
                if calibration_enabled and calib_affine is not None:
                    x_pred, y_pred = apply_affine(calib_affine, x_pred, y_pred)

                track_x.append(x_pred)
                track_y.append(y_pred)
                x_cl = clamp_value(int(x_pred), scr_w-1)
                y_cl = clamp_value(int(y_pred), scr_h-1)
                track_err.append(screen_err[x_cl, y_cl]*.75)

                x_vis = _weighted_avg(track_x, w_pos)
                y_vis = _weighted_avg(track_y, w_pos)
                x_vis = clamp_value(int(x_vis), scr_w-1)
                y_vis = clamp_value(int(y_vis), scr_h-1)
                rad   = _weighted_avg(track_err, w_err)

                if is_recording:
                    traj_points.append((x_vis, y_vis))
            else:
                # When tracking is paused, use last known position or center
                # Still update frame rate for display purposes
                now = time.time()
                if now - last < 1/30.0:  # 30 FPS for display when paused
                    continue
                fps = 1 / (now - last) if last > 0 else 0
                last = now
                
                if len(track_x) > 0:
                    x_vis = int(_weighted_avg(track_x, w_pos))
                    y_vis = int(_weighted_avg(track_y, w_pos))
                    rad = int(_weighted_avg(track_err, w_err))
                else:
                    x_vis = scr_w // 2
                    y_vis = scr_h // 2
                    rad = 20

            frame   = np.array(sct.grab(monitor))
            centre  = (int(x_vis), int(y_vis))
            
            # Draw overlay and circle only when tracking is active
            if is_tracking_active:
                overlay = frame.copy()
                cv2.circle(overlay, centre, int(rad), (255,255,255,60), -1)
                cv2.circle(frame, centre, int(rad), COLOURS["green"], 4)
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            else:
                # When paused, just draw a red circle outline
                cv2.circle(frame, centre, int(rad), COLOURS["red"], 4)

            # Status text at bottom
            status_lines = []
            if is_recording:
                status_lines.append("Currently recording trajectory! Press R to stop!")
            else:
                status_lines.append("Press R to record trajectory")
            
            if is_tracking_active:
                status_lines.append("Tracking: ACTIVE (Press Q to stop)")
            else:
                status_lines.append("Tracking: PAUSED (Press S to start)")
            
            # Calibration status
            if calib_affine is not None:
                calib_status = "ON" if calibration_enabled else "OFF"
                status_lines.append(f"Calibration: {calib_status} (Press C to toggle)")
            else:
                status_lines.append("Calibration: Not available")
            
            status_lines.append("Press ESC to exit")
            
            y_offset = frame.shape[0] - 20
            for i, text in enumerate(reversed(status_lines)):
                color = COLOURS["green"] if is_tracking_active else COLOURS["yellow"]
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tx = (frame.shape[1] - tw) // 2
                cv2.putText(frame, text, (tx, y_offset - i * (th + 5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Top-left info
            y_info = 30
            if is_tracking_active:
                cv2.putText(frame, f"fps {fps:5.1f}", (10, y_info),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOURS["green"], 2)
                y_info += 30
                if len(track_x) > 0:
                    last_x = track_x[-1] if track_x else 0
                    last_y = track_y[-1] if track_y else 0
                    cv2.putText(frame, f"({last_x:4.0f},{last_y:4.0f})", (10, y_info),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS["green"], 2)
                    y_info += 30
            else:
                cv2.putText(frame, "TRACKING PAUSED", (10, y_info),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOURS["yellow"], 2)
                y_info += 30
            
            # Show calibration status in top-left
            if calib_affine is not None:
                calib_color = COLOURS["green"] if calibration_enabled else COLOURS["gray"]
                calib_text = f"Calibration: {'ON' if calibration_enabled else 'OFF'}"
                cv2.putText(frame, calib_text, (10, y_info),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 2)
            

            frame = cv2.resize(frame,
                               (int(scr_w*EYETRACKER["screen_capture_scale"]),
                                int(scr_h*EYETRACKER["screen_capture_scale"])))
            excluded_once = False
            cv2.imshow("Eye-tracker", frame)
            if not excluded_once:
                excluded_once = _exclude_from_capture("Eye-tracker")
            if writer: writer.write(frame)

        if writer: writer.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker()
