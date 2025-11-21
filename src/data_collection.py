import sys
from collections import deque
from pathlib import Path
import shutil
import itertools
import random
import pandas as pd
from PIL import Image
import cv2

import numpy as np
import pygame
import torch


from utils import get_config, get_calibration_zones, clamp_value, get_undersampled_region  # utils.py
from gaze import Detector, Predictor                               # gaze.py
from models import FullModel                                       # models.py
from target import Target                                          # target.py

# Read config.ini (same pattern as data_collection.py)
SETTINGS, COLOURS, EYETRACKER, TF = get_config("src/config.ini")

MODEL_PATH = Path("src/trained_models/full/eyetracking_model.pt")
CFG_JSON   = Path("src/trained_models/full/eyetracking_config.json")
CALIB_PATH_POLY2 = Path("src/trained_models/full/calibration_poly2.npy")
CALIB_PATH_AFFINE = Path("src/trained_models/full/calibration_affine.npy")
FINE_TUNED_MODEL_PATH = Path("src/trained_models/full/fine_tuned_eyetracking_model.pt")
FINE_TUNED_CFG_JSON = Path("src/trained_models/full/fine_tuned_eyetracking_config.json")
DATA_DIR = Path("src/data")  # Match project structure from README

# Calibration settings
CALIB_WEIGHT = SETTINGS.get("calibration_weight", 0)  # Weight of calibration (0.0-1.0)
CALIB_ENABLED = SETTINGS.get("calibration_enabled", True)  # Enable/disable calibration
CALIB_TYPE = SETTINGS.get("calibration_type", "affine")  # "affine" or "poly2"

# Menu icons from data_collection.py
MENU_CALIBRATION_IMG = "src/menu/calibration_mode.png"
MENU_COLLECTION_IMG  = "src/menu/collection_mode.png"  # unused, but we still load
MENU_TRACKING_IMG    = "src/menu/tracking_mode.png"


# --------- calibration maths: affine and quadratic (poly2) mapping from preds -> targets ---------

def fit_affine(preds, targets):
    """
    Affine calibration (linear transformation):
      x = a0 + a1*xh + a2*yh
      y = b0 + b1*xh + b2*yh
    
    This is a 2D affine transformation (translation, rotation, scaling, shearing).
    Requires at least 3 points (ideally 9 for stability).
    
    preds   : (N, 2) raw model predictions (xh, yh)
    targets : (N, 2) true screen coords (x, y)
    returns : M (3x2) so that [1, xh, yh] @ M ≈ (x, y)
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    
    if len(preds) < 3:
        raise ValueError("Affine calibration requires at least 3 points")
    
    xh = preds[:, 0]
    yh = preds[:, 1]
    
    # Design matrix: [1, xh, yh] for each sample
    Phi = np.stack(
        [
            np.ones_like(xh),
            xh,
            yh,
        ],
        axis=1,
    )  # (N, 3)
    
    # Solve Phi @ M ≈ targets, M shape (3, 2)
    M, *_ = np.linalg.lstsq(Phi, targets, rcond=None)
    return M  # (3, 2)


def apply_affine(M, xh, yh):
    """
    Apply affine mapping M (3x2) to a single raw prediction (xh, yh).
    Returns (x_cal, y_cal).
    """
    phi = np.array([1.0, xh, yh], dtype=np.float64)  # (3,)
    x_cal, y_cal = phi @ M  # (2,)
    return float(x_cal), float(y_cal)


def fit_poly2(preds, targets):
    """
    Quadratic calibration:
      x = a0 + a1*xh + a2*yh + a3*xh^2 + a4*xh*yh + a5*yh^2
      y = b0 + b1*xh + b2*yh + b3*xh^2 + b4*xh*yh + b5*yh^2

    preds   : (N, 2) raw model predictions (xh, yh)
    targets : (N, 2) true screen coords (x, y)
    returns : W (6x2) so that [1, xh, yh, xh^2, xh*yh, yh^2] @ W ≈ (x, y)
    """
    preds   = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    xh = preds[:, 0]
    yh = preds[:, 1]

    # design matrix Φ: (N, 6)
    Phi = np.stack(
        [
            np.ones_like(xh),
            xh,
            yh,
            xh**2,
            xh * yh,
            yh**2,
        ],
        axis=1,
    )  # (N, 6)

    # Solve Φ W ≈ targets, W shape (6, 2)
    W, *_ = np.linalg.lstsq(Phi, targets, rcond=None)
    return W  # (6, 2)


def apply_poly2(W, xh, yh):
    """
    Apply quadratic mapping W (6x2) to a single raw prediction (xh, yh).
    Returns (x_cal, y_cal).
    """
    phi = np.array(
        [1.0, xh, yh, xh * xh, xh * yh, yh * yh],
        dtype=np.float64,
    )  # (6,)
    x_cal, y_cal = phi @ W  # (2,)
    return float(x_cal), float(y_cal)


def get_calibration_zones_5x5(w, h, target_radius):
    """
    Generate a 5x5 grid of calibration zones (25 points).
    
    Args:
        w, h: Screen width and height
        target_radius: Radius of target (for margin)
    
    Returns:
        List of (x, y) tuples covering the screen in a 5x5 grid
    """
    margin = target_radius * 2
    # Create 5 positions per axis
    x_positions = np.linspace(margin, w - margin, 5)
    y_positions = np.linspace(margin, h - margin, 5)
    
    # Generate all combinations (5x5 = 25 points)
    zones = list(itertools.product(x_positions, y_positions))
    
    # Shuffle for random order
    random.shuffle(zones)
    
    return zones


def backup_model_once(model_path: Path, backup_suffix="_pretrained"):
    """
    Backup original model weights once (if not already backed up).
    Does NOT modify the model.
    """
    backup_path = model_path.with_name(
        model_path.stem + backup_suffix + model_path.suffix
    )
    if not backup_path.exists():
        shutil.copy2(model_path, backup_path)
        print(f"[+] Backed up original model to {backup_path}")
    else:
        print(f"[!] Backup already exists at {backup_path}, not overwriting")


# --------- smooth continuous sweep calibration (second step) ---------

def run_sweep_calibration(detector, predictor, screen, target, w, h, preds, targets):
    """
    Second calibration step (smooth pursuit):

      - The dot moves along a continuous Lissajous-like path:
            x(t) = cx + Ax * sin(2π t)
            y(t) = cy + Ay * sin(2π * 2t + φ)

      - At each frame we record (xh, yh) + (tx, ty).

      - All samples are appended to preds/targets (passed by reference).

    Goal: ~500–700 samples at typical 30 FPS (~20 seconds).
    """
    font  = pygame.font.SysFont(None, 30)
    clock = pygame.time.Clock()
    fps   = SETTINGS["record_frame_rate"]

    # ---------- Intro screen ----------
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("[!] Sweep calibration skipped by user")
                    return True   # skip, but keep whatever we already have
                elif event.key == pygame.K_SPACE:
                    waiting = False

        screen.fill(COLOURS["black"])
        t1 = font.render("Second calibration step (smooth pursuit):", True, COLOURS["white"])
        t2 = font.render("Follow the moving dot with your eyes.", True, COLOURS["white"])
        t3 = font.render("Press SPACE to start, ESC to skip.", True, COLOURS["white"])
        screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*2))
        screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()))
        screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2))
        pygame.display.flip()
        clock.tick(30)

    # ---------- Continuous path parameters ----------
    margin   = SETTINGS["target_radius"] * 2
    cx, cy   = w / 2.0, h / 2.0
    Ax       = (w / 2.0) - margin
    Ay       = (h / 2.0) - margin
    duration = 20.0                         # seconds
    total_frames = max(1, int(duration * fps))
    phase    = np.pi / 2.0                  # phase shift for vertical motion

    print(f"[*] Sweep calibration: {total_frames} frames (~{duration:.1f}s)")

    # ---------- Run continuous smooth pursuit ----------
    for i in range(total_frames):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                print("[!] Sweep calibration aborted by user")
                print(f"[*] Collected {i} sweep samples so far")
                return False  # abort whole calibration

        t = i / (total_frames - 1) if total_frames > 1 else 1.0

        # Lissajous-style path
        tx = cx + Ax * np.sin(2.0 * np.pi * t)
        ty = cy + Ay * np.sin(2.0 * np.pi * 2.0 * t + phase)

        # Render target
        screen.fill(COLOURS["black"])
        target.x = tx
        target.y = ty
        target.radius = SETTINGS["target_radius"]
        target.render(screen)

        msg = "Follow the dot – ESC to abort"
        txt = font.render(msg, True, COLOURS["white"])
        screen.blit(txt, (20, 20))

        # Capture gaze from webcam
        l_eye, r_eye, face, face_al, head_pos, head_ang = detector.get_frame()
        x_hat, y_hat = predictor.predict(
            face_al, l_eye, r_eye, head_pos, head_angle=head_ang
        )

        preds.append((float(x_hat), float(y_hat)))
        targets.append((tx, ty))

        pygame.display.flip()
        clock.tick(fps)

    print(f"[*] Sweep calibration collected {total_frames} smooth samples")
    return True


# --------- calibration UI loop with affine and polynomial models (5x5 grid, 1.5s per point) ---------

def run_calibration(detector, predictor, screen, target, w, h):
    """
    Calibration flow with affine and polynomial models:
    
    - 5x5 grid (25 points) covering the screen
    - For each point:
        * Show dot at that position
        * User presses SPACE when fixating
        * Record for 1.5 seconds (~50 frames at 30 FPS)
        * Show error visualization (line from target to prediction)
        * Store all (pred, target) pairs
    
    - After calibration:
        * Fit affine transformation (linear mapping)
        * Fit polynomial (poly2) transformation (quadratic mapping)
        * Save both calibration matrices
    """
    
    font = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    fps = SETTINGS["record_frame_rate"]
    
    # Use 5x5 grid (25 points)
    zones = get_calibration_zones_5x5(w, h, SETTINGS["target_radius"])
    print(f"[*] Using {len(zones)} calibration points (5x5 grid)")
    
    # Collect all predictions and targets (not averaged per point)
    all_preds = []  # Raw model predictions
    all_targets = []  # True screen coordinates
    
    # Record for 1.5 seconds per point (~50 frames at 30 FPS)
    duration_per_point = 1.5  # seconds
    frames_per_point = int(duration_per_point * fps)
    
    # --------- Step 1: 25-point calibration with error visualization ---------
    for idx, (tx, ty) in enumerate(zones):
        done_point = False
        
        while not done_point:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("[!] Calibration aborted by user")
                        return False
                    elif event.key == pygame.K_SPACE:
                        # User confirms: "I'm looking at the dot now"
                        buf_preds = []
                        buf_targets = []
                        start_time = pygame.time.get_ticks()
                        
                        while (pygame.time.get_ticks() - start_time) / 1000.0 < duration_per_point:
                            # Allow ESC during capture
                            for e2 in pygame.event.get():
                                if e2.type == pygame.KEYDOWN and e2.key == pygame.K_ESCAPE:
                                    print("[!] Calibration aborted during capture")
                                    return False
                            
                            # Get prediction
                            l_eye, r_eye, face, face_al, head_pos, head_ang = detector.get_frame()
                            x_hat, y_hat = predictor.predict(
                                face_al, l_eye, r_eye, head_pos, head_angle=head_ang
                            )
                            
                            buf_preds.append((x_hat, y_hat))
                            buf_targets.append((tx, ty))
                            
                            # Calculate error for visualization
                            error_x = x_hat - tx
                            error_y = y_hat - ty
                            error_px = np.sqrt(error_x**2 + error_y**2)
                            
                            # Calculate angle error (in degrees)
                            # Angle between vectors: target->prediction and target->center
                            if error_px > 0:
                                # Normalize error vector
                                error_vec = np.array([error_x, error_y]) / error_px
                                # Angle in degrees
                                angle_error = np.arctan2(error_y, error_x) * 180 / np.pi
                            else:
                                angle_error = 0.0
                            
                            # Draw visualization
                            screen.fill(COLOURS["black"])
                            
                            # Draw target (green circle)
                            target.x = tx
                            target.y = ty
                            target.radius = SETTINGS["target_radius"]
                            pygame.draw.circle(screen, COLOURS["green"], (int(tx), int(ty)), 
                                             SETTINGS["target_radius"], 3)
                            
                            # Draw prediction (red circle)
                            pygame.draw.circle(screen, COLOURS["red"], (int(x_hat), int(y_hat)), 
                                             SETTINGS["target_radius"] // 2, 2)
                            
                            # Draw error line (from target to prediction)
                            pygame.draw.line(screen, COLOURS["yellow"], 
                                           (int(tx), int(ty)), 
                                           (int(x_hat), int(y_hat)), 2)
                            
                            # Display info
                            elapsed = (pygame.time.get_ticks() - start_time) / 1000.0
                            progress = min(1.0, elapsed / duration_per_point)
                            
                            info_lines = [
                                f"Point {idx+1}/{len(zones)}",
                                f"Recording: {elapsed:.1f}s / {duration_per_point:.1f}s",
                                f"Error: {error_px:.1f} px",
                                f"Angle: {angle_error:.1f}°",
                                f"Target: ({tx:.0f}, {ty:.0f})",
                                f"Prediction: ({x_hat:.0f}, {y_hat:.0f})"
                            ]
                            
                            y_offset = 20
                            for i, line in enumerate(info_lines):
                                color = COLOURS["green"] if i == 0 else COLOURS["white"]
                                txt = font_small.render(line, True, color)
                                screen.blit(txt, (20, y_offset))
                                y_offset += font_small.get_height() + 3
                            
                            # Progress bar
                            bar_width = 300
                            bar_height = 10
                            bar_x, bar_y = 20, y_offset + 10
                            pygame.draw.rect(screen, COLOURS["gray"], 
                                           (bar_x, bar_y, bar_width, bar_height))
                            pygame.draw.rect(screen, COLOURS["green"], 
                                           (bar_x, bar_y, int(bar_width * progress), bar_height))
                            
                            pygame.display.flip()
                            clock.tick(fps)
                        
                        # Add all collected frames to dataset
                        all_preds.extend(buf_preds)
                        all_targets.extend(buf_targets)
                        
                        print(f"[*] Point {idx+1}/{len(zones)}: Collected {len(buf_preds)} samples")
                        done_point = True
                        break
            
            if done_point:
                break
            
            # Normal idle display while waiting for SPACE
            screen.fill(COLOURS["black"])
            target.x = tx
            target.y = ty
            target.radius = SETTINGS["target_radius"]
            pygame.draw.circle(screen, COLOURS["green"], (int(tx), int(ty)), 
                             SETTINGS["target_radius"], 3)
            
            txt1 = font.render(
                f"Calibration point {idx+1}/{len(zones)}", True, COLOURS["white"]
            )
            txt2 = font.render(
                "Look at the dot and press SPACE when ready",
                True,
                COLOURS["white"],
            )
            txt3 = font.render(
                "ESC to abort calibration",
                True,
                COLOURS["white"],
            )
            screen.blit(txt1, (20, 20))
            screen.blit(txt2, (20, 20 + txt1.get_height()))
            screen.blit(txt3, (20, 20 + txt1.get_height() * 2))
            
            pygame.display.flip()
            clock.tick(30)
    
    # --------- Fit calibration models (affine and polynomial) ---------
    print(f"[*] Fitting calibration models on {len(all_preds)} samples...")
    
    if len(all_preds) < 3:
        print("[!] Error: Need at least 3 samples for calibration")
        return False
    
    # Convert to numpy arrays
    preds_arr = np.array(all_preds, dtype=np.float64)  # Model predictions
    targets_arr = np.array(all_targets, dtype=np.float64)  # True coordinates
    
    # Fit affine calibration
    if len(all_preds) >= 3:
        M_affine = fit_affine(preds_arr, targets_arr)
        
        # Evaluate affine calibration
        xh, yh = preds_arr[:, 0], preds_arr[:, 1]
        Phi_affine = np.stack([np.ones_like(xh), xh, yh], axis=1)
        preds_affine = Phi_affine @ M_affine
        err_affine = np.linalg.norm(preds_affine - targets_arr, axis=1)
        rmse_affine = float(np.sqrt(np.mean(err_affine**2)))
        mae_affine = float(np.mean(err_affine))
        max_err_affine = float(np.max(err_affine))
        
        print(f"[*] Affine calibration metrics:")
        print(f"    RMSE: {rmse_affine:.2f} px")
        print(f"    MAE: {mae_affine:.2f} px")
        print(f"    Max error: {max_err_affine:.2f} px")
        
        # Save affine calibration
        CALIB_PATH_AFFINE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CALIB_PATH_AFFINE, M_affine)
        print(f"[+] Saved affine calibration to {CALIB_PATH_AFFINE}")
    
    # Fit polynomial (poly2) calibration
    if len(all_preds) >= 6:  # Poly2 needs at least 6 points
        W_poly2 = fit_poly2(preds_arr, targets_arr)
        
        # Evaluate poly2 calibration
        xh, yh = preds_arr[:, 0], preds_arr[:, 1]
        Phi_poly2 = np.stack([
            np.ones_like(xh),
            xh,
            yh,
            xh**2,
            xh * yh,
            yh**2,
        ], axis=1)
        preds_poly2 = Phi_poly2 @ W_poly2
        err_poly2 = np.linalg.norm(preds_poly2 - targets_arr, axis=1)
        rmse_poly2 = float(np.sqrt(np.mean(err_poly2**2)))
        mae_poly2 = float(np.mean(err_poly2))
        max_err_poly2 = float(np.max(err_poly2))
        
        print(f"[*] Poly2 calibration metrics:")
        print(f"    RMSE: {rmse_poly2:.2f} px")
        print(f"    MAE: {mae_poly2:.2f} px")
        print(f"    Max error: {max_err_poly2:.2f} px")
        
        # Save poly2 calibration
        CALIB_PATH_POLY2.parent.mkdir(parents=True, exist_ok=True)
        np.save(CALIB_PATH_POLY2, W_poly2)
        print(f"[+] Saved poly2 calibration to {CALIB_PATH_POLY2}")
        
        # Use poly2 metrics for display (usually better)
        rmse = rmse_poly2
        mae = mae_poly2
        calib_type_used = "poly2"
    else:
        # Use affine metrics for display
        rmse = rmse_affine
        mae = mae_affine
        calib_type_used = "affine"
        print(f"[!] Warning: Only {len(all_preds)} samples. Poly2 needs at least 6 samples.")
        print(f"[!] Only affine calibration was saved.")
    
    # Completion screen
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    done = True
                elif event.key == pygame.K_ESCAPE:
                    return False
        
        screen.fill(COLOURS["black"])
        t1 = font.render("Calibration complete!", True, COLOURS["white"])
        t2 = font.render(
            f"{calib_type_used.upper()} RMSE: {rmse:.1f} px (MAE: {mae:.1f} px)",
            True,
            COLOURS["white"],
        )
        t3 = font.render(
            f"Trained on {len(all_preds)} samples from {len(zones)} points",
            True,
            COLOURS["white"],
        )
        t4 = font.render("Press SPACE to return to menu", True, COLOURS["white"])
        
        screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*2))
        screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()))
        screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2))
        screen.blit(t4, (w // 2 - t4.get_width() // 2, h // 2 + t1.get_height()))
        
        pygame.display.flip()
        clock.tick(30)
    
    return True


# --------- tracking UI loop (SPACE to return to menu) with poly2 ---------

def run_tracking(detector, predictor, screen, target, w, h, screen_errors, calib_affine=None, calib_poly2=None):
    """
    Real-time tracking view using current model (pretrained) + optional calibration.

    - By default: shows raw model output only
    - Press 'A': applies affine transformation (if available)
    - Press 'P': applies poly2 transformation (if available)
    - Press 'N': returns to raw output
    - SPACE returns to menu, ESC quits.
    """
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont(None, 30)

    win_pos = SETTINGS["avg_window_length"]
    win_err = win_pos * 2

    track_x   = deque([0] * win_pos, maxlen=win_pos)
    track_y   = deque([0] * win_pos, maxlen=win_pos)
    track_err = deque([0] * win_err, maxlen=win_err)

    w_pos = np.arange(1, win_pos + 1)
    w_err = np.arange(1, win_err + 1)

    # Current calibration mode: None (raw), "affine", or "poly2"
    active_calib_mode = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # exit program
                elif event.key == pygame.K_SPACE:
                    running = False  # back to menu
                elif event.key == pygame.K_a:
                    # Toggle affine calibration
                    if calib_affine is not None:
                        if active_calib_mode == "affine":
                            active_calib_mode = None  # Turn off
                        else:
                            active_calib_mode = "affine"  # Turn on
                elif event.key == pygame.K_p:
                    # Toggle poly2 calibration
                    if calib_poly2 is not None:
                        if active_calib_mode == "poly2":
                            active_calib_mode = None  # Turn off
                        else:
                            active_calib_mode = "poly2"  # Turn on
                elif event.key == pygame.K_n:
                    # Return to raw output
                    active_calib_mode = None

        # Get frame + raw prediction
        l_eye, r_eye, face, face_al, head_pos, head_ang = detector.get_frame()
        x_hat, y_hat = predictor.predict(
            face_al, l_eye, r_eye, head_pos, head_angle=head_ang
        )

        # Apply calibration based on active mode
        x_use, y_use = x_hat, y_hat  # Default: raw output
        x_cal, y_cal = None, None
        
        if active_calib_mode == "affine" and calib_affine is not None:
            x_cal, y_cal = apply_affine(calib_affine, x_hat, y_hat)
            x_use, y_use = x_cal, y_cal  # Use calibrated output
        elif active_calib_mode == "poly2" and calib_poly2 is not None:
            x_cal, y_cal = apply_poly2(calib_poly2, x_hat, y_hat)
            x_use, y_use = x_cal, y_cal  # Use calibrated output

        # Rolling buffers based on current coords
        track_x.append(x_use)
        track_y.append(y_use)

        x_clamp = clamp_value(int(x_use), w - 1)
        y_clamp = clamp_value(int(y_use), h - 1)
        err_px  = screen_errors[x_clamp, y_clamp] * 0.75
        track_err.append(err_px)

        # Weighted smoothing
        x_vis = float(np.average(track_x, weights=w_pos))
        y_vis = float(np.average(track_y, weights=w_pos))
        rad   = float(np.average(track_err, weights=w_err))

        x_vis = max(0, min(x_vis, w - 1))
        y_vis = max(0, min(y_vis, h - 1))

        target.x      = x_vis
        target.y      = y_vis
        target.radius = rad

        # Draw
        screen.fill(COLOURS["black"])
        target.render(screen)
        
        # Display raw output
        txt = font.render(f"raw: ({x_hat:5.1f}, {y_hat:5.1f})", True, COLOURS["white"])
        screen.blit(txt, (20, 20))
        
        y_offset = txt.get_height()
        
        # Display current mode and calibrated output if active
        if active_calib_mode == "affine" and x_cal is not None:
            mode_text = font.render("Mode: AFFINE (A to toggle)", True, COLOURS["green"])
            screen.blit(mode_text, (20, 20 + y_offset))
            y_offset += mode_text.get_height()
            
            calib_text = font.render(f"calibrated: ({x_cal:5.1f}, {y_cal:5.1f})", True, COLOURS["yellow"])
            screen.blit(calib_text, (20, 20 + y_offset))
            y_offset += calib_text.get_height()
        elif active_calib_mode == "poly2" and x_cal is not None:
            mode_text = font.render("Mode: POLY2 (P to toggle)", True, COLOURS["green"])
            screen.blit(mode_text, (20, 20 + y_offset))
            y_offset += mode_text.get_height()
            
            calib_text = font.render(f"calibrated: ({x_cal:5.1f}, {y_cal:5.1f})", True, COLOURS["yellow"])
            screen.blit(calib_text, (20, 20 + y_offset))
            y_offset += calib_text.get_height()
        else:
            mode_text = font.render("Mode: RAW (A=affine, P=poly2, N=raw)", True, COLOURS["white"])
            screen.blit(mode_text, (20, 20 + y_offset))
            y_offset += mode_text.get_height()
        
        # Show available calibrations
        available_text = "Available: "
        if calib_affine is not None:
            available_text += "AFFINE "
        if calib_poly2 is not None:
            available_text += "POLY2"
        if calib_affine is None and calib_poly2 is None:
            available_text += "none"
        
        avail_text = font.render(available_text, True, COLOURS["gray"])
        screen.blit(avail_text, (20, 20 + y_offset))
        y_offset += avail_text.get_height()
        
        # Controls
        controls_text = font.render("SPACE: menu | ESC: quit", True, COLOURS["white"])
        screen.blit(controls_text, (20, 20 + y_offset))

        pygame.display.flip()
        clock.tick(SETTINGS["record_frame_rate"])

    return True


# --------- data collection UI loop (saves images + positions) ---------

def get_grid_positions(w, h, grid_size=9, margin=None):
    """
    Generate a grid of positions covering the whole screen.
    
    Args:
        w, h: Screen width and height
        grid_size: Number of positions per axis (default 9 for 9x9 = 81 positions)
        margin: Optional margin from edges (defaults to target_radius * 2)
    
    Returns:
        List of (x, y) tuples covering the screen in a grid pattern
    """
    if margin is None:
        margin = SETTINGS["target_radius"] * 2
    
    # Create grid positions
    x_positions = np.linspace(margin, w - margin, grid_size)
    y_positions = np.linspace(margin, h - margin, grid_size)
    
    # Generate all combinations
    positions = list(itertools.product(x_positions, y_positions))
    
    # Shuffle for random order
    random.shuffle(positions)
    
    return positions


def run_data_collection(detector, predictor, screen, target, w, h):
    """
    Data collection mode:
    - Target moves through a 9x9 grid (81 positions) covering the whole screen
    - User follows with eyes
    - Captures data continuously while moving between positions
    - Saves images (face_aligned, l_eye, r_eye, head_pos) + positions.csv
    - Shows completion tracker at top-left
    """
    font = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    fps = SETTINGS["record_frame_rate"]
    skip_frames = SETTINGS.get("skip_frames", 3)
    
    # Setup data directory structure
    DATA_DIR.mkdir(exist_ok=True)
    img_dirs = {
        "face_aligned": DATA_DIR / "face_aligned",
        "l_eye": DATA_DIR / "l_eye",
        "r_eye": DATA_DIR / "r_eye",
        "head_pos": DATA_DIR / "head_pos",
    }
    for d in img_dirs.values():
        d.mkdir(exist_ok=True)
    
    # Load or create positions.csv
    positions_file = DATA_DIR / "positions.csv"
    if positions_file.exists():
        df = pd.read_csv(positions_file)
        start_id = df["id"].max() + 1 if len(df) > 0 else 0
        existing_samples = len(df)
        print(f"[*] Resuming data collection from ID {start_id} ({existing_samples} existing samples)")
    else:
        df = pd.DataFrame(columns=["id", "x", "y", "head_angle"])
        start_id = 0
        existing_samples = 0
        print("[*] Starting new data collection")
    
    # Generate 9x9 grid (81 positions)
    grid_positions = get_grid_positions(w, h, grid_size=9)
    target_samples = len(grid_positions)  # Exactly 81 samples
    current_id = start_id
    frame_count = 0
    grid_index = 0
    
    # Intro screen
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("[!] Data collection cancelled by user")
                    return False
                elif event.key == pygame.K_SPACE:
                    waiting = False
        
        screen.fill(COLOURS["black"])
        t1 = font.render("Data Collection Mode", True, COLOURS["white"])
        t2 = font.render("Follow the moving dot with your eyes.", True, COLOURS["white"])
        t3 = font.render(f"Grid: 9x9 = {target_samples} positions", True, COLOURS["white"])
        t4 = font.render("Press SPACE to start, ESC to cancel.", True, COLOURS["white"])
        
        screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*3))
        screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()*2))
        screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2 - t3.get_height()))
        screen.blit(t4, (w // 2 - t4.get_width() // 2, h // 2))
        pygame.display.flip()
        clock.tick(30)
    
    # Main collection loop
    running = True
    target.x = w // 2
    target.y = h // 2
    
    # Movement parameters - faster but still smooth
    movement_threshold = 10.0  # Consider "arrived" when within this many pixels
    
    # Use 70% of normal speed (faster than before but not instant)
    original_speed = target.speed
    target.speed = original_speed * 0.7
    
    # Start with first grid position
    if grid_index < len(grid_positions):
        target_location = grid_positions[grid_index]
    
    while running and grid_index < len(grid_positions):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    running = False
                    break
        
        if not running:
            break
        
        # Check if target has reached its destination
        dist_to_target = np.sqrt((target.x - target_location[0])**2 + (target.y - target_location[1])**2)
        
        # Move to next grid position if we've reached the current one
        if dist_to_target < movement_threshold and grid_index < len(grid_positions):
            grid_index += 1
            if grid_index < len(grid_positions):
                target_location = grid_positions[grid_index]
        
        # Smoothly move target towards destination
        delta_time_ms = int(1000 / fps)  # Convert FPS to milliseconds per frame
        target.move(target_location, delta_time_ms)
        
        # Capture data continuously while moving (respect skip_frames)
        frame_count += 1
        if frame_count % skip_frames == 0:
            l_eye, r_eye, face, face_al, head_pos, head_ang = detector.get_frame()
            
            # Save images
            filename = f"{current_id}.jpg"
            
            # Convert numpy arrays to PIL Images and save
            for img_type, img_dir in img_dirs.items():
                if img_type == "face_aligned":
                    img = face_al
                elif img_type == "l_eye":
                    img = l_eye
                elif img_type == "r_eye":
                    img = r_eye
                elif img_type == "head_pos":
                    img = head_pos
                else:
                    continue
                
                # Ensure image is uint8 before color conversion
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        # Normalized float [0, 1] -> uint8 [0, 255]
                        img = (img * 255).astype(np.uint8)
                    else:
                        # Float with values > 1 -> clip and convert
                        img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Convert to PIL Image
                if img_type == "head_pos":
                    # Grayscale
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    pil_img = Image.fromarray(img, mode='L')
                else:
                    # RGB
                    if img.ndim == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                
                pil_img.save(img_dir / filename)
            
            # Save position (use current target position, not destination)
            new_row = pd.DataFrame({
                "id": [current_id],
                "x": [target.x],
                "y": [target.y],
                "head_angle": [head_ang]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            
            current_id += 1
        
        # Render frame
        screen.fill(COLOURS["black"])
        target.render(screen)
        
        # Completion tracker (top-left)
        progress = grid_index / target_samples
        progress_pct = int(progress * 100)
        tracker_text = f"Progress: {progress_pct}% ({grid_index}/{target_samples})"
        tracker_surf = font_small.render(tracker_text, True, COLOURS["white"])
        screen.blit(tracker_surf, (10, 10))
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x, bar_y = 10, 40
        pygame.draw.rect(screen, COLOURS["gray"], (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, COLOURS["green"], (bar_x, bar_y, int(bar_width * progress), bar_height))
        pygame.draw.rect(screen, COLOURS["white"], (bar_x, bar_y, bar_width, bar_height), 2)
        
        info_text = font_small.render("SPACE: finish | ESC: cancel", True, COLOURS["white"])
        screen.blit(info_text, (10, h - info_text.get_height() - 10))
        
        pygame.display.flip()
        clock.tick(fps)
    
    # Save data
    df.to_csv(positions_file, index=False)
    print(f"[+] Saved {len(df)} total samples to {positions_file}")
    print(f"[+] Collected {current_id - start_id} new samples from grid")
    
    # Completion screen
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_ESCAPE:
                    done = True
        
        screen.fill(COLOURS["black"])
        t1 = font.render("Data Collection Complete!", True, COLOURS["white"])
        t2 = font.render(f"Collected {current_id - start_id} new samples", True, COLOURS["white"])
        t3 = font.render(f"Total samples: {len(df)}", True, COLOURS["white"])
        t4 = font.render("Press SPACE/ESC to return to menu", True, COLOURS["white"])
        
        screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*2))
        screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()))
        screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2))
        screen.blit(t4, (w // 2 - t4.get_width() // 2, h // 2 + t1.get_height()))
        
        pygame.display.flip()
        clock.tick(30)
    
    return True


# --------- cursor-based data collection UI loop ---------

def run_cursor_data_collection(detector, predictor, screen, w, h):
    """
    Cursor-based data collection mode:
    - User moves cursor to indicate where they're looking
    - On mouse click, captures webcam data and associates it with cursor position
    - Saves images (face_aligned, l_eye, r_eye, head_pos) + positions.csv
    - Useful for targeted data collection to improve fine-tuning
    """
    font = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    fps = SETTINGS["record_frame_rate"]
    
    # Enable mouse visibility for this mode
    pygame.mouse.set_visible(True)
    
    # Setup data directory structure (same as run_data_collection)
    DATA_DIR.mkdir(exist_ok=True)
    img_dirs = {
        "face_aligned": DATA_DIR / "face_aligned",
        "l_eye": DATA_DIR / "l_eye",
        "r_eye": DATA_DIR / "r_eye",
        "head_pos": DATA_DIR / "head_pos",
    }
    for d in img_dirs.values():
        d.mkdir(exist_ok=True)
    
    # Load or create positions.csv
    positions_file = DATA_DIR / "positions.csv"
    if positions_file.exists():
        df = pd.read_csv(positions_file)
        start_id = df["id"].max() + 1 if len(df) > 0 else 0
        existing_samples = len(df)
        print(f"[*] Resuming cursor data collection from ID {start_id} ({existing_samples} existing samples)")
    else:
        df = pd.DataFrame(columns=["id", "x", "y", "head_angle"])
        start_id = 0
        existing_samples = 0
        print("[*] Starting new cursor data collection")
    
    current_id = start_id
    samples_collected = 0
    
    # Intro screen
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("[!] Cursor data collection cancelled by user")
                    pygame.mouse.set_visible(False)
                    return False
                elif event.key == pygame.K_SPACE:
                    waiting = False
        
        screen.fill(COLOURS["black"])
        t1 = font.render("Cursor Data Collection Mode", True, COLOURS["white"])
        t2 = font.render("Move your cursor to where you're looking, then CLICK to capture.", True, COLOURS["white"])
        t3 = font.render("This will improve fine-tuning with targeted data.", True, COLOURS["white"])
        t4 = font.render("Press SPACE to start, ESC to cancel.", True, COLOURS["white"])
        
        screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*3))
        screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()*2))
        screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2 - t3.get_height()))
        screen.blit(t4, (w // 2 - t4.get_width() // 2, h // 2))
        pygame.display.flip()
        clock.tick(30)
    
    # Create static heatmap from initial data (before starting collection)
    # Use a copy of df to preserve the initial state
    initial_df = df.copy()
    static_heatmap = create_heatmap_surface(w, h, initial_df, grid_size=50, alpha=120)
    
    # Main collection loop
    running = True
    last_click_time = 0
    click_cooldown = 0.2  # Minimum seconds between clicks (to avoid accidental double-clicks)
    
    while running:
        current_time = pygame.time.get_ticks() / 1000.0  # Current time in seconds
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == pygame.K_SPACE:
                    running = False
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Left click to capture data
                if event.button == 1:  # Left mouse button
                    if current_time - last_click_time >= click_cooldown:
                        # Get cursor position
                        cursor_x, cursor_y = pygame.mouse.get_pos()
                        
                        # Clamp to screen bounds
                        cursor_x = max(0, min(cursor_x, w - 1))
                        cursor_y = max(0, min(cursor_y, h - 1))
                        
                        # Capture frame from webcam
                        l_eye, r_eye, face, face_al, head_pos, head_ang = detector.get_frame()
                        
                        # Save images
                        filename = f"{current_id}.jpg"
                        
                        # Convert numpy arrays to PIL Images and save
                        for img_type, img_dir in img_dirs.items():
                            if img_type == "face_aligned":
                                img = face_al
                            elif img_type == "l_eye":
                                img = l_eye
                            elif img_type == "r_eye":
                                img = r_eye
                            elif img_type == "head_pos":
                                img = head_pos
                            else:
                                continue
                            
                            # Ensure image is uint8 before color conversion
                            if img.dtype != np.uint8:
                                if img.max() <= 1.0:
                                    # Normalized float [0, 1] -> uint8 [0, 255]
                                    img = (img * 255).astype(np.uint8)
                                else:
                                    # Float with values > 1 -> clip and convert
                                    img = np.clip(img, 0, 255).astype(np.uint8)
                            
                            # Convert to PIL Image
                            if img_type == "head_pos":
                                # Grayscale
                                if img.ndim == 3:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                pil_img = Image.fromarray(img, mode='L')
                            else:
                                # RGB
                                if img.ndim == 3:
                                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img)
                            
                            pil_img.save(img_dir / filename)
                        
                        # Save position (cursor position)
                        new_row = pd.DataFrame({
                            "id": [current_id],
                            "x": [cursor_x],
                            "y": [cursor_y],
                            "head_angle": [head_ang]
                        })
                        df = pd.concat([df, new_row], ignore_index=True)
                        
                        current_id += 1
                        samples_collected += 1
                        last_click_time = current_time
                        
                        print(f"[*] Captured sample {current_id} at ({cursor_x:.1f}, {cursor_y:.1f})")
        
        # Get current cursor position for display
        cursor_x, cursor_y = pygame.mouse.get_pos()
        cursor_x = max(0, min(cursor_x, w - 1))
        cursor_y = max(0, min(cursor_y, h - 1))
        
        # Render frame
        screen.fill(COLOURS["black"])
        
        # Draw static heatmap background (no updates during collection)
        if static_heatmap is not None:
            screen.blit(static_heatmap, (0, 0))
        
        # Draw crosshair at cursor position (on top of heatmap)
        crosshair_size = 20
        # Horizontal line
        pygame.draw.line(screen, COLOURS["white"], 
                         (cursor_x - crosshair_size, cursor_y), 
                         (cursor_x + crosshair_size, cursor_y), 3)
        # Vertical line
        pygame.draw.line(screen, COLOURS["white"], 
                         (cursor_x, cursor_y - crosshair_size), 
                         (cursor_x, cursor_y + crosshair_size), 3)
        # Center circle
        pygame.draw.circle(screen, COLOURS["red"], (int(cursor_x), int(cursor_y)), 8, 3)
        
        # Display instructions and stats
        y_offset = 20
        instructions = [
            f"Samples collected this session: {samples_collected}",
            f"Total samples: {len(df)}",
            f"Cursor: ({cursor_x:.0f}, {cursor_y:.0f})",
            "",
            "Heatmap: Green=low, Yellow=medium, Red=high density",
            "Dark areas = missing data (focus here!)",
            "",
            "LEFT CLICK: Capture data at cursor position",
            "SPACE: Finish and save | ESC: Cancel"
        ]
        
        for i, text in enumerate(instructions):
            if text:  # Skip empty lines
                color = COLOURS["green"] if i == 0 else COLOURS["white"]
                txt = font_small.render(text, True, color)
                screen.blit(txt, (20, y_offset))
            y_offset += font_small.get_height() + 5
        
        pygame.display.flip()
        clock.tick(fps)
    
    # Disable mouse visibility when exiting
    pygame.mouse.set_visible(False)
    
    # Save data
    df.to_csv(positions_file, index=False)
    print(f"[+] Saved {len(df)} total samples to {positions_file}")
    print(f"[+] Collected {samples_collected} new samples from cursor clicks")
    
    # Completion screen
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_ESCAPE:
                    done = True
        
        screen.fill(COLOURS["black"])
        t1 = font.render("Cursor Data Collection Complete!", True, COLOURS["white"])
        t2 = font.render(f"Collected {samples_collected} new samples", True, COLOURS["white"])
        t3 = font.render(f"Total samples: {len(df)}", True, COLOURS["white"])
        t4 = font.render("Press SPACE/ESC to return to menu", True, COLOURS["white"])
        
        screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*2))
        screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()))
        screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2))
        screen.blit(t4, (w // 2 - t4.get_width() // 2, h // 2 + t1.get_height()))
        
        pygame.display.flip()
        clock.tick(30)
    
    return True


# --------- data collection statistics visualization ---------

def create_heatmap_surface(w, h, df, grid_size=50, alpha=150):
    """
    Create a static heatmap surface showing data collection density.
    
    Args:
        w, h: Screen width and height
        df: DataFrame with 'x' and 'y' columns
        grid_size: Size of the grid for density calculation
        alpha: Transparency level (0-255)
    
    Returns:
        pygame.Surface: Static heatmap surface (or None if no data)
    """
    if len(df) == 0:
        return None  # No data to visualize
    
    # Create a surface for the heatmap
    heatmap_surface = pygame.Surface((w, h))
    heatmap_surface.fill(COLOURS["black"])
    
    # Clamp positions to screen bounds
    df_clamped = df.copy()
    df_clamped['x'] = df_clamped['x'].clip(0, w - 1)
    df_clamped['y'] = df_clamped['y'].clip(0, h - 1)
    
    # Create density map (2D histogram)
    x_bins = np.linspace(0, w, grid_size + 1)
    y_bins = np.linspace(0, h, grid_size + 1)
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        df_clamped['x'].values, 
        df_clamped['y'].values, 
        bins=[x_bins, y_bins]
    )
    
    # Normalize histogram for visualization (0-255)
    hist_max = hist.max()
    if hist_max > 0:
        hist_norm = (hist / hist_max * 255).astype(np.uint8)
    else:
        hist_norm = hist.astype(np.uint8)
    
    # Draw density heatmap
    cell_w = w / grid_size
    cell_h = h / grid_size
    
    for i in range(grid_size):
        for j in range(grid_size):
            density = hist_norm[i, j]
            if density > 0:
                # Color: green (low) -> yellow (medium) -> red (high)
                if density < 85:
                    # Low density - green
                    color_intensity = int(density * 3)
                    color = (0, min(255, color_intensity), 0)
                elif density < 170:
                    # Medium density - yellow
                    color_intensity = int((density - 85) * 3)
                    color = (min(255, color_intensity), 255, 0)
                else:
                    # High density - red
                    color_intensity = int((density - 170) * 3)
                    color = (255, max(0, 255 - color_intensity), 0)
                
                # Draw cell with transparency
                x_pos = int(i * cell_w)
                y_pos = int(j * cell_h)
                alpha_surface = pygame.Surface((int(cell_w) + 1, int(cell_h) + 1))
                alpha_surface.set_alpha(min(alpha, density))
                alpha_surface.fill(color)
                heatmap_surface.blit(alpha_surface, (x_pos, y_pos))
    
    # Set overall transparency for the heatmap
    heatmap_surface.set_alpha(alpha)
    return heatmap_surface


def run_data_stats(screen, w, h):
    """
    Display statistics panel showing where data has been collected.
    - Shows a heatmap/density visualization of collected data points
    - Highlights areas with low/no coverage
    - Displays statistics about data distribution
    """
    font = pygame.font.SysFont(None, 30)
    font_small = pygame.font.SysFont(None, 24)
    font_large = pygame.font.SysFont(None, 40)
    clock = pygame.time.Clock()
    
    # Load positions.csv
    positions_file = DATA_DIR / "positions.csv"
    if not positions_file.exists():
        # Show error message
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        waiting = False
            
            screen.fill(COLOURS["black"])
            t1 = font_large.render("No data found!", True, COLOURS["red"])
            t2 = font.render(f"positions.csv not found in {DATA_DIR}", True, COLOURS["white"])
            t3 = font.render("Please collect some data first.", True, COLOURS["white"])
            t4 = font.render("Press SPACE/ESC to return to menu", True, COLOURS["white"])
            
            screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*2))
            screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()))
            screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2))
            screen.blit(t4, (w // 2 - t4.get_width() // 2, h // 2 + t1.get_height()))
            pygame.display.flip()
            clock.tick(30)
        return True
    
    # Load data
    df = pd.read_csv(positions_file)
    if len(df) == 0:
        # Show empty data message
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        waiting = False
            
            screen.fill(COLOURS["black"])
            t1 = font_large.render("No data points found!", True, COLOURS["red"])
            t2 = font.render("The positions.csv file is empty.", True, COLOURS["white"])
            t3 = font.render("Please collect some data first.", True, COLOURS["white"])
            t4 = font.render("Press SPACE/ESC to return to menu", True, COLOURS["white"])
            
            screen.blit(t1, (w // 2 - t1.get_width() // 2, h // 2 - t1.get_height()*2))
            screen.blit(t2, (w // 2 - t2.get_width() // 2, h // 2 - t2.get_height()))
            screen.blit(t3, (w // 2 - t3.get_width() // 2, h // 2))
            screen.blit(t4, (w // 2 - t4.get_width() // 2, h // 2 + t1.get_height()))
            pygame.display.flip()
            clock.tick(30)
        return True
    
    # Clamp positions to screen bounds
    df['x'] = df['x'].clip(0, w - 1)
    df['y'] = df['y'].clip(0, h - 1)
    
    # Create density map (2D histogram)
    # Use a grid to count samples per region
    grid_size = 50  # 50x50 grid for density visualization
    x_bins = np.linspace(0, w, grid_size + 1)
    y_bins = np.linspace(0, h, grid_size + 1)
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        df['x'].values, 
        df['y'].values, 
        bins=[x_bins, y_bins]
    )
    
    # Normalize histogram for visualization (0-255)
    hist_max = hist.max()
    if hist_max > 0:
        hist_norm = (hist / hist_max * 255).astype(np.uint8)
    else:
        hist_norm = hist.astype(np.uint8)
    
    # Calculate statistics
    total_samples = len(df)
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_mean, y_mean = df['x'].mean(), df['y'].mean()
    
    # Calculate coverage (percentage of grid cells with at least 1 sample)
    coverage = (hist > 0).sum() / (grid_size * grid_size) * 100
    
    # Calculate density statistics
    non_zero_density = hist[hist > 0]
    if len(non_zero_density) > 0:
        avg_density = non_zero_density.mean()
        max_density = hist.max()
        min_density = non_zero_density.min()
    else:
        avg_density = max_density = min_density = 0
    
    # Main visualization loop
    running = True
    show_points = False  # Toggle to show individual points
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                    running = False
                elif event.key == pygame.K_p:
                    # Toggle point display
                    show_points = not show_points
        
        # Draw background
        screen.fill(COLOURS["black"])
        
        # Draw density heatmap
        cell_w = w / grid_size
        cell_h = h / grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                density = hist_norm[i, j]
                if density > 0:
                    # Color: green (low) -> yellow (medium) -> red (high)
                    if density < 85:
                        # Low density - green
                        color_intensity = int(density * 3)
                        color = (0, min(255, color_intensity), 0)
                    elif density < 170:
                        # Medium density - yellow
                        color_intensity = int((density - 85) * 3)
                        color = (min(255, color_intensity), 255, 0)
                    else:
                        # High density - red
                        color_intensity = int((density - 170) * 3)
                        color = (255, max(0, 255 - color_intensity), 0)
                    
                    # Draw cell with transparency effect
                    x_pos = int(i * cell_w)
                    y_pos = int(j * cell_h)
                    alpha_surface = pygame.Surface((int(cell_w) + 1, int(cell_h) + 1))
                    alpha_surface.set_alpha(min(200, density))
                    alpha_surface.fill(color)
                    screen.blit(alpha_surface, (x_pos, y_pos))
        
        # Draw individual points if enabled
        if show_points:
            for _, row in df.iterrows():
                pygame.draw.circle(screen, COLOURS["white"], 
                                 (int(row['x']), int(row['y'])), 2, 1)
        
        # Draw statistics panel (top-left)
        panel_x, panel_y = 20, 20
        panel_width = 400
        panel_height = 300
        
        # Semi-transparent background for panel
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill(COLOURS["black"])
        screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw panel border
        pygame.draw.rect(screen, COLOURS["white"], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Statistics text
        stats_text = [
            "DATA COLLECTION STATS",
            "",
            f"Total samples: {total_samples}",
            f"Coverage: {coverage:.1f}%",
            "",
            f"X range: {x_min:.0f} - {x_max:.0f}",
            f"Y range: {y_min:.0f} - {y_max:.0f}",
            f"Mean: ({x_mean:.0f}, {y_mean:.0f})",
            "",
            f"Max density: {max_density:.0f}",
            f"Avg density: {avg_density:.1f}",
            "",
            "Controls:",
            "P: Toggle points",
            "SPACE/ESC: Back"
        ]
        
        y_offset = panel_y + 10
        for i, text in enumerate(stats_text):
            if text:
                if i == 0:  # Title
                    txt = font_large.render(text, True, COLOURS["green"])
                elif text.startswith("Controls:"):
                    txt = font_small.render(text, True, COLOURS["yellow"])
                else:
                    txt = font_small.render(text, True, COLOURS["white"])
                screen.blit(txt, (panel_x + 10, y_offset))
            y_offset += font_small.get_height() + 3
        
        # Draw legend (bottom-right)
        legend_x = w - 250
        legend_y = h - 150
        legend_width = 230
        legend_height = 130
        
        legend_surface = pygame.Surface((legend_width, legend_height))
        legend_surface.set_alpha(200)
        legend_surface.fill(COLOURS["black"])
        screen.blit(legend_surface, (legend_x, legend_y))
        
        pygame.draw.rect(screen, COLOURS["white"], 
                        (legend_x, legend_y, legend_width, legend_height), 2)
        
        legend_text = [
            "DENSITY LEGEND",
            "",
            "Green: Low",
            "Yellow: Medium",
            "Red: High",
            "",
            f"Max: {max_density:.0f} samples"
        ]
        
        y_offset = legend_y + 10
        for i, text in enumerate(legend_text):
            if text:
                if i == 0:
                    txt = font.render(text, True, COLOURS["green"])
                elif "Low" in text:
                    txt = font_small.render(text, True, COLOURS["green"])
                elif "Medium" in text:
                    txt = font_small.render(text, True, COLOURS["yellow"])
                elif "High" in text:
                    txt = font_small.render(text, True, COLOURS["red"])
                else:
                    txt = font_small.render(text, True, COLOURS["white"])
                screen.blit(txt, (legend_x + 10, y_offset))
            y_offset += font_small.get_height() + 3
        
        # Draw grid overlay (optional, can be toggled)
        # Draw some grid lines to help visualize regions
        if show_points:
            for i in range(0, grid_size, 5):  # Every 5th line
                x_line = int(i * cell_w)
                pygame.draw.line(screen, COLOURS["gray"], 
                               (x_line, 0), (x_line, h), 1)
            for j in range(0, grid_size, 5):
                y_line = int(j * cell_h)
                pygame.draw.line(screen, COLOURS["gray"], 
                               (0, y_line), (w, y_line), 1)
        
        pygame.display.flip()
        clock.tick(30)
    
    return True


# --------- main menu / entry point (with menu icons) ---------

def main():
    pygame.init()
    pygame.mouse.set_visible(0)
    pygame.display.set_caption("Calibrate and Track (poly2)")

    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h   = pygame.display.get_surface().get_size()

    # Load menu images like in data_collection.py
    menu_calibration = pygame.image.load(MENU_CALIBRATION_IMG).convert_alpha()
    menu_collection  = pygame.image.load(MENU_COLLECTION_IMG).convert_alpha()
    menu_tracking    = pygame.image.load(MENU_TRACKING_IMG).convert_alpha()

    target = Target(
        (w // 2, h // 2),
        speed=SETTINGS["target_speed"],
        radius=SETTINGS["target_radius"],
    )

    # Detector + predictor (no debug camera windows now)
    detector = Detector(output_size=SETTINGS["image_size"])
    
    # Try to load fine-tuned model, fallback to pretrained
    if FINE_TUNED_MODEL_PATH.exists() and FINE_TUNED_CFG_JSON.exists():
        print(f"[+] Loading fine-tuned model from {FINE_TUNED_MODEL_PATH}")
        predictor = Predictor(
            FullModel,
            model_path=FINE_TUNED_MODEL_PATH,
            cfg_json=FINE_TUNED_CFG_JSON,
            gpu=0,
        )
        using_finetuned = True
    else:
        print(f"[*] Using pretrained model from {MODEL_PATH}")
        predictor = Predictor(
            FullModel,
            model_path=MODEL_PATH,
            cfg_json=CFG_JSON,
            gpu=0,
        )
        using_finetuned = False

    # Make sure we have a backup of the original model at least once
    backup_model_once(MODEL_PATH)

    screen_errors = np.load("src/trained_models/full/errors.npy")

    # Try to load calibrations separately (if available)
    calib_affine = None
    calib_poly2 = None
    
    if CALIB_PATH_AFFINE.exists():
        calib_affine = np.load(CALIB_PATH_AFFINE)
        print(f"[+] Loaded existing affine calibration from {CALIB_PATH_AFFINE}")
    
    if CALIB_PATH_POLY2.exists():
        calib_poly2 = np.load(CALIB_PATH_POLY2)
        print(f"[+] Loaded existing poly2 calibration from {CALIB_PATH_POLY2}")

    font_large = pygame.font.SysFont(None, 40)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    detector.close()
                    pygame.quit()
                    sys.exit(0)
                elif event.key == pygame.K_1:
                    ok = run_calibration(detector, predictor, screen, target, w, h)
                    # Reload calibrations from disk after a successful run
                    if ok:
                        if CALIB_PATH_AFFINE.exists():
                            calib_affine = np.load(CALIB_PATH_AFFINE)
                            print(f"[+] Reloaded affine calibration")
                        if CALIB_PATH_POLY2.exists():
                            calib_poly2 = np.load(CALIB_PATH_POLY2)
                            print(f"[+] Reloaded poly2 calibration")
                    if not ok:
                        detector.close()
                        pygame.quit()
                        sys.exit(0)
                elif event.key == pygame.K_2:
                    ok = run_data_collection(detector, predictor, screen, target, w, h)
                    if not ok:
                        detector.close()
                        pygame.quit()
                        sys.exit(0)
                elif event.key == pygame.K_3:
                    ok = run_tracking(
                        detector,
                        predictor,
                        screen,
                        target,
                        w,
                        h,
                        screen_errors,
                        calib_affine,
                        calib_poly2,
                    )
                    if not ok:
                        detector.close()
                        pygame.quit()
                        sys.exit(0)
                elif event.key == pygame.K_4:
                    ok = run_cursor_data_collection(detector, predictor, screen, w, h)
                    if not ok:
                        detector.close()
                        pygame.quit()
                        sys.exit(0)
                elif event.key == pygame.K_5:
                    ok = run_data_stats(screen, w, h)
                    if not ok:
                        detector.close()
                        pygame.quit()
                        sys.exit(0)

        # Draw menu icons in the same vertical layout as data_collection.py
        screen.fill(COLOURS["black"])

        center_x = w // 2
        top_y    = h // 2 - menu_calibration.get_height() - 20

        for idx, img in enumerate([menu_calibration, menu_collection, menu_tracking]):
            rect = img.get_rect(center=(center_x, top_y + idx * (img.get_height() + 20)))
            screen.blit(img, rect)

        # Bottom-left legend (keys)
        padding = 10
        t1 = font_large.render("(1) Calibration", True, COLOURS["white"])
        t2 = font_large.render("(2) Data Collection", True, COLOURS["white"])
        t3 = font_large.render("(3) Tracking", True, COLOURS["white"])
        t4 = font_large.render("(4) Cursor Data Collection", True, COLOURS["white"])
        t5 = font_large.render("(5) Data Statistics", True, COLOURS["white"])
        t6 = font_large.render("(ESC) Exit", True, COLOURS["white"])

        screen.blit(t1, (padding, h - t1.get_height()*6 - padding*6))
        screen.blit(t2, (padding, h - t2.get_height()*5 - padding*5))
        screen.blit(t3, (padding, h - t3.get_height()*4 - padding*4))
        screen.blit(t4, (padding, h - t4.get_height()*3 - padding*3))
        screen.blit(t5, (padding, h - t5.get_height()*2 - padding*2))
        screen.blit(t6, (padding, h - t6.get_height()    - padding))

        # Model and calibration status
        status_y = h - font_large.get_height()*7 - padding*7
        if using_finetuned:
            model_status = font_large.render("Model: Fine-tuned", True, COLOURS["green"])
        else:
            model_status = font_large.render("Model: Pretrained", True, COLOURS["yellow"])
        screen.blit(model_status, (padding, status_y))
        
        # Calibration status
        calib_status_text = "Calibration: "
        calib_list = []
        if calib_affine is not None:
            calib_list.append("AFFINE")
        if calib_poly2 is not None:
            calib_list.append("POLY2")
        
        if calib_list:
            calib_status_text += ", ".join(calib_list)
            calib_color = COLOURS["green"]
        else:
            calib_status_text += "none"
            calib_color = COLOURS["red"]
        calib_status = font_large.render(calib_status_text, True, calib_color)
        screen.blit(calib_status, (padding, status_y - font_large.get_height() - padding))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
