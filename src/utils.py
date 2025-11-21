import ast
import datetime
import inspect
import itertools
import json
import random
from configparser import ConfigParser
from pathlib import Path
from typing import List
from typing import Sequence, Dict, Any
from ray.tune import ExperimentAnalysis
import matplotlib as mpl
import matplotlib.pyplot as plt
import mediapipe as mp
from scipy.interpolate import griddata
from tqdm.auto import tqdm
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from models import (
    GazeDataModule,
    SingleModel,
    EyesModel,
    FullModel,
)
import os
import cv2
from typing import Tuple
from models import GazeDataset

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def get_config(path="config.ini", comment_char=";"):
    config_file = ConfigParser(inline_comment_prefixes=comment_char)
    config_file.read(path)
    print(config_file)
    config_default = config_file["DEFAULT"]
    config_colours = config_file["COLOURS"]
    config_eyetracker = config_file["EYETRACKER"]
    config_tf = config_file["TF"]

    def safe_literal_eval(value):
        """Safely evaluate config value, handling strings and other literals."""
        value = value.strip()
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If literal_eval fails, treat as string (remove quotes if present)
            if value.startswith('"') and value.endswith('"'):
                return value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                return value[1:-1]
            else:
                return value  # Return as-is (unquoted string)

    settings = {key: safe_literal_eval(config_default[key]) for key in config_default}
    colours = {key: safe_literal_eval(config_colours[key]) for key in config_colours}
    eyetracker = {
        key: safe_literal_eval(config_eyetracker[key]) for key in config_eyetracker
    }
    tf = {key: safe_literal_eval(config_tf[key]) for key in config_tf}

    return settings, colours, eyetracker, tf

# Crop the right eye region
def getRightEye(image, landmarks, eye_center):
    eye_top = int(landmarks[257].y * image.shape[0])
    eye_left = int(landmarks[362].x * image.shape[1])
    eye_bottom = int(landmarks[374].y * image.shape[0])
    eye_right = int(landmarks[263].x * image.shape[1])
    eye_width = eye_right - eye_left

    top = eye_center[1] - int(eye_width / 2)
    bottom = eye_center[1] + int(eye_width / 2)

    eye_img = image[
              top:bottom,
              eye_left:eye_right
              ]
    return eye_img


def getLeftEye(image, landmarks, eye_center):
    eye_top = int(landmarks[159].y * image.shape[0])
    eye_left = int(landmarks[33].x * image.shape[1])
    eye_bottom = int(landmarks[145].y * image.shape[0])
    eye_right = int(landmarks[133].x * image.shape[1])
    eye_width = eye_right - eye_left

    top = eye_center[1] - int(eye_width / 2)
    bottom = eye_center[1] + int(eye_width / 2)

    eye_img = image[
              top:bottom,
              eye_left:eye_right
              ]
    return eye_img


def bgr_to_rgb(img):
    """Convert from opencv BGR to RGB"""
    return img[..., ::-1].copy()


def clamp_value(x, max_value):
    """Restrict values to a range"""
    if x < 0:
        return 0
    if x > max_value:
        return max_value
    return x


def plot_region_map(path, region_map, map_scale, cmap="inferno"):
    """Create plot of number of data samples at each screen coordinate"""
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(
        np.transpose(region_map).repeat(map_scale, axis=0).repeat(map_scale, axis=1),
        interpolation="bicubic",
        cmap=cmap,
    )
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    plt.colorbar(
        im, ticks=np.arange(np.min(region_map), np.max(region_map) + 1), cax=cax
    )
    ax.set_title("Number of samples at each screen region")
    plt.savefig(path)


def get_calibration_zones(w, h, target_radius):
    xs = (0 + target_radius, w / 2, w - target_radius)
    ys = (0 + target_radius, h / 2, h - target_radius)
    zones = list(itertools.product(xs, ys))
    random.shuffle(zones)
    return zones


def get_undersampled_region(region_map, map_scale):
    min_coords = np.where(region_map == np.min(region_map))
    idx = random.randint(0, len(min_coords[0]) - 1)
    return (min_coords[0][idx] * map_scale, min_coords[1][idx] * map_scale)

def _build_datamodule( data_dir: str | Path, img_types: Sequence[str], cfg: Dict[str, Any]) -> GazeDataModule:
    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    return GazeDataModule(
        data_dir=data_dir,
        batch_size=int(cfg.get("bs", 128)),
        img_types=img_types,
        seed=int(cfg.get("seed", 87)),
    )


def _build_model(cfg: Dict[str, Any], img_types: Sequence[str]):
    if len(img_types) == 1:
        cls = SingleModel
    elif set(img_types) == {"l_eye", "r_eye"}:
        cls = EyesModel
    else:
        cls = FullModel

    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in cfg.items() if k in sig.parameters}

    if len(img_types) == 1:
        return cls(img_type=img_types[0], **valid)
    return cls(**valid)


def _train_model(
    cfg: Dict[str, Any],
    img_types: Sequence[str],
    data_dir: Path | str | None = None,
    num_epochs: int = 20,
    tune_report: bool = False,
):
    pl.seed_everything(int(cfg.get("seed", 87)), workers=True)
    data_dir = Path(data_dir or "data")

    dm = _build_datamodule(data_dir, img_types, cfg)
    model = _build_model(cfg, img_types)

    callbacks: List[Any] = []
    if tune_report:
        callbacks.append(
            TuneReportCheckpointCallback(
                metrics={"loss": "val_loss", "mae": "val_mae"}, on="validation_end"
            )
        )

    tb_logger = TensorBoardLogger(
        save_dir="tb_logs",
        name="gaze",
        version=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps = 10,
    )

    trainer.fit(model, dm)

    if not tune_report:
        metrics = trainer.callback_metrics
        print(f"\nFinished – val_loss: {metrics['val_loss']:.3f}, val_mae: {metrics['val_mae']:.3f}")
        return metrics


def train_single(cfg: Dict[str, Any], img_types: Sequence[str], num_epochs: int = 15, data_dir: Path | str = "data"):
    assert len(img_types) == 1, "SingleModel expects exactly one img_type"
    _train_model(cfg, img_types, data_dir, num_epochs, tune_report=False)


def train_eyes(cfg: Dict[str, Any], img_types: Sequence[str] = ("l_eye", "r_eye"), num_epochs: int = 15, data_dir: Path | str = "data"):
    assert set(img_types) == {"l_eye", "r_eye"}, "EyesModel needs both eyes"
    _train_model(cfg, img_types, data_dir, num_epochs, tune_report=False)


def train_full(cfg: Dict[str, Any], img_types: Sequence[str] = ("face_aligned", "l_eye", "r_eye", "head_pos", "head_angle"), num_epochs: int = 20, data_dir: Path | str = "data"):
    _train_model(cfg, img_types, data_dir, num_epochs, tune_report=False)


def tune_asha(
    search_space: Dict[str, Any],
    train_func: str,  # "single", "eyes", or "full"
    name: str,
    img_types: Sequence[str],
    num_samples: int = 10,
    num_epochs: int = 20,
    data_dir: Path | str = "data",
    seed: int = 1,
):

    train_map = {
        "single": train_single,
        "eyes": train_eyes,
        "full": train_full,
    }
    assert train_func in train_map, "train_func must be 'single', 'eyes', or 'full'"

    def _tune_wrapper(cfg):
        _train_model(cfg, img_types, data_dir, num_epochs, tune_report=True)

    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    def _short_name(trial):
        return str(trial.trial_id)

    analysis = tune.run(
        _tune_wrapper,
        resources_per_trial={"cpu": 8, "gpu": 1 if torch.cuda.is_available() else 0},
        metric="loss",
        mode="min",
        num_samples=num_samples,
        config=search_space,
        scheduler=scheduler,
        name=f"{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        storage_path=Path.cwd() / "logs",
        log_to_file=True,
        fail_fast=True,
        trial_dirname_creator=_short_name,
        verbose=3,
        raise_on_failed_trial=False,
    )

    print("Best hyperparameters: ", analysis.best_config)
    return analysis


def latest_tune_dir(parent: Path) -> Path:
    subdirs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("tune_")]
    if not subdirs:
        raise FileNotFoundError(f"No tune_* folder in {parent}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _ensure_fig_dir() -> Path:
    fig_dir = Path.cwd() / "figs"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


def get_tune_results(analysis):
    if analysis.best_checkpoint:
        print("Directory:", analysis.best_checkpoint)
    print("Loss:", round(analysis.best_result["loss"], 3))
    print("Pixel error:", round(np.sqrt(analysis.best_result["loss"]), 3))
    print("Hyper-params: ", json.dumps(analysis.best_config, indent=2))


def get_best_results(path):
    analysis = ExperimentAnalysis(path, default_metric="loss", default_mode="min")
    df = analysis.dataframe()
    df.sort_values("loss", inplace=True)
    best = df.head(1)

    print(f"\n--- Best of '{path}' ---\n")
    print(f"Directory: {best['logdir'].values[0]}")
    print(f"Loss: {round(best['loss'].values[0],2)}")
    print(f"Pixel error: {round(np.sqrt(best['loss'].values[0]),2)}")

    hyperparams = best.filter(like="config", axis=1)
    print("Hyperparameters...")
    for column in hyperparams:
        name = column.split("/")[1]
        value = hyperparams[column].values[0]
        print(f"- {name}: {value}")

    return analysis.get_best_config()


def save_model(model, config, path_weights, path_config):
    """Save trained torch weights with config"""

    Path(path_weights).parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path_weights)

    with open(path_config, "w") as fp:
        json.dump(config, fp, indent=4)


def plot_parallel_param_loss(
    analysis,
    cols=("bs", "lr", "channels", "hidden"),
    cmap_name="plasma",
    save_path="media/images/1_face_explore_parallel.png",
):
    df   = analysis.dataframe()
    cols = [c for c in cols if f"config/{c}" in df]
    if not cols:
        raise ValueError("None of the requested columns found in analysis")

    pc = df[[f"config/{c}" for c in cols] + ["loss"]].copy()
    pc.columns = list(cols) + ["loss"]      # rename for easy access

    tick_labels = {}
    for c in cols:
        series = pc[c]

        # categorical?
        if isinstance(series.iloc[0], (tuple, str)):
            cats = series.astype(str).unique().tolist()
            tick_labels[c] = cats
            codes = series.astype(str).apply(cats.index).astype(float)

            if len(cats) > 1:
                pc[c] = codes / (len(cats) - 1)
            else:
                pc[c] = 0.5
        else:
            x = series.astype(float)
            tick_labels[c] = np.linspace(x.min(), x.max(), 5)
            rng = x.max() - x.min()
            pc[c] = (x - x.min()) / (rng + 1e-9)

    loss = pc["loss"].astype(float)
    norm = mpl.colors.Normalize(loss.min(), loss.max())
    cmap = mpl.cm.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(2.6*len(cols), 3),
                           constrained_layout=True)

    for _, row in pc.iterrows():
        ax.plot(cols, row[cols],
                color=cmap(norm(row["loss"])), alpha=0.7, linewidth=1)

    ax.set_ylabel("normalised 0-1")
    ax.set_title("Hyper-parameter exploration – colour = val_loss")
    ax.grid(True, axis="y", alpha=0.3)

    # put category labels / numeric ticks
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30)

    # numeric y-tick labels only once (left)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

    # annotate categorical axes
    for i, c in enumerate(cols):
        if isinstance(pc[c].iloc[0], (tuple, str)):
            cats = tick_labels[c]
            ys   = np.linspace(0, 1, len(cats))
            for y, txt in zip(ys, cats):
                ax.text(i, y, txt, va="center", ha="right",
                        fontsize=7, color="black",
                        transform=ax.transData)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.01).set_label("val_loss")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.show()
    return Path(save_path)


def plot_asha_param_grid(analysis,
                         params=("bs","lr","channels","hidden"),
                         save_path="media/images/1_face_explore_scatter.png",
                         cmap_name="plasma"):

    df   = analysis.dataframe()
    loss = df["loss"].astype(float)
    norm = mpl.colors.Normalize(loss.min(), loss.max())
    cmap = mpl.cm.get_cmap(cmap_name)

    # ★ enable smarter layout
    fig, axes = plt.subplots(1, len(params),
                             figsize=(2.1*len(params), 3.5),
                             sharey=True,
                             constrained_layout=True)

    for ax, p in zip(axes, params):
        col = f"config/{p}"
        if col not in df: ax.set_visible(False); continue

        x_raw = df[col]
        if isinstance(x_raw.iloc[0], (tuple, str)):
            cats = x_raw.astype(str).unique().tolist()
            x = x_raw.astype(str).apply(cats.index)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats, rotation=60, fontsize=7)
        else:
            x = x_raw.astype(float)

        ax.scatter(x, loss, c=cmap(norm(loss)),
                   s=20, edgecolor="k", linewidth=.3, alpha=.8)
        ax.set_xlabel(p, fontsize=8, rotation=25)
        ax.grid(alpha=.3);  ax.spines[["top","right"]].set_visible(False)

    axes[0].set_ylabel("val_loss"); axes[0].invert_yaxis()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120); plt.show()
    return Path(save_path)


def plot_screen_errors(x, y, z, path_plot=None, path_errors=None):
    """Plot prediction errors over screen space"""
    # create grid
    xi = np.arange(0, 1920, 1)
    yi = np.arange(0, 1080, 1)
    xi, yi = np.meshgrid(xi, yi)

    # interpolate
    zi = griddata((x, y), z, (xi, yi), method="nearest")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.contourf(xi, yi, zi)
    cbar = plt.colorbar()

    cbar.ax.set_ylabel("Error (px)")
    plt.xlabel("Screen X")
    plt.ylabel("Screen Y")
    plt.gca().invert_yaxis()

    if path_plot is not None:
        plt.savefig(path_plot, dpi=100)

    if path_errors is not None:
        np.save(path_errors, zi.T)

    plt.show()

    # Error histogram
    plt.hist(z, edgecolor="black")
    plt.xlabel("Error (px)")
    plt.ylabel("Count")
    plt.show()

    return zi.T


def predict_screen_errors(*img_types, path_model, path_config, path_plot=None, path_errors=None, steps=10):
    with open(path_config) as fp:
        cfg = json.load(fp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = _build_model(cfg, img_types).to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()

    ds = GazeDataset(Path.cwd()/"data", img_types, augment=False)

    xs, ys, errs = [], [], []
    for i, sample in tqdm(enumerate(ds), total=len(ds)):
        if i % steps: continue
        imgs = [sample[t].unsqueeze(0).to(device) for t in img_types]
        target = sample["targets"].to(device)
        with torch.no_grad():
            pred  = model(*imgs)[0]
        dist = torch.linalg.vector_norm(pred - target)
        xs.append(float(target[0])); ys.append(float(target[1])); errs.append(float(dist))

    print(f"Average error: {np.mean(errs):.2f}px over {len(errs)} samples")
    plot_screen_errors(xs, ys, errs, path_plot, path_errors)


def plot_trajectory(screenshot: np.ndarray,
                    points: Sequence[Tuple[int,int]],
                    out_dir: str = "media/trajectories") -> str:

    os.makedirs(out_dir, exist_ok=True)
    img = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(img.shape[1]/100, img.shape[0]/100), dpi=100)
    plt.imshow(img)

    if points:
        xs, ys = zip(*points)
        # draw the trajectory line in light gray
        plt.plot(xs, ys, linewidth=2, color='lightgray')
        # draw intermediate points in red
        plt.scatter(xs, ys, s=16, color='red', zorder=2)
        # first point in green
        plt.scatter([xs[0]], [ys[0]], s=64, color='green', marker='o', zorder=3, label='Start')
        # last point in blue
        plt.scatter([xs[-1]], [ys[-1]], s=64, color='blue', marker='o', zorder=3, label='End')
        # optional legend
        plt.legend(loc='upper right')

    plt.axis('off')

    filename = f"trajectory_{len(os.listdir(out_dir))+1:03d}.png"
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return out_path


