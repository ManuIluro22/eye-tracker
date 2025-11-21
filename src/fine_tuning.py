"""
Fine-tuning script for the pretrained eye-tracking model.

This script loads the pretrained model and fine-tunes it on user-collected data
from the data collection mode. The fine-tuned model is saved separately so the
original pretrained model remains intact.
"""

import sys
from pathlib import Path
import json
import datetime
import csv
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from utils import get_config, _build_model, _build_datamodule
from models import FullModel

# Paths
PRETRAINED_MODEL_PATH = Path("src/trained_models/full/eyetracking_model_pretrained.pt")
# Fallback to regular model if pretrained backup doesn't exist
if not PRETRAINED_MODEL_PATH.exists():
    PRETRAINED_MODEL_PATH = Path("src/trained_models/full/eyetracking_model.pt")
PRETRAINED_CFG_JSON = Path("src/trained_models/full/eyetracking_config.json")
FINE_TUNED_MODEL_PATH = Path("src/trained_models/full/fine_tuned_eyetracking_model.pt")
FINE_TUNED_CFG_JSON = Path("src/trained_models/full/fine_tuned_eyetracking_config.json")
DATA_DIR = Path("src/data")  # Match project structure from README
FINE_TUNE_LOGS_DIR = Path("src/trained_models/full/fine_tune_logs")  # Training logs directory

# Fine-tuning hyperparameters (can be adjusted)
FINE_TUNE_CONFIG = {
    "lr": 1e-4,  # Higher learning rate for fine-tuning (with 3000+ samples)
    "bs": 32,    # Batch size
    "num_epochs": 15,  # More epochs for better convergence
    "seed": 87,
    "lr_scheduler": "reduce_on_plateau",  # Learning rate scheduler type
    "lr_patience": 5,  # Epochs to wait before reducing LR
    "lr_factor": 0.5,  # Factor to reduce LR by
    "lr_min": 1e-6,  # Minimum learning rate
}


def load_pretrained_model(cfg_path: Path, weights_path: Path, device: torch.device):
    """Load pretrained model architecture and weights."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    
    # Build model with same architecture
    img_types = ["face_aligned", "l_eye", "r_eye", "head_pos", "head_angle"]
    
    # Try to load as Lightning checkpoint first
    if weights_path.suffix == ".ckpt":
        try:
            model = FullModel.load_from_checkpoint(str(weights_path)).to(device)
            print(f"[+] Loaded pretrained model from Lightning checkpoint: {weights_path}")
            return model, cfg
        except Exception as e:
            print(f"[!] Failed to load as Lightning checkpoint: {e}")
            print("[*] Trying to load as PyTorch state dict...")
    
    # Load as PyTorch state dict
    model = _build_model(cfg, img_types).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Handle both direct state_dict and wrapped in 'state_dict' key
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    
    model.load_state_dict(state, strict=True)
    print(f"[+] Loaded pretrained model from PyTorch weights: {weights_path}")
    return model, cfg


class MetricsTracker(Callback):
    """Callback to track and save training metrics over epochs."""
    
    def __init__(self, log_file: Path):
        super().__init__()
        self.log_file = log_file
        self.metrics_history = []
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def on_validation_end(self, trainer, pl_module):
        """Save metrics after each validation epoch."""
        metrics = trainer.callback_metrics
        
        # Extract epoch metrics
        epoch_metrics = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "train_loss": float(metrics.get("train_loss", 0.0)),
            "val_loss": float(metrics.get("val_loss", 0.0)),
            "train_mae": float(metrics.get("train_mae", 0.0)),
            "val_mae": float(metrics.get("val_mae", 0.0)),
            "train_rmse": float(metrics.get("train_rmse", 0.0)),
            "val_rmse": float(metrics.get("val_rmse", 0.0)),
        }
        
        self.metrics_history.append(epoch_metrics)
        
        # Save to JSON after each epoch
        with open(self.log_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Also save as CSV for easy viewing
        csv_file = self.log_file.with_suffix(".csv")
        if len(self.metrics_history) == 1:
            # Write header
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=epoch_metrics.keys())
                writer.writeheader()
                writer.writerow(epoch_metrics)
        else:
            # Append row
            with open(csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=epoch_metrics.keys())
                writer.writerow(epoch_metrics)
    
    def on_train_end(self, trainer, pl_module):
        """Print summary at end of training."""
        if self.metrics_history:
            print("\n[*] Training Metrics Summary:")
            print(f"    Total epochs: {len(self.metrics_history)}")
            print(f"    Best val_loss: {min(m['val_loss'] for m in self.metrics_history):.4f}")
            print(f"    Best val_mae: {min(m['val_mae'] for m in self.metrics_history):.4f}")
            print(f"    Final val_loss: {self.metrics_history[-1]['val_loss']:.4f}")
            print(f"    Final val_mae: {self.metrics_history[-1]['val_mae']:.4f}")
            print(f"\n[*] Metrics saved to:")
            print(f"    JSON: {self.log_file}")
            print(f"    CSV: {self.log_file.with_suffix('.csv')}")


def fine_tune_model(
    pretrained_model_path: Path,
    pretrained_cfg_path: Path,
    data_dir: Path,
    output_model_path: Path,
    output_cfg_path: Path,
    fine_tune_cfg: dict,
):
    """
    Fine-tune the pretrained model on user-collected data.
    
    Args:
        pretrained_model_path: Path to pretrained model weights
        pretrained_cfg_path: Path to pretrained model config
        data_dir: Directory containing collected data
        output_model_path: Where to save fine-tuned model
        output_cfg_path: Where to save fine-tuned config
        fine_tune_cfg: Fine-tuning hyperparameters
    """
    # Check if data exists
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Please run data collection mode first (option 2 in data_collection.py)"
        )
    
    positions_file = data_dir / "positions.csv"
    if not positions_file.exists():
        raise FileNotFoundError(
            f"positions.csv not found in {data_dir}\n"
            "Please run data collection mode first to collect data."
        )
    
    # Check if we have enough data
    import pandas as pd
    df = pd.read_csv(positions_file)
    if len(df) < 50:
        print(f"[!] Warning: Only {len(df)} samples found. Fine-tuning may not be effective.")
        print("[!] Consider collecting more data (recommended: 500+ samples)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    # Load pretrained model
    model, pretrained_cfg = load_pretrained_model(
        pretrained_cfg_path, pretrained_model_path, device
    )
    
    # Create fine-tuned config (merge pretrained config with fine-tune settings)
    fine_tuned_cfg = pretrained_cfg.copy()
    fine_tuned_cfg.update(fine_tune_cfg)
    
    # Create Lightning model for training with LR scheduler support
    img_types = ["face_aligned", "l_eye", "r_eye", "head_pos", "head_angle"]
    
    # Store scheduler config in a variable for the model to access
    use_lr_scheduler = fine_tuned_cfg.get("lr_scheduler") == "reduce_on_plateau"
    lr_scheduler_config = {
        "factor": fine_tuned_cfg.get("lr_factor", 0.5),
        "patience": fine_tuned_cfg.get("lr_patience", 5),
        "min_lr": fine_tuned_cfg.get("lr_min", 1e-6),
    } if use_lr_scheduler else None
    
    # Create a custom model class that supports LR scheduling
    class FineTuneFullModel(FullModel):
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
            
            # Add learning rate scheduler if configured
            if use_lr_scheduler and lr_scheduler_config:
                scheduler = {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="min",
                        factor=lr_scheduler_config["factor"],
                        patience=lr_scheduler_config["patience"],
                        min_lr=lr_scheduler_config["min_lr"],
                        verbose=True,
                    ),
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            return optimizer
    
    lightning_model = FineTuneFullModel(
        lr=fine_tuned_cfg["lr"],
        face_channels=pretrained_cfg["face_channels"],
        eye_channels=pretrained_cfg["eye_channels"],
        head_pos_channels=pretrained_cfg["head_pos_channels"],
        hidden=pretrained_cfg["hidden"],
    )
    
    # Load pretrained weights into Lightning model
    lightning_model.load_state_dict(model.state_dict(), strict=True)
    lightning_model = lightning_model.to(device)
    
    print(f"[*] Fine-tuning configuration:")
    print(f"    Learning rate: {fine_tuned_cfg['lr']}")
    if fine_tuned_cfg.get("lr_scheduler"):
        print(f"    LR scheduler: {fine_tuned_cfg['lr_scheduler']} (patience={fine_tuned_cfg.get('lr_patience', 5)}, factor={fine_tuned_cfg.get('lr_factor', 0.5)})")
    print(f"    Batch size: {fine_tuned_cfg['bs']}")
    print(f"    Epochs: {fine_tuned_cfg['num_epochs']}")
    print(f"    Data samples: {len(df)}")
    
    # Setup data module
    data_module = _build_datamodule(str(data_dir), img_types, fine_tuned_cfg)
    
    # Setup logging directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = FINE_TUNE_LOGS_DIR / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup loggers
    tb_logger = TensorBoardLogger(
        save_dir=str(FINE_TUNE_LOGS_DIR),
        name="fine_tune",
        version=timestamp,
        log_graph=True,
    )
    
    csv_logger = CSVLogger(
        save_dir=str(FINE_TUNE_LOGS_DIR),
        name="fine_tune",
        version=timestamp,
    )
    
    # Setup metrics tracker
    metrics_file = log_dir / "metrics_history.json"
    metrics_tracker = MetricsTracker(metrics_file)
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="fine_tuned-{epoch:02d}-{val_loss:.3f}",
        dirpath=str(log_dir),
    )
    
    # Setup trainer with logging
    trainer = pl.Trainer(
        max_epochs=fine_tuned_cfg["num_epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint_callback, metrics_tracker],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    print(f"[*] Logging to: {log_dir}")
    print(f"[*] TensorBoard logs: {tb_logger.log_dir}")
    print(f"[*] View with: tensorboard --logdir {FINE_TUNE_LOGS_DIR}")
    
    # Fine-tune
    print("[*] Starting fine-tuning...")
    trainer.fit(lightning_model, data_module)
    
    # Get best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"[+] Best model checkpoint: {best_model_path}")
        best_model = FullModel.load_from_checkpoint(best_model_path)
    else:
        print("[!] No checkpoint saved, using current model state")
        best_model = lightning_model
    
    # Save fine-tuned model weights (PyTorch format)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_model_path)
    print(f"[+] Saved fine-tuned model weights to {output_model_path}")
    
    # Save fine-tuned config
    # Note: This config contains the same architecture parameters as the pretrained model,
    # plus the fine-tuning hyperparameters. It can be used for both training and inference.
    with open(output_cfg_path, "w") as f:
        json.dump(fine_tuned_cfg, f, indent=4)
    print(f"[+] Saved fine-tuned config to {output_cfg_path}")
    
    # Print final metrics
    metrics = trainer.callback_metrics
    print(f"\n[+] Fine-tuning complete!")
    print(f"    Final val_loss: {metrics.get('val_loss', 'N/A'):.3f}")
    print(f"    Final val_mae: {metrics.get('val_mae', 'N/A'):.3f}")
    
    # Save training summary
    summary = {
        "timestamp": timestamp,
        "pretrained_model": str(pretrained_model_path),
        "data_samples": len(df),
        "hyperparameters": fine_tune_cfg,
        "final_metrics": {
            "val_loss": float(metrics.get("val_loss", 0.0)),
            "val_mae": float(metrics.get("val_mae", 0.0)),
            "val_rmse": float(metrics.get("val_rmse", 0.0)),
        },
        "best_checkpoint": str(best_model_path) if best_model_path else None,
        "output_model": str(output_model_path),
        "log_directory": str(log_dir),
    }
    
    summary_file = log_dir / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[*] Training summary saved to: {summary_file}")
    print(f"[*] View TensorBoard: tensorboard --logdir {FINE_TUNE_LOGS_DIR}")
    print(f"\n[*] You can now use the fine-tuned model in data_collection.py")


def main():
    """Main entry point for fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune pretrained eye-tracking model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing collected data (default: data/)",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=str(PRETRAINED_MODEL_PATH),
        help="Path to pretrained model weights",
    )
    parser.add_argument(
        "--pretrained-config",
        type=str,
        default=str(PRETRAINED_CFG_JSON),
        help="Path to pretrained model config",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(FINE_TUNED_MODEL_PATH),
        help="Output path for fine-tuned model",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=str(FINE_TUNED_CFG_JSON),
        help="Output path for fine-tuned config",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=FINE_TUNE_CONFIG["lr"],
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=FINE_TUNE_CONFIG["bs"],
        help="Batch size for fine-tuning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=FINE_TUNE_CONFIG["num_epochs"],
        help="Number of epochs for fine-tuning",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["reduce_on_plateau", "none"],
        default=FINE_TUNE_CONFIG.get("lr_scheduler", "reduce_on_plateau"),
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=FINE_TUNE_CONFIG.get("lr_patience", 5),
        help="Epochs to wait before reducing LR (for ReduceLROnPlateau)",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=FINE_TUNE_CONFIG.get("lr_factor", 0.5),
        help="Factor to reduce LR by (for ReduceLROnPlateau)",
    )
    
    args = parser.parse_args()
    
    # Update fine-tune config with CLI args
    fine_tune_cfg = FINE_TUNE_CONFIG.copy()
    fine_tune_cfg["lr"] = args.lr
    fine_tune_cfg["bs"] = args.batch_size
    fine_tune_cfg["num_epochs"] = args.epochs
    fine_tune_cfg["lr_scheduler"] = args.lr_scheduler if args.lr_scheduler != "none" else None
    fine_tune_cfg["lr_patience"] = args.lr_patience
    fine_tune_cfg["lr_factor"] = args.lr_factor
    
    try:
        fine_tune_model(
            pretrained_model_path=Path(args.pretrained_model),
            pretrained_cfg_path=Path(args.pretrained_config),
            data_dir=Path(args.data_dir),
            output_model_path=Path(args.output_model),
            output_cfg_path=Path(args.output_config),
            fine_tune_cfg=fine_tune_cfg,
        )
    except Exception as e:
        print(f"[!] Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

