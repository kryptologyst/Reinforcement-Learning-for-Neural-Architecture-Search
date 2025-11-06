"""
Configuration and utility modules for the RL NAS project.

This module provides configuration management, logging, checkpointing,
and other utility functions.
"""

from typing import Dict, Any, Optional, List, Union
import yaml
import json
import os
import logging
import torch
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
try:
    import wandb
except ImportError:
    # Fallback: create a dummy wandb
    class DummyWandb:
        def init(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass
        def finish(self):
            pass
        class run:
            pass
    wandb = DummyWandb()
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        # Fallback: create a dummy SummaryWriter
        class SummaryWriter:
            def __init__(self, *args, **kwargs):
                pass
            def add_scalar(self, *args, **kwargs):
                pass
            def add_histogram(self, *args, **kwargs):
                pass
            def add_image(self, *args, **kwargs):
                pass
            def close(self):
                pass
import pickle
from datetime import datetime


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    algorithm: str = "ppo"
    environment: str = "nas"
    total_timesteps: int = 100000
    eval_freq: int = 1000
    save_freq: int = 5000
    log_freq: int = 100
    
    # Environment specific
    env_config: Dict[str, Any] = None
    
    # Algorithm specific
    algorithm_config: Dict[str, Any] = None
    
    # Training specific
    device: str = "cpu"
    seed: int = 42
    num_envs: int = 1
    
    # Logging
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_dir: str = "./logs"
    
    def __post_init__(self):
        if self.env_config is None:
            self.env_config = {}
        if self.algorithm_config is None:
            self.algorithm_config = {}


class ConfigManager:
    """Manages configuration loading and saving."""
    
    @staticmethod
    def load_config(config_path: str) -> TrainingConfig:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return TrainingConfig(**config_dict)
    
    @staticmethod
    def save_config(config: TrainingConfig, config_path: str):
        """Save configuration to YAML or JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                yaml.dump(config_dict, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")


class Logger:
    """Unified logging interface for the project."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup basic logging
        self._setup_basic_logging()
        
        # Setup TensorBoard
        if config.use_tensorboard:
            self._setup_tensorboard()
        
        # Setup Weights & Biases
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_basic_logging(self):
        """Setup basic Python logging."""
        log_file = self.log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        tb_dir = self.log_dir / "tensorboard"
        tb_dir.mkdir(exist_ok=True)
        self.tb_writer = SummaryWriter(str(tb_dir))
    
    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        wandb.init(
            project="rl-nas",
            config=asdict(self.config),
            dir=str(self.log_dir)
        )
    
    def log_scalar(self, key: str, value: float, step: int):
        """Log scalar value."""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.add_scalar(key, value, step)
        
        if hasattr(self, 'wandb') and wandb.run is not None:
            wandb.log({key: value}, step=step)
        
        self.logger.info(f"Step {step}: {key} = {value:.4f}")
    
    def log_histogram(self, key: str, values: np.ndarray, step: int):
        """Log histogram."""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.add_histogram(key, values, step)
        
        if hasattr(self, 'wandb') and wandb.run is not None:
            wandb.log({key: wandb.Histogram(values)}, step=step)
    
    def log_image(self, key: str, image: np.ndarray, step: int):
        """Log image."""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.add_image(key, image, step)
        
        if hasattr(self, 'wandb') and wandb.run is not None:
            wandb.log({key: wandb.Image(image)}, step=step)
    
    def log_dict(self, metrics: Dict[str, Any], step: int):
        """Log dictionary of metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(key, value, step)
            elif isinstance(value, np.ndarray):
                self.log_histogram(key, value, step)
    
    def close(self):
        """Close all logging handlers."""
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
        
        if hasattr(self, 'wandb') and wandb.run is not None:
            wandb.finish()


class CheckpointManager:
    """Manages model checkpointing and loading."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> str:
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{step}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {}
        }
        
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint file."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pth"))
        
        if not checkpoint_files:
            return None
        
        # Sort by step number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return str(checkpoint_files[-1])


class MetricsTracker:
    """Tracks and computes training metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.steps = []
    
    def add_metric(self, key: str, value: float, step: int):
        """Add a metric value."""
        if key not in self.metrics:
            self.metrics[key] = []
        
        self.metrics[key].append(value)
        self.steps.append(step)
        
        # Keep only recent values
        if len(self.metrics[key]) > self.window_size:
            self.metrics[key] = self.metrics[key][-self.window_size:]
    
    def get_mean(self, key: str) -> float:
        """Get mean of recent values for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return np.mean(self.metrics[key])
    
    def get_std(self, key: str) -> float:
        """Get standard deviation of recent values for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return np.std(self.metrics[key])
    
    def get_latest(self, key: str) -> float:
        """Get latest value for a metric."""
        if key not in self.metrics or not self.metrics[key]:
            return 0.0
        return self.metrics[key][-1]
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for key in self.metrics:
            summary[key] = {
                'mean': self.get_mean(key),
                'std': self.get_std(key),
                'latest': self.get_latest(key),
                'count': len(self.metrics[key])
            }
        return summary


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should stop early."""
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == 'max':
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get available device (CPU or CUDA)."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from file."""
    with open(filepath, 'r') as f:
        return json.load(f)
