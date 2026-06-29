from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    data_base_path: str = "datahome"
    
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 50
    train_ratio: float = 0.8
    
    device: str = "auto"
    
    checkpoint_dir: str = "models"
    model_name: str = "char_segment_1d_unet"
    
    log_interval: int = 1
    
    lr_scheduler_type: str = "plateau"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    lr_scheduler_min_lr: float = 1e-6
    
    warmup_epochs: int = 3
    
    num_workers: int = 4
    use_amp: bool = True
    
    split_file: str = "ai_model/data/dataset_split.json"
    
    seed: int = 42
    
    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.use_amp = False