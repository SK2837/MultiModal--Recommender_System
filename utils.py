"""
Utility functions for Multi-Modal Recommendation System
"""
import os
import pickle
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from datetime import datetime

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def save_pickle(obj: Any, filepath: Path):
    """Save object to pickle file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to {filepath}")

def load_pickle(filepath: Path) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_json(obj: Any, filepath: Path):
    """Save object to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"Saved to {filepath}")

def load_json(filepath: Path) -> Any:
    """Load object from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_embeddings(embeddings: np.ndarray, ids: List[str], filepath: Path):
    """Save embeddings with their IDs."""
    data = {
        'embeddings': embeddings,
        'ids': ids
    }
    save_pickle(data, filepath)

def load_embeddings(filepath: Path) -> Tuple[np.ndarray, List[str]]:
    """Load embeddings and their IDs."""
    data = load_pickle(filepath)
    return data['embeddings'], data['ids']

def save_model(model: torch.nn.Module, filepath: Path, metadata: Optional[Dict] = None):
    """Save PyTorch model with metadata."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat()
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(model: torch.nn.Module, filepath: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """Load PyTorch model and return model with metadata."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract metadata
    metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    
    print(f"Model loaded from {filepath}")
    return model, metadata

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_user_item_matrix(ratings_df: pd.DataFrame, 
                            user_col: str = 'userId',
                            item_col: str = 'movieId',
                            rating_col: str = 'rating') -> pd.DataFrame:
    """Create user-item rating matrix from ratings dataframe."""
    return ratings_df.pivot(index=user_col, columns=item_col, values=rating_col)

def train_test_split_by_user(ratings_df: pd.DataFrame,
                              test_ratio: float = 0.2,
                              val_ratio: float = 0.1,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ratings by user (each user has ratings in all splits)."""
    np.random.seed(random_state)
    
    train_list, val_list, test_list = [], [], []
    
    for user_id, user_data in ratings_df.groupby('userId'):
        user_data = user_data.sample(frac=1, random_state=random_state)  # Shuffle
        n = len(user_data)
        
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val
        
        train_list.append(user_data.iloc[:n_train])
        val_list.append(user_data.iloc[n_train:n_train+n_val])
        test_list.append(user_data.iloc[n_train+n_val:])
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    return train_df, val_df, test_df

def normalize_embeddings(embeddings: np.ndarray, method: str = 'l2') -> np.ndarray:
    """Normalize embeddings."""
    if method == 'l2':
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    elif method == 'standard':
        mean = embeddings.mean(axis=0)
        std = embeddings.std(axis=0)
        std = np.where(std == 0, 1, std)
        return (embeddings - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics dictionary."""
    if prefix:
        print(f"\n{prefix}")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    print("-" * 50)

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name: str):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"

if __name__ == "__main__":
    # Test utilities
    logger = setup_logging()
    logger.info("Utilities module loaded successfully")
    
    # Test seed setting
    set_seed(42)
    print("Random seed set to 42")
    
    # Test timestamp
    print(f"Current timestamp: {get_timestamp()}")
