"""
PyTorch Dataset classes for recommendation system
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from pathlib import Path

import config
from utils import load_embeddings, load_pickle


class RatingsDataset(Dataset):
    """Dataset for user-item ratings with multi-modal features."""
    
    def __init__(self,
                 ratings_df: pd.DataFrame,
                 item_features: Optional[np.ndarray] = None,
                 use_features: bool = True):
        """
        Args:
            ratings_df: DataFrame with columns [user_idx, movie_idx, rating]
            item_features: Pre-computed item features [num_items, feature_dim]
            use_features: Whether to return item features
        """
        self.ratings_df = ratings_df.reset_index(drop=True)
        self.item_features = item_features
        self.use_features = use_features
        
        # Convert to numpy for faster indexing
        self.user_ids = self.ratings_df['user_idx'].values
        self.item_ids = self.ratings_df['movie_idx'].values
        self.ratings = self.ratings_df['rating'].values.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.ratings_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        user_id = self.user_ids[idx]
        item_id = self.item_ids[idx]
        rating = self.ratings[idx]
        
        if self.use_features and self.item_features is not None:
            item_feature = self.item_features[item_id]
            return (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(item_id, dtype=torch.long),
                torch.tensor(item_feature, dtype=torch.float32),
                torch.tensor(rating, dtype=torch.float32)
            )
        else:
            return (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(item_id, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float32)
            )


def load_multimodal_features() -> np.ndarray:
    """Load pre-computed multi-modal item features."""
    embeddings_path = config.EMBEDDINGS_DIR / "multimodal_embeddings.pkl"
    
    if embeddings_path.exists():
        embeddings, movie_ids = load_embeddings(embeddings_path)
        return embeddings
    else:
        raise FileNotFoundError(
            f"Multi-modal embeddings not found at {embeddings_path}. "
            "Please run multimodal_fusion.py first."
        )


def create_dataloaders(
    batch_size: int = None,
    num_workers: int = None,
    use_features: bool = True
) -> Tuple:
    """Create train, validation, and test dataloaders."""
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    
    # Load ratings
    train_df = pd.read_csv(config.PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(config.PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(config.PROCESSED_DATA_DIR / "test.csv")
    
    # Load item features if needed
    item_features = None
    if use_features:
        item_features = load_multimodal_features()
    
    # Create datasets
    train_dataset = RatingsDataset(train_df, item_features, use_features)
    val_dataset = RatingsDataset(val_df, item_features, use_features)
    test_dataset = RatingsDataset(test_df, item_features, use_features)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader, item_features


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")
    
    train_loader, val_loader, test_loader, item_features = create_dataloaders(
        batch_size=32,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for batch in train_loader:
        if len(batch) == 4:
            user_ids, item_ids, item_features, ratings = batch
            print(f"\nBatch shapes:")
            print(f"  User IDs: {user_ids.shape}")
            print(f"  Item IDs: {item_ids.shape}")
            print(f"  Item features: {item_features.shape}")
            print(f"  Ratings: {ratings.shape}")
        break
    
    print("\n✓ Dataset test passed!")
