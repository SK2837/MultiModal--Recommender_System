"""
Multi-Modal Feature Fusion Module
Combines text and visual embeddings into unified representations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

import config
from utils import (setup_logging, load_embeddings, save_embeddings, 
                  normalize_embeddings, set_seed)

logger = setup_logging(log_file=config.LOGS_DIR / "multimodal_fusion.log")


class MultiModalFusion:
    """Combine text and visual features into multi-modal representations."""
    
    def __init__(self, fusion_method: str = None):
        self.fusion_method = fusion_method or config.FUSION_METHOD
        logger.info(f"Using fusion method: {self.fusion_method}")
    
    def concatenate_fusion(self, text_emb: np.ndarray, visual_emb: np.ndarray) -> np.ndarray:
        """Simple concatenation of text and visual embeddings."""
        logger.info("Concatenating text and visual embeddings...")
        
        assert text_emb.shape[0] == visual_emb.shape[0], \
            f"Mismatch in number of samples: {text_emb.shape[0]} vs {visual_emb.shape[0]}"
        
        fused = np.concatenate([text_emb, visual_emb], axis=1)
        
        logger.info(f"Fused shape: {fused.shape}")
        return fused
    
    def weighted_fusion(self, text_emb: np.ndarray, visual_emb: np.ndarray,
                       text_weight: float = 0.5) -> np.ndarray:
        """Weighted combination (requires same dimensions)."""
        logger.info(f"Weighted fusion: text={text_weight}, visual={1-text_weight}")
        
        # This requires embeddings to have same dimensions
        # We'd need to project them to common space first
        raise NotImplementedError("Weighted fusion requires dimension alignment")
    
    def fuse(self, text_emb: np.ndarray, visual_emb: np.ndarray) -> np.ndarray:
        """Fuse embeddings based on configured method."""
        if self.fusion_method == "concatenate":
            return self.concatenate_fusion(text_emb, visual_emb)
        elif self.fusion_method == "weighted":
            return self.weighted_fusion(text_emb, visual_emb)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")


def load_features() -> Tuple[np.ndarray, np.ndarray, list]:
    """Load text and visual features from disk."""
    logger.info("Loading pre-computed features...")
    
    # Load text embeddings
    text_path = config.EMBEDDINGS_DIR / "text_embeddings.pkl"
    text_emb, text_ids = load_embeddings(text_path)
    logger.info(f"Loaded text embeddings: {text_emb.shape}")
    
    # Load visual embeddings
    visual_path = config.EMBEDDINGS_DIR / "visual_embeddings.pkl"
    visual_emb, visual_ids = load_embeddings(visual_path)
    logger.info(f"Loaded visual embeddings: {visual_emb.shape}")
    
    # Verify IDs match
    assert text_ids == visual_ids, "Mismatch in movie IDs between text and visual embeddings"
    
    return text_emb, visual_emb, text_ids


def main():
    """Main multi-modal fusion pipeline."""
    logger.info("=" * 80)
    logger.info("Multi-Modal Feature Fusion")
    logger.info("=" * 80)
    
    set_seed(config.RANDOM_SEED)
    
    # Load features
    text_emb, visual_emb, movie_ids = load_features()
    
    # Fuse features
    fusion = MultiModalFusion()
    multimodal_emb = fusion.fuse(text_emb, visual_emb)
    
    # Normalize fused embeddings
    logger.info("Normalizing fused embeddings...")
    multimodal_emb = normalize_embeddings(multimodal_emb, method='l2')
    
    # Print statistics
    logger.info("\nMulti-Modal Embedding Statistics:")
    logger.info(f"  Shape: {multimodal_emb.shape}")
    logger.info(f"  Mean: {multimodal_emb.mean():.4f}")
    logger.info(f"  Std: {multimodal_emb.std():.4f}")
    logger.info(f"  Min: {multimodal_emb.min():.4f}")
    logger.info(f"  Max: {multimodal_emb.max():.4f}")
    
    # Save fused embeddings
    save_path = config.EMBEDDINGS_DIR / "multimodal_embeddings.pkl"
    save_embeddings(multimodal_emb, movie_ids, save_path)
    logger.info(f"Multi-modal embeddings saved to {save_path}")
    
    logger.info("=" * 80)
    logger.info("Multi-Modal Fusion Complete!")
    logger.info("=" * 80)
    
    return multimodal_emb, movie_ids


if __name__ == "__main__":
    main()
