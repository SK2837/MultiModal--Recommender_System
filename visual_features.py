"""
Visual Feature Extraction Module
Extracts visual embeddings from movie posters using pre-trained CNNs
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms

import config
from utils import setup_logging, save_embeddings, normalize_embeddings, set_seed

logger = setup_logging(log_file=config.LOGS_DIR / "visual_features.log")


class VisualFeatureExtractor:
    """Extract visual embeddings from images using pre-trained CNNs."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.VISUAL_MODEL_NAME
        self.device = config.DEVICE
        
        logger.info(f"Loading visual model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained model
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Remove final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif self.model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ])
        
        logger.info(f"Visual model loaded successfully")
    
    def load_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """Load and preprocess an image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            logger.debug(f"Error loading {image_path}: {e}")
            return None
    
    def create_placeholder_embedding(self) -> np.ndarray:
        """Create a placeholder embedding for missing images."""
        # Use zeros as placeholder
        return np.zeros(config.VISUAL_EMBEDDING_DIM)
    
    def encode_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Encode image into embedding."""
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            embedding = self.model(image_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        return embedding
    
    def encode_batch(self, image_tensors: List[torch.Tensor]) -> np.ndarray:
        """Encode a batch of images."""
        with torch.no_grad():
            batch = torch.stack(image_tensors).to(self.device)
            embeddings = self.model(batch)
            embeddings = embeddings.squeeze().cpu().numpy()
        return embeddings
    
    def extract_features(self, movies_df: pd.DataFrame, 
                        batch_size: int = 32,
                        normalize: bool = True) -> np.ndarray:
        """Extract visual features for all movies."""
        logger.info("=" * 80)
        logger.info("Extracting Visual Features")
        logger.info("=" * 80)
        
        embeddings_list = []
        missing_count = 0
        
        # Process in batches
        for i in tqdm(range(0, len(movies_df), batch_size), desc="Extracting visual features"):
            batch_df = movies_df.iloc[i:i+batch_size]
            batch_tensors = []
            batch_embeddings = []
            
            for _, row in batch_df.iterrows():
                # Check if poster exists
                poster_path = None
                if 'local_poster_path' in row and pd.notna(row['local_poster_path']):
                    poster_path = Path(row['local_poster_path'])
                else:
                    # Try default path
                    default_path = config.IMAGES_DIR / f"{row['movieId']}.jpg"
                    if default_path.exists():
                        poster_path = default_path
                
                # Load image
                if poster_path and poster_path.exists():
                    image_tensor = self.load_image(poster_path)
                    if image_tensor is not None:
                        batch_tensors.append(image_tensor)
                        continue
                
                # Use placeholder if image not available
                missing_count += 1
                batch_embeddings.append(self.create_placeholder_embedding())
            
            # Encode batch of valid images
            if batch_tensors:
                valid_embeddings = self.encode_batch(batch_tensors)
                if len(valid_embeddings.shape) == 1:
                    valid_embeddings = valid_embeddings.reshape(1, -1)
                batch_embeddings.extend(valid_embeddings)
            
            embeddings_list.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list)
        
        logger.info(f"Missing images: {missing_count}/{len(movies_df)} ({missing_count/len(movies_df)*100:.1f}%)")
        
        # Normalize
        if normalize:
            logger.info("Normalizing embeddings...")
            embeddings = normalize_embeddings(embeddings, method='l2')
        
        # Verify shape
        expected_shape = (len(movies_df), config.VISUAL_EMBEDDING_DIM)
        assert embeddings.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {embeddings.shape}"
        
        logger.info(f"✓ Visual features extracted: {embeddings.shape}")
        logger.info("=" * 80)
        
        return embeddings
    
    def save_features(self, embeddings: np.ndarray, movies_df: pd.DataFrame):
        """Save visual features to disk."""
        movie_ids = movies_df['movieId'].tolist()
        save_path = config.EMBEDDINGS_DIR / "visual_embeddings.pkl"
        save_embeddings(embeddings, movie_ids, save_path)
        logger.info(f"Visual features saved to {save_path}")


def create_placeholder_images(movies_df: pd.DataFrame, num_samples: int = 10):
    """Create simple placeholder images for movies without posters."""
    from PIL import ImageDraw, ImageFont
    
    logger.info(f"Creating {num_samples} placeholder images for visualization...")
    
    for i, (_, row) in enumerate(movies_df.head(num_samples).iterrows()):
        poster_path = config.IMAGES_DIR / f"{row['movieId']}.jpg"
        
        if not poster_path.exists():
            # Create a simple colored placeholder
            img = Image.new('RGB', (500, 750), color=(100, 100, 100))
            draw = ImageDraw.Draw(img)
            
            # Add movie title
            title = row['title'][:30] if len(row['title']) > 30 else row['title']
            draw.text((20, 350), title, fill=(255, 255, 255))
            
            img.save(poster_path)
            logger.debug(f"Created placeholder for: {title}")


def main():
    """Main visual feature extraction pipeline."""
    set_seed(config.RANDOM_SEED)
    
    # Load movies data
    movies_df = pd.read_csv(config.PROCESSED_DATA_DIR / "movies_final.csv")
    logger.info(f"Loaded {len(movies_df)} movies")
    
    # Extract visual features
    extractor = VisualFeatureExtractor()
    embeddings = extractor.extract_features(movies_df)
    extractor.save_features(embeddings, movies_df)
    
    return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Test mode')
    parser.add_argument('--create-placeholders', action='store_true', help='Create placeholder images')
    args = parser.parse_args()
    
    if args.create_placeholders:
        movies_df = pd.read_csv(config.PROCESSED_DATA_DIR / "movies_final.csv")
        create_placeholder_images(movies_df, num_samples=20)
    elif args.test:
        logger.info("Running in test mode...")
        logger.info(f"✓ Visual feature extractor initialized successfully")
    else:
        main()
