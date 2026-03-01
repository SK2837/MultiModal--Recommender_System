"""
Text Feature Extraction Module
Extracts text embeddings from movie titles, descriptions, and genres using NLP
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

import config
from utils import setup_logging, save_embeddings, normalize_embeddings, set_seed

logger = setup_logging(log_file=config.LOGS_DIR / "text_features.log")


class TextFeatureExtractor:
    """Extract text embeddings using transformer models."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.TEXT_MODEL_NAME
        self.device = config.DEVICE
        
        logger.info(f"Loading text model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Text model loaded successfully")
    
    def create_text_description(self, row: pd.Series) -> str:
        """Create a comprehensive text description for a movie."""
        parts = []
        
        # Title
        if 'title' in row and pd.notna(row['title']):
            parts.append(f"Title: {row['title']}")
        
        # Genres
        if 'genres' in row and pd.notna(row['genres']):
            genres = row['genres'].replace('|', ', ')
            parts.append(f"Genres: {genres}")
        
        # Overview/Description
        if 'overview' in row and pd.notna(row['overview']) and row['overview']:
            parts.append(f"Description: {row['overview']}")
        
        # Combine all parts
        text = '. '.join(parts)
        
        return text if text else "No description available"
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings using the transformer model."""
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=config.MAX_TEXT_LENGTH,
                    return_tensors='pt'
                )
                
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                # Get embeddings
                outputs = self.model(**encoded)
                
                # Use mean pooling over sequence
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Move to CPU and convert to numpy
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Encoded {len(texts)} texts into shape {all_embeddings.shape}")
        
        return all_embeddings
    
    def extract_features(self, movies_df: pd.DataFrame, normalize: bool = True) -> np.ndarray:
        """Extract text features for all movies."""
        logger.info("=" * 80)
        logger.info("Extracting Text Features")
        logger.info("=" * 80)
        
        # Create text descriptions
        logger.info("Creating text descriptions...")
        texts = [self.create_text_description(row) for _, row in movies_df.iterrows()]
        
        # Sample output
        logger.info(f"Sample text description:\n{texts[0]}\n")
        
        # Encode texts
        embeddings = self.encode_text(texts)
        
        # Normalize
        if normalize:
            logger.info("Normalizing embeddings...")
            embeddings = normalize_embeddings(embeddings, method='l2')
        
        # Verify shape
        expected_shape = (len(movies_df), config.TEXT_EMBEDDING_DIM)
        assert embeddings.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {embeddings.shape}"
        
        logger.info(f"✓ Text features extracted: {embeddings.shape}")
        logger.info("=" * 80)
        
        return embeddings
    
    def save_features(self, embeddings: np.ndarray, movies_df: pd.DataFrame):
        """Save text features to disk."""
        movie_ids = movies_df['movieId'].tolist()
        save_path = config.EMBEDDINGS_DIR / "text_embeddings.pkl"
        save_embeddings(embeddings, movie_ids, save_path)
        logger.info(f"Text features saved to {save_path}")


class TFIDFFeatureExtractor:
    """Fallback: Extract text features using TF-IDF (simpler, no GPU needed)."""
    
    def __init__(self, max_features: int = 1000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english', 
            ngram_range=(1, 2)
        )
    
    def create_text_description(self, row: pd.Series) -> str:
        """Create text description (same as transformer version)."""
        parts = []
        
        if 'title' in row and pd.notna(row['title']):
            parts.append(row['title'])
        
        if 'genres' in row and pd.notna(row['genres']):
            genres = row['genres'].replace('|', ' ')
            parts.append(genres)
        
        if 'overview' in row and pd.notna(row['overview']) and row['overview']:
            parts.append(row['overview'])
        
        return ' '.join(parts) if parts else "unknown"
    
    def extract_features(self, movies_df: pd.DataFrame) -> np.ndarray:
        """Extract TF-IDF features."""
        logger.info("Extracting TF-IDF features...")
        
        texts = [self.create_text_description(row) for _, row in movies_df.iterrows()]
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        embeddings = tfidf_matrix.toarray()
        
        logger.info(f"TF-IDF features: {embeddings.shape}")
        
        return embeddings


def main(use_tfidf: bool = False):
    """Main text feature extraction pipeline."""
    set_seed(config.RANDOM_SEED)
    
    # Load movies data
    movies_df = pd.read_csv(config.PROCESSED_DATA_DIR / "movies_final.csv")
    logger.info(f"Loaded {len(movies_df)} movies")
    
    if use_tfidf:
        # Use TF-IDF (faster, no GPU needed)
        extractor = TFIDFFeatureExtractor()
        embeddings = extractor.extract_features(movies_df)
    else:
        # Use transformer model (better quality)
        extractor = TextFeatureExtractor()
        embeddings = extractor.extract_features(movies_df)
        extractor.save_features(embeddings, movies_df)
    
    return embeddings


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfidf', action='store_true', help='Use TF-IDF instead of transformers')
    parser.add_argument('--test', action='store_true', help='Test mode with sample data')
    args = parser.parse_args()
    
    if args.test:
        logger.info("Running in test mode...")
        # Create sample data
        sample_data = pd.DataFrame({
            'movieId': [1, 2, 3],
            'title': ['Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)'],
            'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance'],
            'overview': [
                'A story about toys that come to life',
                'A magical board game adventure',
                'A romantic comedy about old men'
            ]
        })
        
        extractor = TextFeatureExtractor()
        embeddings = extractor.extract_features(sample_data)
        logger.info(f"Test passed! Embeddings shape: {embeddings.shape}")
    else:
        main(use_tfidf=args.tfidf)
