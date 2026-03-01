"""
Data Preprocessing Module
Cleans and prepares data for model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

import config
from utils import setup_logging, save_pickle, train_test_split_by_user, set_seed

logger = setup_logging(log_file=config.LOGS_DIR / "preprocessing.log")


class DataPreprocessor:
    """Preprocess MovieLens data for recommendation model."""
    
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
        self.user_id_map = None
        self.movie_id_map = None
        self.id_user_map = None
        self.id_movie_map = None
    
    def load_data(self):
        """Load preprocessed data."""
        logger.info("Loading data...")
        
        self.ratings_df = pd.read_csv(config.PROCESSED_DATA_DIR / "ratings.csv")
        self.movies_df = pd.read_csv(config.PROCESSED_DATA_DIR / "movies_enriched.csv")
        self.users_df = pd.read_csv(config.PROCESSED_DATA_DIR / "users.csv")
        
        logger.info(f"Loaded {len(self.ratings_df)} ratings")
        
    def filter_sparse_data(self):
        """Filter out users and movies with too few ratings."""
        logger.info("Filtering sparse data...")
        
        initial_ratings = len(self.ratings_df)
        initial_users = self.ratings_df['userId'].nunique()
        initial_movies = self.ratings_df['movieId'].nunique()
        
        # Iteratively filter until convergence
        prev_size = 0
        current_size = len(self.ratings_df)
        
        while prev_size != current_size:
            prev_size = current_size
            
            # Filter users with too few ratings
            user_counts = self.ratings_df['userId'].value_counts()
            valid_users = user_counts[user_counts >= config.MIN_RATINGS_PER_USER].index
            self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
            
            # Filter movies with too few ratings
            movie_counts = self.ratings_df['movieId'].value_counts()
            valid_movies = movie_counts[movie_counts >= config.MIN_RATINGS_PER_ITEM].index
            self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(valid_movies)]
            
            current_size = len(self.ratings_df)
        
        final_users = self.ratings_df['userId'].nunique()
        final_movies = self.ratings_df['movieId'].nunique()
        
        logger.info(f"Filtered data:")
        logger.info(f"  - Ratings: {initial_ratings} → {len(self.ratings_df)}")
        logger.info(f"  - Users: {initial_users} → {final_users}")
        logger.info(f"  - Movies: {initial_movies} → {final_movies}")
        
        # Update movies and users dataframes
        self.movies_df = self.movies_df[self.movies_df['movieId'].isin(
            self.ratings_df['movieId'].unique()
        )]
        self.users_df = self.users_df[self.users_df['userId'].isin(
            self.ratings_df['userId'].unique()
        )]
    
    def create_id_mappings(self):
        """Create continuous ID mappings for users and movies."""
        logger.info("Creating ID mappings...")
        
        # Get unique IDs
        unique_users = sorted(self.ratings_df['userId'].unique())
        unique_movies = sorted(self.ratings_df['movieId'].unique())
        
        # Create mappings (original ID -> continuous ID)
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}
        
        # Reverse mappings (continuous ID -> original ID)
        self.id_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.id_movie_map = {idx: mid for mid, idx in self.movie_id_map.items()}
        
        # Apply mappings to ratings
        self.ratings_df['user_idx'] = self.ratings_df['userId'].map(self.user_id_map)
        self.ratings_df['movie_idx'] = self.ratings_df['movieId'].map(self.movie_id_map)
        
        # Apply mapping to movies
        self.movies_df['movie_idx'] = self.movies_df['movieId'].map(self.movie_id_map)
        self.movies_df = self.movies_df.sort_values('movie_idx').reset_index(drop=True)
        
        logger.info(f"Created mappings for {len(self.user_id_map)} users and {len(self.movie_id_map)} movies")
        
        # Save mappings
        save_pickle(self.user_id_map, config.PROCESSED_DATA_DIR / "user_id_map.pkl")
        save_pickle(self.movie_id_map, config.PROCESSED_DATA_DIR / "movie_id_map.pkl")
        save_pickle(self.id_user_map, config.PROCESSED_DATA_DIR / "id_user_map.pkl")
        save_pickle(self.id_movie_map, config.PROCESSED_DATA_DIR / "id_movie_map.pkl")
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data...")
        
        set_seed(config.RANDOM_SEED)
        
        # Split by user (each user has ratings in all splits)
        train_df, val_df, test_df = train_test_split_by_user(
            self.ratings_df,
            test_ratio=config.TEST_RATIO,
            val_ratio=config.VAL_RATIO,
            random_state=config.RANDOM_SEED
        )
        
        logger.info(f"Split sizes:")
        logger.info(f"  - Train: {len(train_df)} ({len(train_df)/len(self.ratings_df)*100:.1f}%)")
        logger.info(f"  - Val:   {len(val_df)} ({len(val_df)/len(self.ratings_df)*100:.1f}%)")
        logger.info(f"  - Test:  {len(test_df)} ({len(test_df)/len(self.ratings_df)*100:.1f}%)")
        
        # Save splits
        train_df.to_csv(config.PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(config.PROCESSED_DATA_DIR / "val.csv", index=False)
        test_df.to_csv(config.PROCESSED_DATA_DIR / "test.csv", index=False)
        
        return train_df, val_df, test_df
    
    def compute_statistics(self):
        """Compute and save dataset statistics."""
        logger.info("Computing dataset statistics...")
        
        stats = {
            'num_users': len(self.user_id_map),
            'num_movies': len(self.movie_id_map),
            'num_ratings': len(self.ratings_df),
            'sparsity': 1 - len(self.ratings_df) / (len(self.user_id_map) * len(self.movie_id_map)),
            'avg_ratings_per_user': self.ratings_df.groupby('userId').size().mean(),
            'avg_ratings_per_movie': self.ratings_df.groupby('movieId').size().mean(),
            'rating_distribution': self.ratings_df['rating'].value_counts().to_dict(),
            'min_rating': self.ratings_df['rating'].min(),
            'max_rating': self.ratings_df['rating'].max(),
            'mean_rating': self.ratings_df['rating'].mean(),
            'std_rating': self.ratings_df['rating'].std(),
        }
        
        # Print statistics
        logger.info("=" * 60)
        logger.info("Dataset Statistics:")
        logger.info(f"  Number of users: {stats['num_users']}")
        logger.info(f"  Number of movies: {stats['num_movies']}")
        logger.info(f"  Number of ratings: {stats['num_ratings']}")
        logger.info(f"  Sparsity: {stats['sparsity']:.4f}")
        logger.info(f"  Avg ratings per user: {stats['avg_ratings_per_user']:.2f}")
        logger.info(f"  Avg ratings per movie: {stats['avg_ratings_per_movie']:.2f}")
        logger.info(f"  Rating range: [{stats['min_rating']}, {stats['max_rating']}]")
        logger.info(f"  Mean rating: {stats['mean_rating']:.2f} ± {stats['std_rating']:.2f}")
        logger.info("=" * 60)
        
        # Save statistics
        save_pickle(stats, config.PROCESSED_DATA_DIR / "statistics.pkl")
        
        return stats
    
    def preprocess(self):
        """Complete preprocessing pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Data Preprocessing Pipeline")
        logger.info("=" * 80)
        
        self.load_data()
        self.filter_sparse_data()
        self.create_id_mappings()
        train_df, val_df, test_df = self.split_data()
        stats = self.compute_statistics()
        
        # Save final processed movies data
        self.movies_df.to_csv(config.PROCESSED_DATA_DIR / "movies_final.csv", index=False)
        
        logger.info("=" * 80)
        logger.info("Preprocessing Complete!")
        logger.info("=" * 80)
        
        return train_df, val_df, test_df, self.movies_df, stats


def main():
    """Run preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df, movies_df, stats = preprocessor.preprocess()
    return train_df, val_df, test_df, movies_df, stats


if __name__ == "__main__":
    main()
