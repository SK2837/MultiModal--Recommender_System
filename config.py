"""
Configuration file for Multi-Modal Recommendation System
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
IMAGES_DIR = DATA_DIR / "images"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR, 
                 IMAGES_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset Configuration
MOVIELENS_SIZE = "100k"  # Options: "100k", "1m", "10m", "20m"
MOVIELENS_URL = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
}

# TMDB API Configuration
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")  # Get from environment variable
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Data Preprocessing
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_ITEM = 5
RANDOM_SEED = 42

# Feature Extraction Configuration
# Text Features
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight, 384-dim
# Alternative: "sentence-transformers/all-mpnet-base-v2" (768-dim, better quality)
TEXT_EMBEDDING_DIM = 384
MAX_TEXT_LENGTH = 512

# Visual Features
VISUAL_MODEL_NAME = "resnet50"  # Options: "resnet50", "resnet101", "efficientnet_b0"
VISUAL_EMBEDDING_DIM = 2048  # ResNet50 output
IMAGE_SIZE = 224
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
IMAGE_STD = [0.229, 0.224, 0.225]

# Multi-Modal Fusion
FUSION_METHOD = "concatenate"  # Options: "concatenate", "attention", "learnable"
MULTIMODAL_EMBEDDING_DIM = TEXT_EMBEDDING_DIM + VISUAL_EMBEDDING_DIM  # 384 + 2048 = 2432

# User Modeling
USER_MODELING_METHOD = "average"  # Options: "average", "learned", "sequence"
USER_EMBEDDING_DIM = 256
ITEM_EMBEDDING_DIM = 256

# Model Architecture
MODEL_TYPE = "neural_cf"  # Options: "neural_cf", "two_tower", "hybrid"
HIDDEN_LAYERS = [512, 256, 128, 64]
DROPOUT_RATE = 0.3
ACTIVATION = "relu"

# Training Configuration
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP = 1.0
WEIGHT_DECAY = 1e-5

# Device Configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

# Evaluation Configuration
TOP_K_VALUES = [5, 10, 20]
EVAL_METRICS = ["precision", "recall", "ndcg", "map", "rmse", "mae"]

# Recommendation Configuration
NUM_RECOMMENDATIONS = 10
NUM_CANDIDATES = 100  # For two-tower retrieval

# Logging
LOG_INTERVAL = 100  # Log every N batches
SAVE_INTERVAL = 1  # Save checkpoint every N epochs
TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"

# Visualization
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_COMPONENTS = 2

# Demo App Configuration
STREAMLIT_PORT = 8501
DEMO_SAMPLE_USERS = 10  # Number of users to show in demo

print(f"Configuration loaded. Using device: {DEVICE}")
if TMDB_API_KEY:
    print("✓ TMDB API key found")
else:
    print("⚠ TMDB API key not found. Set TMDB_API_KEY environment variable for poster images.")
