"""
Evaluation Module
Computes recommendation metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

import config
from utils import setup_logging, load_pickle, print_metrics

logger = setup_logging(log_file=config.LOGS_DIR / "evaluation.log")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def precision_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
    """Precision@K: fraction of recommended items that are relevant."""
    if len(predictions) == 0 or k == 0:
        return 0.0
    
    top_k = predictions[:k]
    relevant = set(ground_truth)
    hits = len(set(top_k) & relevant)
    
    return hits / k


def recall_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
    """Recall@K: fraction of relevant items that are recommended."""
    if len(ground_truth) == 0 or k == 0:
        return 0.0
    
    top_k = predictions[:k]
    relevant = set(ground_truth)
    hits = len(set(top_k) & relevant)
    
    return hits / len(relevant)


def ndcg_at_k(predictions: List[int], ground_truth: List[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain@K."""
    if len(predictions) == 0 or len(ground_truth) == 0 or k == 0:
        return 0.0
    
    top_k = predictions[:k]
    relevant = set(ground_truth)
    
    # DCG: sum of (relevance / log2(position + 1))
    dcg = sum([1.0 / np.log2(i + 2) if item in relevant else 0.0 
               for i, item in enumerate(top_k)])
    
    # IDCG: DCG of perfect ranking
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def average_precision(predictions: List[int], ground_truth: List[int]) -> float:
    """Average Precision."""
    if len(ground_truth) == 0:
        return 0.0
    
    relevant = set(ground_truth)
    hits = 0
    sum_precisions = 0.0
    
    for i, item in enumerate(predictions):
        if item in relevant:
            hits += 1
            precision = hits / (i + 1)
            sum_precisions += precision
    
    if hits == 0:
        return 0.0
    
    return sum_precisions / len(relevant)


class Evaluator:
    """Evaluate recommendation models."""
    
    def __init__(self, model: nn.Module, model_type: str, device: torch.device = None):
        self.model = model
        self.model_type = model_type
        self.device = device or config.DEVICE
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_ratings(self, data_loader) -> Dict[str, float]:
        """Evaluate rating prediction metrics (RMSE, MAE)."""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating ratings"):
                # Unpack batch
                if len(batch) == 4:
                    user_ids, item_ids, item_features, ratings = batch
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    item_features = item_features.to(self.device)
                else:
                    user_ids, item_ids, ratings = batch
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    item_features = None
                
                # Predictions
                if self.model_type == "neural_cf":
                    predictions = self.model(user_ids, item_ids, item_features)
                elif self.model_type == "two_tower":
                    predictions = self.model(user_ids, item_features)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {
            'rmse': rmse(all_targets, all_predictions),
            'mae': mae(all_targets, all_predictions)
        }
        
        return metrics
    
    def get_user_recommendations(self,
                                 user_id: int,
                                 item_features: torch.Tensor,
                                 k: int = 10,
                                 exclude_items: List[int] = None) -> List[int]:
        """Get top-k recommendations for a user."""
        num_items = len(item_features)
        
        # Create batch for all items
        user_ids = torch.full((num_items,), user_id, dtype=torch.long, device=self.device)
        item_ids = torch.arange(num_items, device=self.device)
        item_features = item_features.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            if self.model_type == "neural_cf":
                predictions = self.model(user_ids, item_ids, item_features)
            elif self.model_type == "two_tower":
                predictions = self.model(user_ids, item_features)
            predictions = predictions.cpu().numpy()
        
        # Exclude items user has already interacted with
        if exclude_items:
            predictions[exclude_items] = -np.inf
        
        # Get top-k
        top_k_indices = np.argsort(predictions)[::-1][:k]
        
        return top_k_indices.tolist()
    
    def evaluate_ranking(self,
                        test_df: pd.DataFrame,
                        item_features: np.ndarray,
                        k_values: List[int] = None) -> Dict[str, float]:
        """Evaluate ranking metrics (Precision@K, Recall@K, NDCG@K, MAP)."""
        k_values = k_values or config.TOP_K_VALUES
        
        # Convert to torch
        item_features_tensor = torch.tensor(item_features, dtype=torch.float32)
        
        # Group test data by user
        user_test_items = defaultdict(list)
        for _, row in test_df.iterrows():
            user_id = int(row['user_idx'])
            item_id = int(row['movie_idx'])
            if row['rating'] >= 3.5:  # Consider as relevant if rating >= 3.5
                user_test_items[user_id].append(item_id)
        
        # Compute metrics for each user
        metrics_by_user = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
        map_scores = []
        
        logger.info(f"Evaluating {len(user_test_items)} users...")
        
        for user_id, relevant_items in tqdm(user_test_items.items(), desc="Evaluating ranking"):
            if len(relevant_items) == 0:
                continue
            
            # Get recommendations
            recommendations = self.get_user_recommendations(
                user_id,
                item_features_tensor,
                k=max(k_values),
                exclude_items=None  # Could exclude training items here
            )
            
            # Compute MAP
            ap = average_precision(recommendations, relevant_items)
            map_scores.append(ap)
            
            # Compute metrics for each k
            for k in k_values:
                prec = precision_at_k(recommendations, relevant_items, k)
                rec = recall_at_k(recommendations, relevant_items, k)
                ndcg = ndcg_at_k(recommendations, relevant_items, k)
                
                metrics_by_user[k]['precision'].append(prec)
                metrics_by_user[k]['recall'].append(rec)
                metrics_by_user[k]['ndcg'].append(ndcg)
        
        # Average metrics
        result = {}
        for k in k_values:
            result[f'precision@{k}'] = np.mean(metrics_by_user[k]['precision'])
            result[f'recall@{k}'] = np.mean(metrics_by_user[k]['recall'])
            result[f'ndcg@{k}'] = np.mean(metrics_by_user[k]['ndcg'])
        
        result['map'] = np.mean(map_scores)
        
        return result
    
    def evaluate_all(self, test_loader, test_df, item_features) -> Dict[str, float]:
        """Evaluate all metrics."""
        logger.info("=" * 80)
        logger.info("Comprehensive Evaluation")
        logger.info("=" * 80)
        
        # Rating prediction metrics
        logger.info("\n1. Rating Prediction Metrics")
        rating_metrics = self.evaluate_ratings(test_loader)
        print_metrics(rating_metrics)
        
        # Ranking metrics
        logger.info("\n2. Ranking Metrics")
        ranking_metrics = self.evaluate_ranking(test_df, item_features)
        print_metrics(ranking_metrics)
        
        # Combine all metrics
        all_metrics = {**rating_metrics, **ranking_metrics}
        
        logger.info("=" * 80)
        
        return all_metrics


def main(model_path: str = None, model_type: str = "neural_cf"):
    """Main evaluation pipeline."""
    from dataset import create_dataloaders
    from models.neural_cf import NeuralCF, TwoTowerModel
    from utils import load_model
    
    # Load data
    train_loader, val_loader, test_loader, item_features = create_dataloaders()
    test_df = pd.read_csv(config.PROCESSED_DATA_DIR / "test.csv")
    
    # Load model statistics
    stats = load_pickle(config.PROCESSED_DATA_DIR / "statistics.pkl")
    num_users = stats['num_users']
    num_items = stats['num_movies']
    
    # Initialize model
    if model_type == "neural_cf":
        model = NeuralCF(
            num_users=num_users,
            num_items=num_items,
            user_embedding_dim=config.USER_EMBEDDING_DIM,
            item_embedding_dim=config.ITEM_EMBEDDING_DIM,
            hidden_layers=config.HIDDEN_LAYERS,
            dropout_rate=config.DROPOUT_RATE,
            use_item_features=True,
            item_feature_dim=config.MULTIMODAL_EMBEDDING_DIM
        )
    elif model_type == "two_tower":
        model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=config.ITEM_EMBEDDING_DIM,
            item_feature_dim=config.MULTIMODAL_EMBEDDING_DIM
        )
    
    # Load trained weights
    model_path = model_path or config.CHECKPOINTS_DIR / f"best_model_{model_type}.pt"
    model, metadata = load_model(model, model_path, config.DEVICE)
    
    logger.info(f"Loaded model from {model_path}")
    logger.info(f"Training metadata: {metadata}")
    
    # Evaluate
    evaluator = Evaluator(model, model_type)
    metrics = evaluator.evaluate_all(test_loader, test_df, item_features)
    
    # Save results
    from utils import save_json
    save_json(metrics, config.LOGS_DIR / f"evaluation_results_{model_type}.json")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='neural_cf',
                       choices=['neural_cf', 'two_tower'])
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    main(model_path=args.checkpoint, model_type=args.model)
