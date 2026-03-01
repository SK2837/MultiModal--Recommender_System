"""
Training Pipeline for Recommendation Models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import time
from typing import Dict, Optional
from tqdm import tqdm

import config
from utils import (setup_logging, save_model, load_model, AverageMeter, 
                  format_time, print_metrics, set_seed, load_pickle)
from models.neural_cf import NeuralCF, TwoTowerModel
from dataset import create_dataloaders

logger = setup_logging(log_file=config.LOGS_DIR / "training.log")


class Trainer:
    """Trainer for recommendation models."""
    
    def __init__(self,
                 model: nn.Module,
                 model_type: str = "neural_cf",
                 learning_rate: float = None,
                 weight_decay: float = None,
                 device: torch.device = None):
        """
        Args:
            model: PyTorch model
            model_type: Type of model ("neural_cf" or "two_tower")
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization)
            device: Device to train on
        """
        self.model = model
        self.model_type = model_type
        self.device = device or config.DEVICE
        self.model.to(self.device)
        
        # Optimizer
        lr = learning_rate or config.LEARNING_RATE
        wd = weight_decay or config.WEIGHT_DECAY
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.TENSORBOARD_LOG_DIR)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        logger.info(f"Trainer initialized for {model_type}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Learning rate: {lr}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        losses = AverageMeter("Loss")
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            if len(batch) == 4:
                user_ids, item_ids, item_features, ratings = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                item_features = item_features.to(self.device)
                ratings = ratings.to(self.device)
            else:
                user_ids, item_ids, ratings = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                item_features = None
            
            # Forward pass
            if self.model_type == "neural_cf":
                predictions = self.model(user_ids, item_ids, item_features)
            elif self.model_type == "two_tower":
                predictions = self.model(user_ids, item_features)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Compute loss
            loss = self.criterion(predictions, ratings)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.GRADIENT_CLIP)
            
            self.optimizer.step()
            
            # Update metrics
            losses.update(loss.item(), len(ratings))
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
            
            # Log to TensorBoard
            if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
        
        epoch_time = time.time() - epoch_start
        
        metrics = {
            'loss': losses.avg,
            'time': epoch_time
        }
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        losses = AverageMeter("Loss")
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Unpack batch
                if len(batch) == 4:
                    user_ids, item_ids, item_features, ratings = batch
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    item_features = item_features.to(self.device)
                    ratings = ratings.to(self.device)
                else:
                    user_ids, item_ids, ratings = batch
                    user_ids = user_ids.to(self.device)
                    item_ids = item_ids.to(self.device)
                    ratings = ratings.to(self.device)
                    item_features = None
                
                # Forward pass
                if self.model_type == "neural_cf":
                    predictions = self.model(user_ids, item_ids, item_features)
                elif self.model_type == "two_tower":
                    predictions = self.model(user_ids, item_features)
                
                # Compute loss
                loss = self.criterion(predictions, ratings)
                
                # Update metrics
                losses.update(loss.item(), len(ratings))
        
        metrics = {
            'loss': losses.avg
        }
        
        return metrics
    
    def fit(self,
            train_loader,
            val_loader,
            num_epochs: int = None,
            early_stopping_patience: int = None):
        """Train the model."""
        num_epochs = num_epochs or config.NUM_EPOCHS
        patience = early_stopping_patience or config.EARLY_STOPPING_PATIENCE
        
        logger.info("=" * 80)
        logger.info("Starting Training")
        logger.info("=" * 80)
        
        training_start = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss:   {val_metrics['loss']:.4f}")
            logger.info(f"  Time:       {format_time(train_metrics['time'])}")
            logger.info(f"  LR:         {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # TensorBoard logging
            self.writer.add_scalar('train/epoch_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.early_stop_counter = 0
                
                save_path = config.CHECKPOINTS_DIR / f"best_model_{self.model_type}.pt"
                save_model(
                    self.model,
                    save_path,
                    metadata={
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'model_type': self.model_type
                    }
                )
                logger.info(f"  ✓ Best model saved (val_loss: {val_metrics['loss']:.4f})")
            else:
                self.early_stop_counter += 1
            
            # Early stopping
            if self.early_stop_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint periodically
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                checkpoint_path = config.CHECKPOINTS_DIR / f"checkpoint_{self.model_type}_epoch{epoch+1}.pt"
                save_model(self.model, checkpoint_path, metadata={'epoch': epoch})
        
        training_time = time.time() - training_start
        
        logger.info("=" * 80)
        logger.info("Training Complete!")
        logger.info(f"Total time: {format_time(training_time)}")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")
        logger.info("=" * 80)
        
        self.writer.close()
        
        return self.best_val_loss


def main(model_type: str = "neural_cf"):
    """Main training pipeline."""
    set_seed(config.RANDOM_SEED)
    
    logger.info(f"Training {model_type} model")
    
    # Load dataset statistics
    stats = load_pickle(config.PROCESSED_DATA_DIR / "statistics.pkl")
    num_users = stats['num_users']
    num_items = stats['num_movies']
    
    logger.info(f"Number of users: {num_users}")
    logger.info(f"Number of items: {num_items}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, item_features = create_dataloaders()
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Initialize trainer
    trainer = Trainer(model, model_type=model_type)
    
    # Train
    best_val_loss = trainer.fit(train_loader, val_loader)
    
    return model, best_val_loss


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='neural_cf',
                       choices=['neural_cf', 'two_tower'],
                       help='Model type to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    
    args = parser.parse_args()
    
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    main(model_type=args.model)
