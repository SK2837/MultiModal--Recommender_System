"""
Neural Collaborative Filtering Model
Combines user and item embeddings with multi-layer perceptron for rating prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NeuralCF(nn.Module):
    """
    Neural Collaborative Filtering model that combines:
    - User embeddings (learned)
    - Item multi-modal features (pre-computed)
    - Deep neural network for prediction
    """
    
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 user_embedding_dim: int = 256,
                 item_embedding_dim: int = 256,
                 hidden_layers: list = [512, 256, 128, 64],
                 dropout_rate: float = 0.3,
                 use_item_features: bool = True,
                 item_feature_dim: Optional[int] = None):
        """
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            user_embedding_dim: Dimension of user  embeddings
            item_embedding_dim: Dimension of item embeddings
            hidden_layers: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_item_features: Whether to use pre-computed item features
            item_feature_dim: Dimension of pre-computed item features (if used)
        """
        super(NeuralCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.use_item_features = use_item_features
        
        # User embedding layer
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        
        # Item embedding layer (or feature projection)
        if use_item_features:
            assert item_feature_dim is not None, "item_feature_dim required when using item features"
            # Project item features to embedding space
            self.item_feature_projection = nn.Sequential(
                nn.Linear(item_feature_dim, item_embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            # Learnable item embeddings (traditional CF)
            self.item_embedding = nn.Embedding(num_items, item_embedding_dim)
        
        # MLP layers
        input_dim = user_embedding_dim + item_embedding_dim
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        if not self.use_item_features:
            nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids: torch.Tensor, 
                item_ids: torch.Tensor,
                item_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: User indices [batch_size]
            item_ids: Item indices [batch_size]
            item_features: Pre-computed item features [batch_size, item_feature_dim] (if using features)
        
        Returns:
            Predicted ratings [batch_size, 1]
        """
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, user_embedding_dim]
        
        # Get item embeddings
        if self.use_item_features:
            assert item_features is not None, "item_features required when use_item_features=True"
            item_emb = self.item_feature_projection(item_features)  # [batch_size, item_embedding_dim]
        else:
            item_emb = self.item_embedding(item_ids)  # [batch_size, item_embedding_dim]
        
        # Concatenate user and item embeddings
        combined = torch.cat([user_emb, item_emb], dim=1)  # [batch_size, user_dim + item_dim]
        
        # Pass through MLP
        mlp_output = self.mlp(combined)  # [batch_size, final_hidden_dim]
        
        # Predict rating
        prediction = self.output(mlp_output)  # [batch_size, 1]
        
        return prediction.squeeze()  # [batch_size]
    
    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user embeddings for given user IDs."""
        return self.user_embedding(user_ids)
    
    def predict_all_items(self, user_id: int, item_features: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for a user on all items.
        
        Args:
            user_id: Single user ID
            item_features: Features for all items [num_items, item_feature_dim]
        
        Returns:
            Predicted ratings for all items [num_items]
        """
        user_ids = torch.full((len(item_features),), user_id, dtype=torch.long, device=item_features.device)
        item_ids = torch.arange(len(item_features), device=item_features.device)
        
        with torch.no_grad():
            predictions = self.forward(user_ids, item_ids, item_features)
        
        return predictions


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for efficient retrieval-based recommendation.
    Separately encodes users and items, then computes similarity.
    """
    
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embedding_dim: int = 256,
                 user_tower_layers: list = [512, 256],
                 item_tower_layers: list = [512, 256],
                 dropout_rate: float = 0.3,
                 item_feature_dim: Optional[int] = None):
        """
        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_dim: Final embedding dimension (for both towers)
            user_tower_layers: Hidden layers for user tower
            item_tower_layers: Hidden layers for item tower
            dropout_rate: Dropout probability
            item_feature_dim: Dimension of pre-computed item features
        """
        super(TwoTowerModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # User Tower
        self.user_embedding = nn.Embedding(num_users, user_tower_layers[0])
        
        user_tower = []
        prev_dim = user_tower_layers[0]
        for hidden_dim in user_tower_layers[1:]:
            user_tower.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        user_tower.append(nn.Linear(prev_dim, embedding_dim))
        self.user_tower = nn.Sequential(*user_tower)
        
        # Item Tower
        assert item_feature_dim is not None, "item_feature_dim required for TwoTower model"
        
        item_tower = []
        prev_dim = item_feature_dim
        for hidden_dim in item_tower_layers:
            item_tower.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        item_tower.append(nn.Linear(prev_dim, embedding_dim))
        self.item_tower = nn.Sequential(*item_tower)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Encode users to embedding space."""
        user_emb = self.user_embedding(user_ids)
        user_encoding = self.user_tower(user_emb)
        # L2 normalize
        user_encoding = F.normalize(user_encoding, p=2, dim=1)
        return user_encoding
    
    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Encode items to embedding space."""
        item_encoding = self.item_tower(item_features)
        # L2 normalize
        item_encoding = F.normalize(item_encoding, p=2, dim=1)
        return item_encoding
    
    def forward(self, user_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: User indices [batch_size]
            item_features: Item features [batch_size, item_feature_dim]
        
        Returns:
            Similarity scores [batch_size]
        """
        user_encoding = self.encode_user(user_ids)  # [batch_size, embedding_dim]
        item_encoding = self.encode_item(item_features)  # [batch_size, embedding_dim]
        
        # Dot product similarity
        similarity = (user_encoding * item_encoding).sum(dim=1)  # [batch_size]
        
        return similarity
    
    def get_all_item_embeddings(self, all_item_features: torch.Tensor) -> torch.Tensor:
        """Pre-compute embeddings for all items (for efficient retrieval)."""
        with torch.no_grad():
            item_embeddings = self.encode_item(all_item_features)
        return item_embeddings
    
    def recommend_top_k(self, user_id: int, item_embeddings: torch.Tensor, k: int = 10) -> torch.Tensor:
        """
        Find top-k items for a user.
        
        Args:
            user_id: Single user ID
            item_embeddings: Pre-computed item embeddings [num_items, embedding_dim]
            k: Number of recommendations
        
        Returns:
            Indices of top-k items [k]
        """
        user_ids = torch.tensor([user_id], device=item_embeddings.device)
        user_encoding = self.encode_user(user_ids)  # [1, embedding_dim]
        
        # Compute similarity with all items
        similarities = torch.matmul(user_encoding, item_embeddings.T).squeeze()  # [num_items]
        
        # Get top-k
        top_k_indices = torch.topk(similarities, k).indices
        
        return top_k_indices


if __name__ == "__main__":
    # Test models
    print("Testing Neural CF...")
    model = NeuralCF(
        num_users=1000,
        num_items=500,
        user_embedding_dim=256,
        item_embedding_dim=256,
        hidden_layers=[512, 256, 128, 64],
        use_item_features=True,
        item_feature_dim=2432  # 384 (text) + 2048 (visual)
    )
    
    batch_size = 32
    user_ids = torch.randint(0, 1000, (batch_size,))
    item_ids = torch.randint(0, 500, (batch_size,))
    item_features = torch.randn(batch_size, 2432)
    
    predictions = model(user_ids, item_ids, item_features)
    print(f"NeuralCF predictions shape: {predictions.shape}")
    print(f"NeuralCF parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting Two-Tower Model...")
    two_tower = TwoTowerModel(
        num_users=1000,
        num_items=500,
        embedding_dim=256,
        item_feature_dim=2432
    )
    
    similarities = two_tower(user_ids, item_features)
    print(f"TwoTower similarities shape: {similarities.shape}")
    print(f"TwoTower parameters: {sum(p.numel() for p in two_tower.parameters()):,}")
    
    print("\n✓ Model tests passed!")
