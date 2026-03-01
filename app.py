"""
Streamlit Demo Application
Interactive web interface for the multi-modal recommendation system
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
from PIL import Image
from pathlib import Path

import config
from utils import load_pickle, load_embeddings, load_model
from models.neural_cf import NeuralCF
from evaluation import Evaluator


@st.cache_resource
def load_data():
    """Load all necessary data."""
    # Load movies
    movies_df = pd.read_csv(config.PROCESSED_DATA_DIR / "movies_final.csv")
    
    # Load statistics
    stats = load_pickle(config.PROCESSED_DATA_DIR / "statistics.pkl")
    
    # Load ID mappings
    id_movie_map = load_pickle(config.PROCESSED_DATA_DIR / "id_movie_map.pkl")
    
    # Load item features
    multimodal_emb, movie_ids = load_embeddings(
        config.EMBEDDINGS_DIR / "multimodal_embeddings.pkl"
    )
    item_features = torch.tensor(multimodal_emb, dtype=torch.float32)
    
    # Load ratings for display
    train_df = pd.read_csv(config.PROCESSED_DATA_DIR / "train.csv")
    
    return movies_df, stats, id_movie_map, item_features, train_df


@st.cache_resource
def load_recommendation_model():
    """Load trained recommendation model."""
    stats = load_pickle(config.PROCESSED_DATA_DIR / "statistics.pkl")
    
    model = NeuralCF(
        num_users=stats['num_users'],
        num_items=stats['num_movies'],
        user_embedding_dim=config.USER_EMBEDDING_DIM,
        item_embedding_dim=config.ITEM_EMBEDDING_DIM,
        hidden_layers=config.HIDDEN_LAYERS,
        dropout_rate=config.DROPOUT_RATE,
        use_item_features=True,
        item_feature_dim=config.MULTIMODAL_EMBEDDING_DIM
    )
    
    checkpoint_path = config.CHECKPOINTS_DIR / "best_model_neural_cf.pt"
    if checkpoint_path.exists():
        model, metadata = load_model(model, checkpoint_path, config.DEVICE)
        return model, metadata
    else:
        st.error("Model checkpoint not found! Please train the model first.")
        return None, None


def get_movie_poster(movie_id, movies_df):
    """Get movie poster image."""
    movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
    
    # Try local poster path
    if 'local_poster_path' in movie and pd.notna(movie['local_poster_path']):
        poster_path = Path(movie['local_poster_path'])
        if poster_path.exists():
            return Image.open(poster_path)
    
    # Try default path
    default_path = config.IMAGES_DIR / f"{movie_id}.jpg"
    if default_path.exists():
        return Image.open(default_path)
    
    return None


def display_movie_card(movie, movies_df, show_rating=None):
    """Display a movie card with poster and info."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        poster = get_movie_poster(movie['movieId'], movies_df)
        if poster:
            st.image(poster, use_column_width=True)
        else:
            st.write("📽️ No poster")
    
    with col2:
        st.subheader(movie['title'])
        
        if 'genres' in movie and pd.notna(movie['genres']):
            st.write(f"**Genres:** {movie['genres'].replace('|', ', ')}")
        
        if 'overview' in movie and pd.notna(movie['overview']) and movie['overview']:
            st.write(f"**Overview:** {movie['overview'][:200]}...")
        
        if show_rating is not None:
            st.write(f"**Predicted Rating:** ⭐ {show_rating:.2f} / 5.0")
        
        if 'vote_average' in movie and pd.notna(movie['vote_average']):
            st.write(f"**TMDB Rating:** {movie['vote_average']:.1f}/10")


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Multi-Modal Movie Recommender",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Multi-Modal Movie Recommendation System")
    st.markdown("**Powered by NLP + Computer Vision**")
    
    # Load data
    with st.spinner("Loading data..."):
        movies_df, stats, id_movie_map, item_features, train_df = load_data()
        model, metadata = load_recommendation_model()
    
    if model is None:
        st.stop()
    
    # Display model info
    with st.sidebar:
        st.header("📊 System Info")
        st.write(f"**Users:** {stats['num_users']:,}")
        st.write(f"**Movies:** {stats['num_movies']:,}")
        st.write(f"**Ratings:** {stats['num_ratings']:,}")
        st.write(f"**Sparsity:** {stats['sparsity']:.2%}")
        
        if metadata:
            st.write("**Model Performance:**")
            if 'val_loss' in metadata:
                st.write(f"Validation Loss: {metadata['val_loss']:.4f}")
            if 'epoch' in metadata:
                st.write(f"Trained Epochs: {metadata['epoch'] + 1}")
        
        st.markdown("---")
        st.write("**Features:**")
        st.write("• Text embeddings (BERT)")
        st.write("• Visual embeddings (ResNet50)")
        st.write("• Neural collaborative filtering")
    
    # Main interface
    st.header("🔍 Get Personalized Recommendations")
    
    # User selection
    user_ids = sorted(train_df['user_idx'].unique())
    selected_user_idx = st.selectbox(
        "Select a user:",
        options=user_ids,
        format_func=lambda x: f"User {x + 1}"
    )
    
    # Number of recommendations
    num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating personalized recommendations..."):
            # Get user's past ratings
            user_ratings = train_df[train_df['user_idx'] == selected_user_idx]
            user_rated_items = user_ratings['movie_idx'].tolist()
            
            # Get recommendations
            evaluator = Evaluator(model, "neural_cf")
            recommended_indices = evaluator.get_user_recommendations(
                user_id=selected_user_idx,
                item_features=item_features,
                k=num_recs,
                exclude_items=user_rated_items  # Exclude already rated
            )
            
            # Get predicted ratings
            user_ids_tensor = torch.full(
                (len(recommended_indices),),
                selected_user_idx,
                dtype=torch.long,
                device=config.DEVICE
            )
            item_ids_tensor = torch.tensor(
                recommended_indices,
                dtype=torch.long,
                device=config.DEVICE
            )
            item_features_tensor = item_features[recommended_indices].to(config.DEVICE)
            
            with torch.no_grad():
                predicted_ratings = model(
                    user_ids_tensor,
                    item_ids_tensor,
                    item_features_tensor
                ).cpu().numpy()
            
            # Convert indices to movie IDs
            recommended_movie_ids = [id_movie_map[idx] for idx in recommended_indices]
        
        # Display recommendations
        st.success(f"✨ Top {num_recs} Recommendations for User {selected_user_idx + 1}")
        
        for i, (movie_id, rating) in enumerate(zip(recommended_movie_ids, predicted_ratings)):
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            
            with st.expander(f"#{i+1}: {movie['title']}", expanded=(i < 3)):
                display_movie_card(movie, movies_df, show_rating=rating)
        
        # Show user's past favorites
        st.markdown("---")
        st.subheader("📚 User's Past Favorites")
        
        top_rated = user_ratings.nlargest(5, 'rating')
        
        cols = st.columns(5)
        for idx, (_, rating_row) in enumerate(top_rated.iterrows()):
            movie_id = id_movie_map[int(rating_row['movie_idx'])]
            movie = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            
            with cols[idx]:
                poster = get_movie_poster(movie_id, movies_df)
                if poster:
                    st.image(poster, use_column_width=True)
                st.caption(f"{movie['title']}")
                st.write(f"⭐ {rating_row['rating']:.1f}")
    
    # Statistics section
    st.markdown("---")
    st.header("📈 Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Users", f"{stats['num_users']:,}")
        st.metric("Total Movies", f"{stats['num_movies']:,}")
    
    with col2:
        st.metric("Total Ratings", f"{stats['num_ratings']:,}")
        st.metric("Avg Ratings/User", f"{stats['avg_ratings_per_user']:.1f}")
    
    with col3:
        st.metric("Data Sparsity", f"{stats['sparsity']:.2%}")
        st.metric("Avg Ratings/Movie", f"{stats['avg_ratings_per_movie']:.1f}")
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_dist = pd.DataFrame.from_dict(
        stats['rating_distribution'],
        orient='index',
        columns=['Count']
    ).sort_index()
    st.bar_chart(rating_dist)


if __name__ == "__main__":
    main()
