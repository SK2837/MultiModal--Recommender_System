"""
Data Acquisition Module
Downloads MovieLens dataset and fetches movie metadata from TMDB
"""
import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import time
from dotenv import load_dotenv

import config
from utils import setup_logging, save_pickle, save_json

# Load environment variables
load_dotenv()

logger = setup_logging(log_file=config.LOGS_DIR / "data_acquisition.log")


class MovieLensDownloader:
    """Download and extract MovieLens dataset."""
    
    def __init__(self, size: str = "100k"):
        self.size = size
        self.url = config.MOVIELENS_URL[size]
        self.raw_dir = config.RAW_DATA_DIR
        
    def download(self) -> Path:
        """Download MovieLens dataset."""
        logger.info(f"Downloading MovieLens {self.size} dataset...")
        
        zip_path = self.raw_dir / f"ml-{self.size}.zip"
        
        # Skip if already downloaded
        if zip_path.exists():
            logger.info(f"Dataset already downloaded at {zip_path}")
            return zip_path
        
        # Download with progress bar
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f, tqdm(
            desc=f"Downloading ml-{self.size}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        logger.info(f"Downloaded to {zip_path}")
        return zip_path
    
    def extract(self, zip_path: Path) -> Path:
        """Extract MovieLens dataset."""
        extract_dir = self.raw_dir / f"ml-{self.size}"
        
        if extract_dir.exists():
            logger.info(f"Dataset already extracted at {extract_dir}")
            return extract_dir
        
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        
        logger.info(f"Extracted to {extract_dir}")
        return extract_dir
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load MovieLens data into DataFrames."""
        data_dir = self.raw_dir / f"ml-{self.size}"
        
        if self.size == "100k":
            # MovieLens 100k has different file structure
            ratings = pd.read_csv(
                data_dir / "u.data",
                sep='\t',
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python'
            )
            
            movies = pd.read_csv(
                data_dir / "u.item",
                sep='|',
                encoding='latin-1',
                names=['movieId', 'title', 'release_date', 'video_release_date', 
                       'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                       'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 
                       'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                engine='python'
            )
            
            # Extract genres
            genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                         'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                         'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
            
            movies['genres'] = movies[genre_cols].apply(
                lambda row: '|'.join([col for col, val in row.items() if val == 1]),
                axis=1
            )
            
            movies = movies[['movieId', 'title', 'genres', 'release_date']]
            
            # Users
            users = pd.read_csv(
                data_dir / "u.user",
                sep='|',
                names=['userId', 'age', 'gender', 'occupation', 'zip_code'],
                engine='python'
            )
        
        else:
            # MovieLens 1M and larger
            ratings = pd.read_csv(
                data_dir / "ratings.dat",
                sep='::',
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python'
            )
            
            movies = pd.read_csv(
                data_dir / "movies.dat",
                sep='::',
                names=['movieId', 'title', 'genres'],
                encoding='latin-1',
                engine='python'
            )
            
            users = pd.read_csv(
                data_dir / "users.dat",
                sep='::',
                names=['userId', 'gender', 'age', 'occupation', 'zip_code'],
                engine='python'
            )
        
        logger.info(f"Loaded {len(ratings)} ratings, {len(movies)} movies, {len(users)} users")
        
        return ratings, movies, users
    
    def download_and_load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete download and load pipeline."""
        zip_path = self.download()
        self.extract(zip_path)
        return self.load_data()


class TMDBFetcher:
    """Fetch movie metadata from TMDB API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.TMDB_API_KEY
        self.base_url = config.TMDB_BASE_URL
        self.image_base_url = config.TMDB_IMAGE_BASE_URL
        self.session = requests.Session()
        
        if not self.api_key:
            logger.warning("TMDB API key not found. Poster fetching will be disabled.")
    
    def search_movie(self, title: str, year: Optional[int] = None) -> Optional[Dict]:
        """Search for a movie on TMDB."""
        if not self.api_key:
            return None
        
        try:
            params = {
                'api_key': self.api_key,
                'query': title
            }
            
            if year:
                params['year'] = year
            
            response = self.session.get(
                f"{self.base_url}/search/movie",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    return results[0]  # Return first match
            
            return None
        
        except Exception as e:
            logger.debug(f"Error searching for '{title}': {e}")
            return None
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """Get detailed movie information."""
        if not self.api_key:
            return None
        
        try:
            response = self.session.get(
                f"{self.base_url}/movie/{tmdb_id}",
                params={'api_key': self.api_key},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            
            return None
        
        except Exception as e:
            logger.debug(f"Error fetching movie {tmdb_id}: {e}")
            return None
    
    def download_poster(self, poster_path: str, movie_id: int) -> Optional[Path]:
        """Download movie poster image."""
        if not poster_path:
            return None
        
        try:
            url = f"{self.image_base_url}{poster_path}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                image_path = config.IMAGES_DIR / f"{movie_id}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                return image_path
            
            return None
        
        except Exception as e:
            logger.debug(f"Error downloading poster for movie {movie_id}: {e}")
            return None
    
    def enrich_movies(self, movies_df: pd.DataFrame, rate_limit: float = 0.25) -> pd.DataFrame:
        """Enrich MovieLens movies with TMDB metadata."""
        if not self.api_key:
            logger.warning("Skipping TMDB enrichment (no API key)")
            movies_df['tmdb_id'] = None
            movies_df['overview'] = ""
            movies_df['poster_path'] = None
            movies_df['vote_average'] = None
            return movies_df
        
        logger.info(f"Enriching {len(movies_df)} movies with TMDB data...")
        
        tmdb_data = []
        
        for idx, row in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Fetching TMDB data"):
            # Extract year from title (if present)
            title = row['title']
            year = None
            if '(' in title and ')' in title:
                try:
                    year_str = title.split('(')[-1].split(')')[0]
                    year = int(year_str) if year_str.isdigit() and len(year_str) == 4 else None
                    title = title.split('(')[0].strip()
                except:
                    pass
            
            # Search movie
            movie_data = self.search_movie(title, year)
            
            if movie_data:
                movie_info = {
                    'movieId': row['movieId'],
                    'tmdb_id': movie_data.get('id'),
                    'overview': movie_data.get('overview', ''),
                    'poster_path': movie_data.get('poster_path'),
                    'vote_average': movie_data.get('vote_average'),
                    'release_date': movie_data.get('release_date')
                }
                
                # Download poster
                if movie_info['poster_path']:
                    poster_path = self.download_poster(
                        movie_info['poster_path'],
                        row['movieId']
                    )
                    movie_info['local_poster_path'] = str(poster_path) if poster_path else None
                
            else:
                movie_info = {
                    'movieId': row['movieId'],
                    'tmdb_id': None,
                    'overview': '',
                    'poster_path': None,
                    'vote_average': None,
                    'release_date': None,
                    'local_poster_path': None
                }
            
            tmdb_data.append(movie_info)
            
            # Rate limiting
            time.sleep(rate_limit)
        
        # Merge with original movies data
        tmdb_df = pd.DataFrame(tmdb_data)
        enriched_df = movies_df.merge(tmdb_df, on='movieId', how='left')
        
        logger.info(f"Successfully enriched {enriched_df['tmdb_id'].notna().sum()} movies")
        
        return enriched_df


def main():
    """Main data acquisition pipeline."""
    logger.info("=" * 80)
    logger.info("Starting Data Acquisition Pipeline")
    logger.info("=" * 80)
    
    # 1. Download MovieLens
    downloader = MovieLensDownloader(size=config.MOVIELENS_SIZE)
    ratings, movies, users = downloader.download_and_load()
    
    # Save raw data
    ratings.to_csv(config.PROCESSED_DATA_DIR / "ratings.csv", index=False)
    movies.to_csv(config.PROCESSED_DATA_DIR / "movies_raw.csv", index=False)
    users.to_csv(config.PROCESSED_DATA_DIR / "users.csv", index=False)
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  - Users: {len(users)}")
    logger.info(f"  - Movies: {len(movies)}")
    logger.info(f"  - Ratings: {len(ratings)}")
    logger.info(f"  - Sparsity: {1 - len(ratings) / (len(users) * len(movies)):.4f}")
    
    # 2. Enrich with TMDB (if API key available)
    tmdb_fetcher = TMDBFetcher()
    enriched_movies = tmdb_fetcher.enrich_movies(movies)
    
    # Save enriched data
    enriched_movies.to_csv(config.PROCESSED_DATA_DIR / "movies_enriched.csv", index=False)
    
    logger.info("=" * 80)
    logger.info("Data Acquisition Complete!")
    logger.info("=" * 80)
    
    return ratings, enriched_movies, users


if __name__ == "__main__":
    main()
