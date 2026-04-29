"""
Módulo unificado de datos - USA SOLO MOVIELENS como fuente principal
TMDB se consulta bajo demanda (no se carga masivamente)

Incluye carga automática de ratings históricos para el sistema de ⭐1-5.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import zipfile
import requests
import shutil
import time
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, get_config, validate_tmdb_key

logger = setup_logging(log_to_ui=True)


class UnifiedDataLoader:
    """
    Cargador unificado - MovieLens como fuente principal.
    TMDB solo bajo demanda (cuando el usuario ve una película).
    """
    
    MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    MOVIELENS_LARGE_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
    
    GENRE_TRANSLATION = {
        'Action': 'Acción', 'Adventure': 'Aventura', 'Animation': 'Animación',
        'Children': 'Infantil', 'Comedy': 'Comedia', 'Crime': 'Crimen',
        'Documentary': 'Documental', 'Drama': 'Drama', 'Fantasy': 'Fantasía',
        'Film-Noir': 'Cine Negro', 'Horror': 'Terror', 'IMAX': 'IMAX',
        'Musical': 'Musical', 'Mystery': 'Misterio', 'Romance': 'Romance',
        'Sci-Fi': 'Ciencia Ficción', 'Thriller': 'Suspenso', 'War': 'Guerra',
        'Western': 'Western'
    }
    
    def __init__(self):
        self.config = get_config()
        self.data_dir = self.config['data_dir']
        self.models_dir = self.config['models_dir']
        
        self.unified_file = self.data_dir / 'movielens_movies.parquet'
        self.ratings_file = self.data_dir / 'movielens_ratings.parquet'
        self.tmdb_cache_file = self.data_dir / 'tmdb_cache.json'
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.tmdb_cache = self._load_tmdb_cache()
        
        # Cache para ratings (acceso rápido)
        self._ratings_df_cache = None
        
        logger.info("UnifiedDataLoader inicializado - MovieLens como fuente principal")
    
    def _load_tmdb_cache(self) -> Dict:
        """Carga caché de TMDB desde archivo JSON."""
        if self.tmdb_cache_file.exists():
            try:
                with open(self.tmdb_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"📦 Caché TMDB cargada con {len(cache)} películas")
                return cache
            except Exception as e:
                logger.warning(f"Error cargando caché TMDB: {e}")
        return {}
    
    def _save_tmdb_cache(self):
        """Guarda caché de TMDB a archivo JSON."""
        try:
            with open(self.tmdb_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.tmdb_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Caché TMDB guardada con {len(self.tmdb_cache)} películas")
        except Exception as e:
            logger.warning(f"Error guardando caché TMDB: {e}")
    
    def download_movielens_small(self) -> Optional[Path]:
        """Descarga MovieLens Small."""
        zip_path = self.data_dir / 'ml-latest-small.zip'
        
        if zip_path.exists():
            logger.info(f"📁 Archivo ya existe: {zip_path.name}")
            return zip_path
        
        logger.info(f"⬇️ Descargando MovieLens Small (10MB)...")
        try:
            response = requests.get(self.MOVIELENS_SMALL_URL, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Descarga completada: {zip_path.name}")
            return zip_path
        except Exception as e:
            logger.error(f"❌ Error descargando: {e}")
            return None
    
    def extract_zip(self, zip_path: Path, delete_zip: bool = True) -> Optional[Path]:
        """Extrae ZIP y opcionalmente lo elimina."""
        extract_name = zip_path.stem.replace('.zip', '')
        extract_dir = self.data_dir / extract_name
        
        if extract_dir.exists() and (extract_dir / 'movies.csv').exists():
            logger.info(f"📁 Ya extraído: {extract_dir}")
            return extract_dir
        
        logger.info(f"📦 Extrayendo {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            logger.info(f"✅ Extraído a {extract_dir}")
            
            if delete_zip:
                zip_path.unlink()
                logger.info(f"🗑️ ZIP eliminado: {zip_path.name}")
            
            return extract_dir
        except Exception as e:
            logger.error(f"❌ Error extrayendo: {e}")
            return None
    
    def load_movielens_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga datos de MovieLens (películas y ratings)."""
        logger.info("\n📀 Cargando MovieLens...")
        
        zip_path = self.download_movielens_small()
        if not zip_path:
            return pd.DataFrame(), pd.DataFrame()
        
        extract_dir = self.extract_zip(zip_path, delete_zip=True)
        if not extract_dir:
            return pd.DataFrame(), pd.DataFrame()
        
        movies_file = extract_dir / 'movies.csv'
        ratings_file = extract_dir / 'ratings.csv'
        
        if not movies_file.exists():
            movies_file = extract_dir / 'ml-latest-small' / 'movies.csv'
            ratings_file = extract_dir / 'ml-latest-small' / 'ratings.csv'
        
        if not movies_file.exists():
            logger.error("❌ No se encontró movies.csv")
            return pd.DataFrame(), pd.DataFrame()
        
        movies_df = pd.read_csv(movies_file)
        movies_df.columns = ['movie_id', 'title', 'genres']
        # Asegurar que movie_id sea int
        movies_df['movie_id'] = movies_df['movie_id'].astype(int)
        
        ratings_df = pd.read_csv(ratings_file)
        ratings_df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
        # Asegurar que IDs sean int
        ratings_df['user_id'] = ratings_df['user_id'].astype(int)
        ratings_df['movie_id'] = ratings_df['movie_id'].astype(int)
        ratings_df = ratings_df.drop('timestamp', axis=1)
        
        movies_df = self._process_movies_df(movies_df)
        
        # Cache de ratings
        self._ratings_df_cache = ratings_df
        
        logger.info(f"✅ Cargadas {len(movies_df):,} películas y {len(ratings_df):,} ratings")
        
        return movies_df, ratings_df
    
    def _process_movies_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Procesa DataFrame de películas de MovieLens."""
        df = df.copy()
        
        df['year'] = df['title'].str.extract(r'\((\d{4})\)').fillna('0').astype(int)
        df['title_clean'] = df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
        
        def process_genres(genres_str):
            if pd.isna(genres_str) or genres_str == '':
                return [], '', 'sin genero'
            
            genres = genres_str.split('|')
            
            translated = []
            english_for_sim = []
            for g in genres:
                if g and g.strip() and g.strip() != '(no genres listed)':
                    translated.append(self.GENRE_TRANSLATION.get(g.strip(), g.strip()))
                    english_for_sim.append(g.strip().lower())
            
            if not translated:
                translated = ['Sin género']
                english_for_sim = ['sin genero']
            
            return translated, ', '.join(translated), ' '.join(english_for_sim)
        
        processed = df['genres'].apply(process_genres)
        df['genres_processed'] = processed.apply(lambda x: x[0])
        df['genres_str'] = processed.apply(lambda x: x[1])
        df['combined_features'] = processed.apply(lambda x: x[2])
        
        df['director'] = 'Pendiente de TMDB'
        df['director_processed'] = 'pendiente'
        df['actors'] = '[]'
        df['actors_processed'] = '[]'
        df['actors_str'] = ''
        df['tmdb_loaded'] = False
        df['tmdb_id'] = None
        df['poster_path'] = ''
        
        df['source'] = 'movielens'
        
        return df
    
    def fetch_tmdb_details_on_demand(self, movie_title: str, year: int = 0, force_refresh: bool = False) -> Dict:
        """Consulta TMDB bajo demanda para una película específica."""
        cache_key = f"{movie_title.lower().strip()}_{year}".replace(' ', '_')
        
        if not force_refresh and cache_key in self.tmdb_cache:
            logger.debug(f"📦 Usando caché TMDB para '{movie_title}'")
            return self.tmdb_cache[cache_key]
        
        if not validate_tmdb_key(logger):
            return {'director': 'No disponible (sin API key)', 'actors': [], 'tmdb_id': None, 'poster_path': ''}
        
        logger.info(f"🔍 Consultando TMDB para: '{movie_title}' (año: {year if year > 0 else 'desconocido'})")
        
        try:
            search_url = "https://api.themoviedb.org/3/search/movie"
            params = {
                'api_key': self.config['tmdb_api_key'],
                'query': movie_title,
                'language': 'es-ES'
            }
            if year > 0:
                params['year'] = year
            
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('results'):
                logger.warning(f"⚠️ No se encontró '{movie_title}' en TMDB")
                result = {'director': 'No encontrado', 'actors': [], 'tmdb_id': None, 'poster_path': ''}
                self.tmdb_cache[cache_key] = result
                self._save_tmdb_cache()
                return result
            
            tmdb_movie = data['results'][0]
            tmdb_id = tmdb_movie['id']
            
            credits_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits"
            credits_params = {
                'api_key': self.config['tmdb_api_key'],
                'language': 'es-ES'
            }
            
            credits_response = requests.get(credits_url, params=credits_params, timeout=10)
            credits_response.raise_for_status()
            credits_data = credits_response.json()
            
            director = 'Desconocido'
            if 'crew' in credits_data:
                directors = [c['name'] for c in credits_data['crew'] if c['job'] == 'Director']
                if directors:
                    director = directors[0]
            
            actors = []
            if 'cast' in credits_data:
                actors = [actor['name'] for actor in credits_data['cast'][:5]]
            
            poster_path = tmdb_movie.get('poster_path', '')
            if poster_path is None:
                poster_path = ''
            
            logger.info(f"🎯 Poster_path obtenido: '{poster_path}'")
            
            result = {
                'director': director,
                'actors': actors,
                'tmdb_id': tmdb_id,
                'title': tmdb_movie.get('title', movie_title),
                'year': int(tmdb_movie.get('release_date', '0000')[:4]) if tmdb_movie.get('release_date') else year,
                'poster_path': poster_path if poster_path else ''
            }
            
            self.tmdb_cache[cache_key] = result
            self._save_tmdb_cache()
            
            logger.info(f"✅ TMDB: '{result['title']}' - Poster: {poster_path if poster_path else 'Sin poster'}")
            return result
            
        except Exception as e:
            logger.warning(f"❌ Error consultando TMDB para '{movie_title}': {e}")
            return {'director': 'Error en consulta', 'actors': [], 'tmdb_id': None, 'poster_path': ''}
    
    def enrich_movie_with_tmdb(self, movie_row: pd.Series, force_refresh: bool = False) -> pd.Series:
        """Enriquece UNA película con datos de TMDB (bajo demanda)."""
        if not force_refresh and movie_row.get('tmdb_loaded', False):
            return movie_row
        
        title = movie_row.get('title_clean', movie_row.get('title', ''))
        year = movie_row.get('year', 0)
        
        tmdb_data = self.fetch_tmdb_details_on_demand(title, year, force_refresh)
        
        movie_row['director'] = tmdb_data.get('director', 'Desconocido')
        movie_row['director_processed'] = tmdb_data.get('director', 'Desconocido')
        movie_row['actors'] = tmdb_data.get('actors', [])
        movie_row['actors_processed'] = ','.join(tmdb_data.get('actors', []))
        movie_row['actors_str'] = ', '.join(tmdb_data.get('actors', [])[:3])
        movie_row['tmdb_id'] = tmdb_data.get('tmdb_id')
        movie_row['poster_path'] = tmdb_data.get('poster_path', '')
        movie_row['tmdb_loaded'] = True
        
        return movie_row
    
    def create_unified_dataset(self, force_reload: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Crea dataset con MovieLens como fuente principal."""
        if not force_reload and self.unified_file.exists() and self.ratings_file.exists():
            logger.info(f"\n📁 Cargando MovieLens desde caché...")
            try:
                movies_df = pd.read_parquet(self.unified_file)
                ratings_df = pd.read_parquet(self.ratings_file)
                self._ratings_df_cache = ratings_df
                logger.info(f"✅ Cargadas {len(movies_df):,} películas y {len(ratings_df):,} ratings")
                return movies_df, ratings_df
            except Exception as e:
                logger.warning(f"Error cargando caché: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("🔨 CARGANDO MOVIELENS (fuente principal)")
        logger.info("="*60)
        
        movies_df, ratings_df = self.load_movielens_data()
        
        if movies_df.empty:
            logger.error("❌ No se pudo cargar MovieLens")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"\n💾 Guardando MovieLens en caché...")
        movies_df.to_parquet(self.unified_file, index=False)
        ratings_df.to_parquet(self.ratings_file, index=False)
        
        logger.info(f"✅ DATOS GUARDADOS: {len(movies_df):,} películas | {len(ratings_df):,} ratings")
        
        return movies_df, ratings_df
    
    def get_ratings_df(self) -> Optional[pd.DataFrame]:
        """Obtiene el DataFrame de ratings (usando caché si está disponible)."""
        if self._ratings_df_cache is not None:
            return self._ratings_df_cache
        
        # Intentar cargar desde archivo
        if self.ratings_file.exists():
            try:
                self._ratings_df_cache = pd.read_parquet(self.ratings_file)
                return self._ratings_df_cache
            except Exception as e:
                logger.warning(f"Error cargando ratings desde caché: {e}")
        
        return None


# ==================== FUNCIONES PÚBLICAS ====================

def load_unified_data(force_reload: bool = False) -> pd.DataFrame:
    """Carga el dataset unificado (SOLO MOVIELENS)."""
    loader = UnifiedDataLoader()
    movies_df, _ = loader.create_unified_dataset(force_reload=force_reload)
    return movies_df


def load_ratings_data(force_reload: bool = False) -> pd.DataFrame:
    """Carga solo los ratings de MovieLens."""
    loader = UnifiedDataLoader()
    _, ratings_df = loader.create_unified_dataset(force_reload=force_reload)
    return ratings_df


def get_ratings_dataframe() -> Optional[pd.DataFrame]:
    """Obtiene el DataFrame de ratings (sin recargar todo)."""
    loader = UnifiedDataLoader()
    return loader.get_ratings_df()


def enrich_movie_with_tmdb(movie_row: pd.Series, force_refresh: bool = False) -> pd.Series:
    """Función pública para enriquecer una película con TMDB bajo demanda."""
    loader = UnifiedDataLoader()
    return loader.enrich_movie_with_tmdb(movie_row, force_refresh)


def get_user_historical_ratings(user_id: int, min_rating: int = 3) -> List[Tuple[int, int]]:
    """
    Obtiene los ratings históricos de un usuario desde MovieLens.
    
    Args:
        user_id: ID del usuario
        min_rating: Rating mínimo para incluir (por defecto 3)
    
    Returns:
        Lista de tuplas (movie_id, rating) con movie_id como int
    """
    ratings_df = get_ratings_dataframe()
    
    if ratings_df is None or ratings_df.empty:
        logger.warning(f"No hay datos de ratings para cargar histórico del usuario {user_id}")
        return []
    
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    
    if user_ratings.empty:
        return []
    
    # Filtrar por rating mínimo y redondear a entero
    result = []
    for _, row in user_ratings.iterrows():
        rating = int(round(row['rating']))
        if rating >= min_rating:
            # Asegurar que movie_id sea int
            movie_id = int(row['movie_id'])
            result.append((movie_id, rating))
    
    logger.info(f"Usuario {user_id}: {len(result)} ratings históricos con ⭐>={min_rating}")
    return result
