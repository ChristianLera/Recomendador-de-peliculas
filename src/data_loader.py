"""
Módulo de carga de datos para el sistema de recomendación.

Permite cargar datasets desde:
- MovieLens small (100k ratings)
- MovieLens large (25M ratings)  
- TMDB API (consulta en tiempo real)

Unifica todos los datasets en un formato común con:
- movie_id (identificador único)
- title (título normalizado)
- genres (lista de géneros)
- actors (lista de actores principales, top 5)
- director (nombre del director)
- year (año de lanzamiento)
- language (idioma original)
- keywords (palabras clave para similitud)
"""

import os
import json
import zipfile
import requests
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm  # Para barras de progreso (opcional)

# Importar utilidades
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, normalize_text, get_config, validate_tmdb_key

# Configurar logger
logger = setup_logging(log_to_ui=True)

class DataLoader:
    """
    Cargador de datos multi-fuente para películas.
    
    Attributes:
        source (str): Fuente de datos ('movielens_small', 'movielens_large', 'tmdb')
        data_dir (Path): Directorio donde almacenar datos
        tmdb_api_key (str): API key de TMDB
        movies_df (pd.DataFrame): DataFrame principal con información de películas
        ratings_df (pd.DataFrame): DataFrame con ratings usuario-película
    """
    
    def __init__(self, source: str = 'movielens_small'):
        """
        Inicializa el cargador de datos.
        
        Args:
            source: Fuente de datos ('movielens_small', 'movielens_large', 'tmdb')
        """
        self.source = source
        self.config = get_config()
        self.data_dir = self.config['data_dir']
        self.tmdb_api_key = self.config['tmdb_api_key']
        
        # DataFrames principales
        self.movies_df = None
        self.ratings_df = None
        
        # Cache para TMDB
        self.tmdb_cache_file = self.data_dir / 'tmdb_cache.json'
        self.tmdb_cache = self._load_tmdb_cache()
        
        logger.info(f"Inicializando DataLoader con fuente: {source}")
    
    def _load_tmdb_cache(self) -> Dict:
        """Carga caché de TMDB desde archivo JSON."""
        if self.tmdb_cache_file.exists():
            try:
                with open(self.tmdb_cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"Caché TMDB cargada con {len(cache)} películas")
                return cache
            except Exception as e:
                logger.warning(f"Error cargando caché TMDB: {e}")
        return {}
    
    def _save_tmdb_cache(self):
        """Guarda caché de TMDB a archivo JSON."""
        try:
            with open(self.tmdb_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.tmdb_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"Caché TMDB guardada con {len(self.tmdb_cache)} películas")
        except Exception as e:
            logger.warning(f"Error guardando caché TMDB: {e}")
    
    def download_movielens(self, url: str) -> Path:
        """
        Descarga dataset de MovieLens si no existe.
        
        Args:
            url: URL del dataset (small o large)
        
        Returns:
            Path: Ruta al archivo descargado
        """
        filename = url.split('/')[-1]
        zip_path = self.data_dir / filename
        
        # Verificar si ya existe
        if zip_path.exists():
            logger.info(f"Dataset ya existe: {zip_path}")
            return zip_path
        
        # Crear directorio si no existe
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Descargar
        logger.info(f"Descargando {filename}... Esto puede tomar varios minutos")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Descarga completada: {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"Error descargando {filename}: {e}")
            raise
    
    def extract_movielens(self, zip_path: Path) -> Path:
        """
        Extrae archivos ZIP de MovieLens.
        
        Args:
            zip_path: Ruta al archivo ZIP
        
        Returns:
            Path: Directorio donde se extrajeron los archivos
        """
        extract_dir = self.data_dir / zip_path.stem.replace('.zip', '')
        
        if extract_dir.exists() and (extract_dir / 'movies.csv').exists():
            logger.info(f"Dataset ya extraído en: {extract_dir}")
            return extract_dir
        
        logger.info(f"Extrayendo {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            logger.info(f"Extracción completada en: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            logger.error(f"Error extrayendo {zip_path}: {e}")
            raise
    
    def load_movielens(self, size: str = 'small') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carga datos de MovieLens."""
        urls = {
            'small': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
            'large': 'https://files.grouplens.org/datasets/movielens/ml-latest.zip'
        }
        
        zip_path = self.download_movielens(urls[size])
        extract_dir = self.extract_movielens(zip_path)
        
        movies_file = extract_dir / 'movies.csv'
        ratings_file = extract_dir / 'ratings.csv'
        
        if not movies_file.exists():
            movies_file = extract_dir / 'ml-latest' / 'movies.csv'
            ratings_file = extract_dir / 'ml-latest' / 'ratings.csv'
        
        logger.info(f"Cargando películas desde {movies_file}")
        movies_df = pd.read_csv(movies_file)
        
        # Normalizar nombres de columnas de películas
        movies_df.columns = movies_df.columns.str.lower()
        if 'movieid' in movies_df.columns:
            movies_df = movies_df.rename(columns={'movieid': 'movie_id'})
        
        logger.info(f"Cargando ratings desde {ratings_file}")
        ratings_df = pd.read_csv(ratings_file)
        
        # Normalizar nombres de columnas de ratings
        ratings_df.columns = ratings_df.columns.str.lower()
        if 'userid' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'userid': 'user_id'})
        if 'movieid' in ratings_df.columns:
            ratings_df = ratings_df.rename(columns={'movieid': 'movie_id'})
        
        # Limpiar y unificar columnas de películas
        movies_df = movies_df.rename(columns={
            'movie_id': 'movie_id',
            'title': 'title',
            'genres': 'genres'
        })
        
        # Extraer año del título
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)').fillna('0').astype(int)
        movies_df['title_clean'] = movies_df['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
        
        # ========== PROCESAR GÉNEROS CORRECTAMENTE ==========
        # Convertir géneros a lista
        movies_df['genres_list'] = movies_df['genres'].str.split('|')
        
        # Traducir géneros a español para mostrar
        genre_translation = {
            'Action': 'Acción',
            'Adventure': 'Aventura',
            'Animation': 'Animación',
            'Children': 'Infantil',
            'Comedy': 'Comedia',
            'Crime': 'Crimen',
            'Documentary': 'Documental',
            'Drama': 'Drama',
            'Fantasy': 'Fantasía',
            'Film-Noir': 'Cine Negro',
            'Horror': 'Terror',
            'IMAX': 'IMAX',
            'Musical': 'Musical',
            'Mystery': 'Misterio',
            'Romance': 'Romance',
            'Sci-Fi': 'Ciencia Ficción',
            'Thriller': 'Suspenso',
            'War': 'Guerra',
            'Western': 'Western'
        }
        
        # Aplicar traducción
        def translate_genres(genres_str):
            if pd.isna(genres_str) or genres_str == '':
                return []
            genres = genres_str.split('|')
            translated = [genre_translation.get(g, g) for g in genres if g and g != '(no genres listed)']
            return translated
        
        movies_df['genres_processed'] = movies_df['genres'].apply(translate_genres)
        movies_df['genres_str'] = movies_df['genres_processed'].apply(lambda x: ', '.join(x))
        
        # Para similitud, usar versión en inglés (original)
        movies_df['combined_features'] = movies_df['genres'].str.replace('|', ' ').str.lower()
        
        # Añadir columnas por defecto para compatibilidad
        movies_df['actors_processed'] = [[] for _ in range(len(movies_df))]
        movies_df['director_processed'] = ['Desconocido'] * len(movies_df)
        
        logger.info(f"Ejemplo de géneros procesados: {movies_df['genres_processed'].iloc[0] if len(movies_df) > 0 else 'N/A'}")
        
        logger.info(f"Cargadas {len(movies_df)} películas y {len(ratings_df)} ratings")
        
        return movies_df, ratings_df
    
    def fetch_tmdb_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """
        Obtiene detalles de una película desde TMDB API.
        
        Args:
            tmdb_id: ID de la película en TMDB
        
        Returns:
            Dict con detalles o None si falla
        """
        if not validate_tmdb_key(logger):
            return None
        
        # Verificar caché
        if str(tmdb_id) in self.tmdb_cache:
            return self.tmdb_cache[str(tmdb_id)]
        
        # URLs de TMDB
        movie_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        credits_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits"
        
        params = {
            'api_key': self.tmdb_api_key,
            'language': 'es-ES'  # Preferir español
        }
        
        try:
            # Obtener detalles básicos
            response = requests.get(movie_url, params=params, timeout=10)
            response.raise_for_status()
            movie_data = response.json()
            
            # Obtener créditos
            credits_response = requests.get(credits_url, params=params, timeout=10)
            credits_response.raise_for_status()
            credits_data = credits_response.json()
            
            # Extraer actores principales (top 5)
            actors = []
            if 'cast' in credits_data:
                actors = [actor['name'] for actor in credits_data['cast'][:5]]
            
            # Extraer director
            director = 'desconocido'
            if 'crew' in credits_data:
                directors = [crew['name'] for crew in credits_data['crew'] if crew['job'] == 'Director']
                if directors:
                    director = directors[0]
            
            # Construir diccionario
            movie_info = {
                'tmdb_id': movie_data['id'],
                'title': movie_data['title'],
                'year': int(movie_data.get('release_date', '0000')[:4]) if movie_data.get('release_date') else 0,
                'genres': [genre['name'] for genre in movie_data.get('genres', [])],
                'actors': actors,
                'director': director,
                'language': movie_data.get('original_language', 'en'),
                'keywords': f"{movie_data.get('overview', '')} {' '.join([g['name'] for g in movie_data.get('genres', [])])}",
                'poster_path': movie_data.get('poster_path', '')
            }
            
            # Guardar en caché
            self.tmdb_cache[str(tmdb_id)] = movie_info
            self._save_tmdb_cache()
            
            logger.info(f"TMDB: Cargada '{movie_info['title']}' ({movie_info['year']})")
            return movie_info
            
        except requests.RequestException as e:
            logger.warning(f"Error fetching TMDB ID {tmdb_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error procesando TMDB ID {tmdb_id}: {e}")
            return None
    
    def load_from_tmdb(self, movie_ids: Optional[List[int]] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Carga datos desde TMDB API para IDs específicos o populares.
        
        Args:
            movie_ids: Lista de IDs de TMDB (si None, carga populares)
            limit: Número máximo de películas a cargar
        
        Returns:
            pd.DataFrame: Datos de películas
        """
        if not validate_tmdb_key(logger):
            logger.error("No se puede cargar TMDB sin API key válida")
            return pd.DataFrame()
        
        movies_list = []
        
        # Si no hay IDs, obtener películas populares
        if movie_ids is None:
            logger.info(f"Obteniendo top {limit} películas populares de TMDB...")
            url = "https://api.themoviedb.org/3/movie/popular"
            params = {
                'api_key': self.tmdb_api_key,
                'language': 'es-ES',
                'page': 1
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                movie_ids = [movie['id'] for movie in data.get('results', [])[:limit]]
            except Exception as e:
                logger.error(f"Error obteniendo películas populares: {e}")
                return pd.DataFrame()
        
        # Cargar detalles de cada película
        logger.info(f"Cargando detalles de {len(movie_ids)} películas desde TMDB...")
        
        for tmdb_id in tqdm(movie_ids[:limit], desc="Cargando TMDB"):
            movie_info = self.fetch_tmdb_movie_details(tmdb_id)
            if movie_info:
                movies_list.append(movie_info)
        
        if not movies_list:
            logger.warning("No se cargaron películas desde TMDB")
            return pd.DataFrame()
        
        # Crear DataFrame
        movies_df = pd.DataFrame(movies_list)
        
        # Unificar formato con MovieLens
        movies_df['movie_id'] = movies_df['tmdb_id']
        movies_df['title_clean'] = movies_df['title']
        
        # Convertir listas a string para similitud
        movies_df['genres_str'] = movies_df['genres'].apply(lambda x: ' '.join(x) if x else '')
        movies_df['actors_str'] = movies_df['actors'].apply(lambda x: ' '.join([normalize_text(a) for a in x]) if x else '')
        movies_df['director_normalized'] = movies_df['director'].apply(lambda x: normalize_text(x))
        
        # Combinar todas las características para similitud de contenido
        movies_df['combined_features'] = (
            movies_df['genres_str'] + ' ' +
            movies_df['actors_str'] + ' ' +
            movies_df['director_normalized'] + ' ' +
            movies_df['keywords'].fillna('')
        )
        
        logger.info(f"Cargadas {len(movies_df)} películas desde TMDB")
        return movies_df
    
    def load_all_sources(self, include_tmdb: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carga datos de todas las fuentes disponibles y los unifica.
        
        Args:
            include_tmdb: Si es True, también intenta cargar datos de TMDB
        
        Returns:
            Dict con 'movies', 'ratings' y opcionalmente 'tmdb_movies'
        """
        result = {}
        
        # Cargar MovieLens
        logger.info("=== CARGANDO MOVIELENS ===")
        try:
            movies_df, ratings_df = self.load_movielens('small')  # Usar small por defecto
            result['movies'] = movies_df
            result['ratings'] = ratings_df
            logger.info(f"MovieLens cargado: {len(movies_df)} películas")
        except Exception as e:
            logger.error(f"Error cargando MovieLens: {e}")
            result['movies'] = pd.DataFrame()
            result['ratings'] = pd.DataFrame()
        
        # Cargar TMDB adicional si se solicita
        if include_tmdb and validate_tmdb_key(logger):
            logger.info("=== CARGANDO TMDB ===")
            tmdb_df = self.load_from_tmdb(limit=500)  # Límite para no saturar API
            if not tmdb_df.empty:
                result['tmdb_movies'] = tmdb_df
                logger.info(f"TMDB cargado: {len(tmdb_df)} películas adicionales")
        
        return result
    
    def prepare_features_for_similarity(self, movies_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara las características para la similitud coseno.
        
        Args:
            movies_df: DataFrame con películas
        
        Returns:
            DataFrame con columna 'combined_features' lista para vectorización
        """
        df = movies_df.copy()
        
        # Si no existe combined_features, crearla
        if 'combined_features' not in df.columns:
            # Para MovieLens, usamos solo géneros
            if 'genres_list' in df.columns:
                df['genres_str'] = df['genres_list'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
            elif 'genres' in df.columns:
                if isinstance(df['genres'].iloc[0], list):
                    df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))
                else:
                    df['genres_str'] = df['genres'].str.replace('|', ' ')
            
            # Actores y director (si existen)
            if 'actors' in df.columns:
                if isinstance(df['actors'].iloc[0], list):
                    df['actors_str'] = df['actors'].apply(lambda x: ' '.join([normalize_text(a) for a in x[:3]]) if x else '')
                else:
                    df['actors_str'] = df['actors']
            else:
                df['actors_str'] = ''
            
            if 'director' in df.columns:
                df['director_str'] = df['director'].apply(lambda x: normalize_text(x) if x else '')
            else:
                df['director_str'] = ''
            
            # Crear combined_features ponderada (repetir más veces los géneros)
            df['combined_features'] = (
                df['genres_str'] + ' ' + df['genres_str'] + ' ' +  # Dar más peso a géneros
                df['actors_str'] + ' ' +
                df['director_str']
            )
        
        # Normalizar: minúsculas, eliminar acentos
        df['combined_features'] = df['combined_features'].apply(
            lambda x: normalize_text(x, capitalize=False) if isinstance(x, str) else ''
        )
        
        logger.info(f"Características preparadas para {len(df)} películas")
        return df
