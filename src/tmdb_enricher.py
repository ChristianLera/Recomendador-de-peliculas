"""
Módulo para enriquecer películas de MovieLens con datos de TMDB bajo demanda.
"""

import requests
import json
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import time

from src.utils import setup_logging, get_config, validate_tmdb_key

logger = setup_logging(log_to_ui=True)

class TMDBEnricher:
    """
    Enriquece películas de MovieLens con datos de TMDB (director, actores).
    """
    
    def __init__(self):
        self.config = get_config()
        self.api_key = self.config['tmdb_api_key']
        self.cache_file = self.config['data_dir'] / 'tmdb_enrichment_cache.json'
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Carga caché de TMDB."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"📦 Caché TMDB cargada con {len(cache)} películas")
                return cache
            except Exception as e:
                logger.warning(f"Error cargando caché: {e}")
        return {}
    
    def _save_cache(self):
        """Guarda caché de TMDB."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Caché TMDB guardada con {len(self.cache)} películas")
        except Exception as e:
            logger.warning(f"Error guardando caché: {e}")
    
    def search_movie(self, title: str, year: int = None) -> Optional[Dict]:
        """
        Busca una película en TMDB por título y año.
        
        Args:
            title: Título de la película
            year: Año de lanzamiento (opcional)
        
        Returns:
            Diccionario con resultados o None
        """
        if not validate_tmdb_key(logger):
            return None
        
        # Construir query
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {
            'api_key': self.api_key,
            'query': title,
            'language': 'es-ES',
            'year': year if year else ''
        }
        
        try:
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('results'):
                # Devolver el primer resultado
                return data['results'][0]
            return None
            
        except Exception as e:
            logger.warning(f"Error buscando '{title}': {e}")
            return None
    
    def get_movie_details(self, tmdb_id: int) -> Optional[Dict]:
        """
        Obtiene detalles completos de una película por TMDB ID.
        
        Args:
            tmdb_id: ID de TMDB
        
        Returns:
            Diccionario con director, actores, etc.
        """
        if not validate_tmdb_key(logger):
            return None
        
        # Verificar caché
        cache_key = str(tmdb_id)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Obtener detalles de la película
            movie_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
            credits_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits"
            
            params = {'api_key': self.api_key, 'language': 'es-ES'}
            
            # Obtener datos básicos
            movie_response = requests.get(movie_url, params=params, timeout=10)
            movie_response.raise_for_status()
            movie_data = movie_response.json()
            
            # Obtener créditos (director, actores)
            credits_response = requests.get(credits_url, params=params, timeout=10)
            credits_response.raise_for_status()
            credits_data = credits_response.json()
            
            # Extraer director
            director = 'Desconocido'
            if 'crew' in credits_data:
                directors = [c['name'] for c in credits_data['crew'] if c['job'] == 'Director']
                if directors:
                    director = directors[0]
            
            # Extraer actores principales (top 5)
            actors = []
            if 'cast' in credits_data:
                actors = [actor['name'] for actor in credits_data['cast'][:5]]
            
            result = {
                'tmdb_id': tmdb_id,
                'director': director,
                'actors': actors,
                'title': movie_data.get('title', ''),
                'year': int(movie_data.get('release_date', '0000')[:4]) if movie_data.get('release_date') else 0,
                'poster_path': movie_data.get('poster_path', ''),
                'overview': movie_data.get('overview', ''),
                'genres': [g['name'] for g in movie_data.get('genres', [])]
            }
            
            # Guardar en caché
            self.cache[cache_key] = result
            self._save_cache()
            
            logger.info(f"🎬 TMDB: Cargado '{result['title']}' - Director: {director}")
            return result
            
        except Exception as e:
            logger.warning(f"Error obteniendo detalles de TMDB ID {tmdb_id}: {e}")
            return None
    
    def enrich_movie(self, movie_row: Dict, force_refresh: bool = False) -> Dict:
        """
        Enriquece una película de MovieLens con datos de TMDB.
        
        Args:
            movie_row: Fila de la película (con title, year)
            force_refresh: Si True, ignora caché
        
        Returns:
            Diccionario enriquecido
        """
        # Verificar si ya está enriquecida
        if not force_refresh and movie_row.get('tmdb_loaded'):
            return movie_row
        
        title = movie_row.get('title_clean', movie_row.get('title', ''))
        year = movie_row.get('year', 0)
        
        # Buscar en TMDB
        search_result = self.search_movie(title, year if year > 0 else None)
        
        if search_result:
            tmdb_id = search_result.get('id')
            if tmdb_id:
                details = self.get_movie_details(tmdb_id)
                if details:
                    movie_row['director'] = details.get('director', 'Desconocido')
                    movie_row['director_processed'] = details.get('director', 'Desconocido')
                    movie_row['actors'] = details.get('actors', [])
                    movie_row['actors_processed'] = details.get('actors', [])
                    movie_row['tmdb_loaded'] = True
                    movie_row['tmdb_id'] = tmdb_id
                else:
                    movie_row['director'] = 'No encontrado en TMDB'
                    movie_row['actors'] = []
                    movie_row['tmdb_loaded'] = False
            else:
                movie_row['director'] = 'No encontrado en TMDB'
                movie_row['actors'] = []
                movie_row['tmdb_loaded'] = False
        else:
            movie_row['director'] = 'No encontrado en TMDB'
            movie_row['actors'] = []
            movie_row['tmdb_loaded'] = False
        
        return movie_row
    
    def enrich_movies_batch(self, movies_df, movie_ids: list, force_refresh: bool = False):
        """
        Enriquece un lote de películas (bajo demanda).
        
        Args:
            movies_df: DataFrame con películas
            movie_ids: Lista de IDs a enriquecer
            force_refresh: Si True, ignora caché
        
        Returns:
            DataFrame actualizado
        """
        for movie_id in movie_ids:
            idx = movies_df[movies_df['movie_id'] == movie_id].index
            if len(idx) > 0:
                movie_row = movies_df.loc[idx[0]].to_dict()
                enriched = self.enrich_movie(movie_row, force_refresh)
                for key, value in enriched.items():
                    movies_df.loc[idx[0], key] = value
                time.sleep(0.1)  # Pequeña pausa para no saturar API
        
        return movies_df
