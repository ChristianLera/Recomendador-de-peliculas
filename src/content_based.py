"""
Módulo de recomendación basada en contenido.

Implementa:
- Cálculo de similitud coseno entre películas usando CountVectorizer
- Recomendación de películas similares basadas en características (género, actores, director)
- Cacheo de matriz de similitud en disco para evitar recálculos
- Búsqueda eficiente de películas similares
- Soporte para sistema de calificaciones ⭐1-5
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importar utilidades
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    setup_logging, safe_execute, MovieNotFoundError, 
    InsufficientDataError, get_config
)
from src.preprocess import DataPreprocessor

logger = setup_logging(log_to_ui=True)


class ContentBasedRecommender:
    """
    Recomendador basado en contenido usando similitud coseno.
    
    Calcula la similitud entre películas basándose en sus características
    (géneros, actores, director, año, idioma) y recomienda las más similares.
    
    Attributes:
        movies_df (pd.DataFrame): DataFrame con información de películas
        preprocessor (DataPreprocessor): Procesador de datos
        similarity_matrix (np.ndarray): Matriz de similitud coseno
        movie_to_idx (Dict): Mapeo de movie_id a índice en matriz
        idx_to_movie (Dict): Mapeo de índice a movie_id
        vectorizer (CountVectorizer): Vectorizador entrenado
        is_fitted (bool): Indica si el modelo está entrenado
    """
    
    def __init__(self, preprocessor: Optional[DataPreprocessor] = None):
        """
        Inicializa el recomendador basado en contenido.
        
        Args:
            preprocessor: Instancia de DataPreprocessor (se crea una nueva si None)
        """
        self.movies_df = None
        self.similarity_matrix = None
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.vectorizer = None
        self.is_fitted = False
        
        self.preprocessor = preprocessor or DataPreprocessor()
        
        # Configuración
        self.config = get_config()
        self.models_dir = self.config['models_dir']
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ContentBasedRecommender inicializado")
    
    def fit(self, movies_df: pd.DataFrame, 
            force_recompute: bool = False,
            cache_name: str = "content_similarity") -> 'ContentBasedRecommender':
        """
        Entrena el recomendador: preprocesa datos y calcula matriz de similitud.
        
        Args:
            movies_df: DataFrame con datos de películas
            force_recompute: Si True, recalcula aunque exista caché
            cache_name: Nombre base para archivos de caché
        
        Returns:
            self (para encadenamiento de métodos)
        """
        logger.info("=== ENTRENANDO CONTENT-BASED RECOMMENDER ===")
        
        # Guardar DataFrame
        self.movies_df = movies_df.copy()
        
        # Preprocesar películas (crear combined_features)
        logger.info("Preprocesando películas...")
        self.movies_df = self.preprocessor.prepare_movies_for_similarity(self.movies_df)
        
        # Crear mapeos
        self.movie_to_idx, self.idx_to_movie = self.preprocessor.get_movie_index_mapping(
            self.movies_df
        )
        
        # Intentar cargar desde caché
        cache_path = self.models_dir / f"{cache_name}.joblib"
        vectorizer_path = self.models_dir / f"{cache_name}_vectorizer.joblib"
        
        if not force_recompute and cache_path.exists() and vectorizer_path.exists():
            logger.info(f"Cargando matriz de similitud desde caché: {cache_path}")
            try:
                self.similarity_matrix = joblib.load(cache_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.is_fitted = True
                logger.info(f"Matriz cargada: {self.similarity_matrix.shape}")
                return self
            except Exception as e:
                logger.warning(f"Error cargando caché: {e}. Recalculando...")
        
        # Crear vectorizador y matriz de características
        logger.info("Creando vectorizador y matriz de características...")
        combined_features = self.movies_df['combined_features']
        
        self.vectorizer = self.preprocessor.create_content_vectorizer(combined_features)
        feature_matrix = self.vectorizer.transform(combined_features.fillna(''))
        
        logger.info(f"Matriz de características creada: {feature_matrix.shape}")
        
        # Calcular similitud coseno
        logger.info("Calculando similitud coseno entre todas las películas...")
        logger.info("Esto puede tomar varios minutos dependiendo del tamaño del dataset...")
        
        self.similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)
        
        logger.info(f"Similitud calculada: {self.similarity_matrix.shape}")
        logger.info(f"Rango de similitud: [{self.similarity_matrix.min():.4f}, {self.similarity_matrix.max():.4f}]")
        
        # Guardar en caché
        logger.info(f"Guardando matriz en caché: {cache_path}")
        joblib.dump(self.similarity_matrix, cache_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        self.is_fitted = True
        logger.info("ContentBasedRecommender entrenado exitosamente")
        
        return self
    
    def get_similar_movies(self, 
                          movie_id: int, 
                          top_n: int = 10,
                          exclude_self: bool = True) -> List[Tuple[int, float]]:
        """
        Obtiene películas similares a una dada.
        
        Args:
            movie_id: ID de la película de referencia
            top_n: Número de películas a recomendar
            exclude_self: Si es True, excluye la película misma de resultados
        
        Returns:
            Lista de tuplas (movie_id, similarity_score)
        
        Raises:
            MovieNotFoundError: Si la película no existe en el dataset
        """
        if not self.is_fitted:
            raise RuntimeError("El recomendador no ha sido entrenado. Llama a fit() primero.")
        
        # Verificar que la película existe
        if movie_id not in self.movie_to_idx:
            available_movies = list(self.movie_to_idx.keys())[:10]
            raise MovieNotFoundError(
                f"Película ID {movie_id} no encontrada. "
                f"IDs disponibles (primeros 10): {available_movies}"
            )
        
        # Obtener índice de la película
        idx = self.movie_to_idx[movie_id]
        
        # Obtener scores de similitud para esta película
        similarity_scores = self.similarity_matrix[idx]
        
        # Crear lista de (índice, score)
        movie_scores = [(i, score) for i, score in enumerate(similarity_scores)]
        
        # Ordenar por score descendente
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Excluir la misma película si se pide
        if exclude_self:
            movie_scores = [m for m in movie_scores if m[0] != idx]
        
        # Tomar top_n
        top_movies = movie_scores[:top_n]
        
        # Convertir índices a movie_id
        result = [(self.idx_to_movie[idx], score) for idx, score in top_movies]
        
        logger.info(f"Encontradas {len(result)} películas similares a movie_id={movie_id}")
        
        return result
    
    def recommend_for_user_history(self, 
                                   watched_movies: List[int],
                                   ratings: Optional[List[float]] = None,
                                   top_n: int = 10,
                                   decay_factor: float = 0.8,
                                   low_rating_penalty: float = 0.1) -> List[Tuple[int, float]]:
        """
        Recomienda películas basadas en el historial de un usuario.
        
        Combina las similitudes de múltiples películas vistas, ponderando por rating.
        Las películas con baja calificación (1-2⭐) tienen peso reducido.
        
        Args:
            watched_movies: Lista de IDs de películas vistas/calificadas
            ratings: Lista de calificaciones correspondientes (1-5)
            top_n: Número de recomendaciones a retornar
            decay_factor: Factor de decaimiento para películas más antiguas (no usado actualmente)
            low_rating_penalty: Penalización para películas con rating bajo (1-2⭐)
        
        Returns:
            Lista de tuplas (movie_id, score) con scores normalizados [0,1]
        """
        if not watched_movies:
            logger.warning("No hay películas en el historial del usuario")
            return []
        
        # Si no hay ratings, asumir rating 3 (neutro)
        if ratings is None:
            ratings = [3.0] * len(watched_movies)
        
        # Diccionario para acumular scores
        aggregated_scores = {}
        
        # Para cada película vista, obtener sus similares
        for movie_id, rating in zip(watched_movies, ratings):
            try:
                similar_movies = self.get_similar_movies(movie_id, top_n=50, exclude_self=True)
                
                # Calcular peso según rating
                # Rating 1-2: penalizados, Rating 3: neutral, Rating 4-5: potenciados
                if rating <= 2:
                    # Penalizar ratings bajos
                    weight_multiplier = low_rating_penalty
                    logger.debug(f"Película {movie_id} con rating {rating}⭐ - penalizada (factor {low_rating_penalty})")
                elif rating >= 4:
                    # Potenciar ratings altos
                    weight_multiplier = 1.2
                    logger.debug(f"Película {movie_id} con rating {rating}⭐ - potenciada")
                else:
                    # Rating 3 es neutral
                    weight_multiplier = 1.0
                
                # Normalizar rating a [0,1] para ponderación
                normalized_rating = (rating - 1) / 4 if rating > 1 else 0
                base_weight = normalized_rating * weight_multiplier
                
                for sim_movie_id, similarity_score in similar_movies:
                    weight = base_weight * similarity_score
                    
                    if sim_movie_id in aggregated_scores:
                        aggregated_scores[sim_movie_id] += weight
                    else:
                        aggregated_scores[sim_movie_id] = weight
                        
            except MovieNotFoundError as e:
                logger.warning(f"Película {movie_id} no encontrada: {e}")
                continue
        
        # Excluir películas ya vistas/calificadas
        for watched_id in watched_movies:
            aggregated_scores.pop(watched_id, None)
        
        if not aggregated_scores:
            logger.warning("No se generaron recomendaciones basadas en contenido")
            return []
        
        # Normalizar scores a [0, 1]
        max_score = max(aggregated_scores.values())
        min_score = min(aggregated_scores.values())
        
        if max_score > min_score:
            normalized_scores = {m: (s - min_score) / (max_score - min_score) for m, s in aggregated_scores.items()}
        else:
            normalized_scores = {m: 0.5 for m in aggregated_scores}
        
        # Ordenar por score acumulado
        sorted_movies = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Tomar top_n
        recommendations = sorted_movies[:top_n]
        
        logger.info(f"Contenido generó {len(recommendations)} recomendaciones")
        if recommendations:
            logger.info(f"Rango de scores: [{recommendations[-1][1]:.3f}, {recommendations[0][1]:.3f}]")
        
        return recommendations
    
    def get_movie_details(self, movie_id: int) -> Dict[str, Any]:
        """Obtiene detalles de una película por su ID."""
        if self.movies_df is None:
            return {}
        
        movie_row = self.movies_df[self.movies_df['movie_id'] == movie_id]
        
        if movie_row.empty:
            return {}
        
        movie = movie_row.iloc[0].to_dict()
        
        # Manejar géneros
        genres = movie.get('genres_processed', [])
        if hasattr(genres, 'tolist'):
            genres = genres.tolist()
        elif genres is None:
            genres = []
        
        # Manejar actores
        actors = movie.get('actors_processed', [])
        if hasattr(actors, 'tolist'):
            actors = actors.tolist()
        elif actors is None:
            actors = []
        
        result = {
            'movie_id': movie.get('movie_id'),
            'title': movie.get('title_clean', movie.get('title', 'Desconocido')),
            'year': movie.get('year', 0),
            'genres': genres,
            'actors': actors[:3],
            'director': movie.get('director_processed', 'Desconocido'),
            'poster_path': movie.get('poster_path', '')
        }
        
        return result
    
    def get_similarity_score(self, movie_id_1: int, movie_id_2: int) -> float:
        """
        Obtiene la similitud entre dos películas específicas.
        
        Args:
            movie_id_1: Primera película
            movie_id_2: Segunda película
        
        Returns:
            Score de similitud (0 a 1)
        """
        if not self.is_fitted:
            raise RuntimeError("El recomendador no ha sido entrenado")
        
        if movie_id_1 not in self.movie_to_idx or movie_id_2 not in self.movie_to_idx:
            return 0.0
        
        idx1 = self.movie_to_idx[movie_id_1]
        idx2 = self.movie_to_idx[movie_id_2]
        
        return float(self.similarity_matrix[idx1, idx2])
    
    @safe_execute(logger, default_return=[])
    def search_movies(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Busca películas por título (búsqueda difusa).
        
        Args:
            query: Texto a buscar
            limit: Número máximo de resultados
        
        Returns:
            Lista de diccionarios con películas encontradas
        """
        if self.movies_df is None:
            return []
        
        # Limpiar query
        query_clean = self.preprocessor.clean_title(query).lower()
        
        # Buscar coincidencias en título limpio
        mask = self.movies_df['title_clean'].str.lower().str.contains(query_clean, na=False)
        results_df = self.movies_df[mask].head(limit)
        
        # También buscar en título original si existe
        if len(results_df) < limit and 'title' in self.movies_df.columns:
            mask_orig = self.movies_df['title'].str.lower().str.contains(query_clean, na=False)
            results_df = pd.concat([results_df, self.movies_df[mask_orig]]).drop_duplicates().head(limit)
        
        # Convertir a lista de diccionarios
        results = []
        for _, row in results_df.iterrows():
            results.append({
                'movie_id': row.get('movie_id'),
                'title': row.get('title_clean', row.get('title', 'Desconocido')),
                'year': row.get('year', 0),
                'genres': row.get('genres_processed', [])
            })
        
        logger.info(f"Búsqueda '{query}' encontró {len(results)} películas")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del recomendador.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.is_fitted:
            return {'is_fitted': False}
        
        stats = {
            'is_fitted': True,
            'num_movies': len(self.movies_df) if self.movies_df is not None else 0,
            'similarity_matrix_shape': self.similarity_matrix.shape if self.similarity_matrix is not None else (0, 0),
            'num_features': len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 0,
            'avg_similarity': float(np.mean(self.similarity_matrix)) if self.similarity_matrix is not None else 0,
            'max_similarity': float(np.max(self.similarity_matrix)) if self.similarity_matrix is not None else 0,
            'min_similarity': float(np.min(self.similarity_matrix)) if self.similarity_matrix is not None else 0
        }
        
        return stats
