"""
Módulo de recomendación colaborativa.

Implementa:
- Filtro colaborativo basado en usuarios (User-Based Collaborative Filtering)
- Matriz usuario-película con ratings explícitos (1-5)
- Cálculo de similitud entre usuarios usando correlación de Pearson o coseno
- Predicción de ratings para películas no vistas
- Recomendaciones personalizadas para cualquier usuario
- Cacheo de matriz de similitud de usuarios
- Soporte para sistema de calificaciones ⭐1-5
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Set
from pathlib import Path
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import correlation
from collections import defaultdict

# Importar utilidades
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    setup_logging, safe_execute, InsufficientDataError, 
    MovieNotFoundError, get_config
)
from src.preprocess import DataPreprocessor

logger = setup_logging(log_to_ui=True)


class CollaborativeRecommender:
    """
    Recomendador colaborativo basado en usuarios.
    
    Utiliza la matriz usuario-película para encontrar usuarios similares
    y recomendar películas que gustaron a esos usuarios similares.
    
    Attributes:
        user_movie_matrix (pd.DataFrame): Matriz de ratings (usuarios x películas)
        user_similarity_matrix (np.ndarray): Matriz de similitud entre usuarios
        user_to_idx (Dict): Mapeo de user_id a índice en matriz
        idx_to_user (Dict): Mapeo de índice a user_id
        movie_to_idx (Dict): Mapeo de movie_id a índice en matriz
        idx_to_movie (Dict): Mapeo de índice a movie_id
        preprocessor (DataPreprocessor): Procesador de datos
        is_fitted (bool): Indica si el modelo está entrenado
        similarity_metric (str): Métrica de similitud ('cosine' o 'pearson')
    """
    
    def __init__(self, 
                 preprocessor: Optional[DataPreprocessor] = None,
                 similarity_metric: str = 'cosine',
                 min_common_movies: int = 3):
        """
        Inicializa el recomendador colaborativo.
        
        Args:
            preprocessor: Instancia de DataPreprocessor (se crea una nueva si None)
            similarity_metric: Métrica de similitud ('cosine' o 'pearson')
            min_common_movies: Mínimo de películas en común para calcular similitud
        """
        self.user_movie_matrix = None
        self.user_similarity_matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.is_fitted = False
        
        self.preprocessor = preprocessor or DataPreprocessor()
        self.similarity_metric = similarity_metric
        self.min_common_movies = min_common_movies
        
        # Configuración
        self.config = get_config()
        self.models_dir = self.config['models_dir']
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CollaborativeRecommender inicializado (métrica: {similarity_metric})")
    
    def fit(self, 
            ratings_df: pd.DataFrame,
            movies_df: pd.DataFrame,
            force_recompute: bool = False,
            cache_name: str = "collaborative_similarity") -> 'CollaborativeRecommender':
        """
        Entrena el recomendador colaborativo.
        
        Args:
            ratings_df: DataFrame con columnas ['user_id', 'movie_id', 'rating']
            movies_df: DataFrame con información de películas (para mapeos)
            force_recompute: Si True, recalcula aunque exista caché
            cache_name: Nombre base para archivos de caché
        
        Returns:
            self (para encadenamiento de métodos)
        """
        logger.info("=== ENTRENANDO COLLABORATIVE RECOMMENDER ===")
        
        # Filtrar ratings (usuarios y películas con pocos datos)
        logger.info("Filtrando ratings...")
        ratings_filtered = self.preprocessor.filter_ratings(ratings_df)
        
        # Crear matriz usuario-película
        logger.info("Creando matriz usuario-película...")
        self.user_movie_matrix = self.preprocessor.create_user_movie_matrix(ratings_filtered)
        
        # Crear mapeos de usuarios
        self.user_to_idx, self.idx_to_user = self.preprocessor.get_user_index_mapping(
            self.user_movie_matrix
        )
        
        # Crear mapeos de películas (desde la matriz)
        movies_in_matrix = self.user_movie_matrix.columns.tolist()
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(movies_in_matrix)}
        self.idx_to_movie = {idx: movie for idx, movie in enumerate(movies_in_matrix)}
        
        logger.info(f"Matriz creada: {len(self.user_to_idx)} usuarios, "
                   f"{len(self.movie_to_idx)} películas")
        
        # Intentar cargar desde caché
        cache_path = self.models_dir / f"{cache_name}.joblib"
        
        if not force_recompute and cache_path.exists():
            logger.info(f"Cargando matriz de similitud de usuarios desde caché: {cache_path}")
            try:
                self.user_similarity_matrix = joblib.load(cache_path)
                self.is_fitted = True
                logger.info(f"Matriz cargada: {self.user_similarity_matrix.shape}")
                return self
            except Exception as e:
                logger.warning(f"Error cargando caché: {e}. Recalculando...")
        
        # Calcular similitud entre usuarios
        logger.info(f"Calculando similitud entre {len(self.user_to_idx)} usuarios...")
        logger.info("Esto puede tomar varios minutos para datasets grandes...")
        
        if self.similarity_metric == 'cosine':
            self.user_similarity_matrix = cosine_similarity(self.user_movie_matrix.values)
        elif self.similarity_metric == 'pearson':
            self.user_similarity_matrix = self._pearson_correlation_matrix(
                self.user_movie_matrix.values
            )
        else:
            raise ValueError(f"Métrica no soportada: {self.similarity_metric}")
        
        # Aplicar máscara para usuarios con poca similitud
        self._apply_similarity_threshold()
        
        logger.info(f"Similitud calculada: {self.user_similarity_matrix.shape}")
        logger.info(f"Rango de similitud: [{self.user_similarity_matrix.min():.4f}, "
                   f"{self.user_similarity_matrix.max():.4f}]")
        
        # Guardar en caché
        logger.info(f"Guardando matriz en caché: {cache_path}")
        joblib.dump(self.user_similarity_matrix, cache_path)
        
        self.is_fitted = True
        logger.info("CollaborativeRecommender entrenado exitosamente")
        
        return self
    
    def _pearson_correlation_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de correlación de Pearson de forma eficiente.
        
        Args:
            matrix: Matriz de ratings (usuarios x items)
        
        Returns:
            Matriz de correlaciones
        """
        n_users = matrix.shape[0]
        correlation_matrix = np.zeros((n_users, n_users))
        
        for i in range(n_users):
            # Calcular correlación con todos los usuarios de una vez es costoso
            # Hacemos pairwise con optimización
            for j in range(i, n_users):
                # Encontrar items que ambos usuarios han rating
                mask = ~(np.isnan(matrix[i]) | np.isnan(matrix[j]))
                
                if mask.sum() < self.min_common_movies:
                    corr = 0.0
                else:
                    # Calcular correlación de Pearson
                    user_i = matrix[i][mask]
                    user_j = matrix[j][mask]
                    
                    if len(user_i) == 0 or len(user_j) == 0:
                        corr = 0.0
                    else:
                        # Pearson correlation
                        corr = np.corrcoef(user_i, user_j)[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        return correlation_matrix
    
    def _apply_similarity_threshold(self, threshold: float = 0.1):
        """
        Aplica un umbral a la matriz de similitud (valores negativos a 0).
        
        Args:
            threshold: Umbral mínimo de similitud
        """
        # Valores negativos no son útiles (usuarios disimilares)
        self.user_similarity_matrix = np.maximum(self.user_similarity_matrix, 0)
        
        # Opcional: aplicar threshold más agresivo
        # self.user_similarity_matrix[self.user_similarity_matrix < threshold] = 0
        
        logger.info(f"Similitud positiva: {(self.user_similarity_matrix > 0).sum() / self.user_similarity_matrix.size * 100:.1f}% de pares")
    
    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predice el rating que un usuario daría a una película.
        
        Usa el promedio ponderado de los ratings de usuarios similares.
        
        Args:
            user_id: ID del usuario
            movie_id: ID de la película
        
        Returns:
            Rating predicho (1-5), 0 si no se puede predecir
        """
        if not self.is_fitted:
            raise RuntimeError("El recomendador no ha sido entrenado.")
        
        # Verificar que el usuario existe
        if user_id not in self.user_to_idx:
            logger.debug(f"Usuario {user_id} no encontrado en la matriz")
            return 0.0
        
        # Verificar que la película existe en la matriz
        if movie_id not in self.movie_to_idx:
            logger.debug(f"Película {movie_id} no encontrada en la matriz")
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        movie_idx = self.movie_to_idx[movie_id]
        
        # Si el usuario ya ha rating esta película, devolver el rating real
        if self.user_movie_matrix.iloc[user_idx, movie_idx] > 0:
            return float(self.user_movie_matrix.iloc[user_idx, movie_idx])
        
        # Obtener similitudes de este usuario con todos los demás
        similarities = self.user_similarity_matrix[user_idx]
        
        # Obtener ratings de la película de otros usuarios
        movie_ratings = self.user_movie_matrix.iloc[:, movie_idx].values
        
        # Calcular promedio ponderado
        numerator = 0.0
        denominator = 0.0
        
        for other_idx, sim in enumerate(similarities):
            if other_idx != user_idx and sim > 0:
                rating = movie_ratings[other_idx]
                if rating > 0:  # Solo si el otro usuario ha rating esta película
                    numerator += sim * rating
                    denominator += sim
        
        if denominator == 0:
            # Fallback: promedio global de la película
            global_avg = movie_ratings[movie_ratings > 0].mean()
            return float(global_avg) if not np.isnan(global_avg) else 3.0
        
        predicted_rating = numerator / denominator
        
        # Redondear a rango válido
        predicted_rating = max(1.0, min(5.0, predicted_rating))
        
        return float(predicted_rating)
    
    def recommend_for_user(self,
                          user_id: int,
                          top_n: int = 10,
                          exclude_watched: bool = True,
                          min_rating_threshold: float = 2.5,
                          low_rating_penalty: float = 0.1) -> List[Tuple[int, float]]:
        """
        Recomienda películas para un usuario específico.
        
        Args:
            user_id: ID del usuario
            top_n: Número de recomendaciones
            exclude_watched: Si True, excluye películas ya vistas/calificadas
            min_rating_threshold: Umbral mínimo de rating predicho
            low_rating_penalty: Penalización para películas con rating bajo
                               (nota: el colaborativo no conoce ratings bajos del usuario,
                                esta penalización se aplica externamente en feedback.py)
        
        Returns:
            Lista de tuplas (movie_id, score) con scores normalizados [0,1]
        """
        if not self.is_fitted:
            raise RuntimeError("El recomendador no ha sido entrenado.")
        
        if user_id not in self.user_to_idx:
            logger.warning(f"Usuario {user_id} no encontrado. Devolviendo lista vacía.")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # Obtener todas las películas en la matriz
        all_movies = list(self.movie_to_idx.keys())
        
        # Excluir películas ya vistas si se pide
        if exclude_watched:
            watched_movies = self.user_movie_matrix.iloc[user_idx]
            watched_movies = watched_movies[watched_movies > 0].index.tolist()
            movies_to_predict = [m for m in all_movies if m not in watched_movies]
        else:
            movies_to_predict = all_movies
        
        # Predecir rating para cada película
        predictions = []
        for movie_id in movies_to_predict:
            predicted_rating = self.predict_rating(user_id, movie_id)
            if predicted_rating >= min_rating_threshold:
                predictions.append((movie_id, predicted_rating))
        
        if not predictions:
            logger.info(f"No se generaron predicciones colaborativas para usuario {user_id}")
            return []
        
        # Normalizar ratings a scores entre 0 y 1
        # Los ratings originales están entre 1 y 5
        # Convertimos a [0,1] con (rating - 1) / 4
        normalized_predictions = [(m, (r - 1) / 4) for m, r in predictions]
        
        # Limitar scores al rango [0,1]
        normalized_predictions = [(m, max(0.0, min(1.0, s))) for m, s in normalized_predictions]
        
        # Ordenar por rating predicho descendente
        normalized_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar top_n
        recommendations = normalized_predictions[:top_n]
        
        logger.info(f"Colaborativo generó {len(recommendations)} recomendaciones para usuario {user_id}")
        if recommendations:
            logger.info(f"Rango de scores: [{recommendations[-1][1]:.3f}, {recommendations[0][1]:.3f}]")
        
        return recommendations
    
    def get_similar_users(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Encuentra usuarios similares a uno dado.
        
        Args:
            user_id: ID del usuario
            top_n: Número de usuarios similares a devolver
        
        Returns:
            Lista de tuplas (user_id, similarity_score)
        """
        if not self.is_fitted:
            raise RuntimeError("El recomendador no ha sido entrenado.")
        
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        similarities = self.user_similarity_matrix[user_idx]
        
        # Crear lista de (usuario, similitud)
        user_scores = [(self.idx_to_user[i], score) 
                       for i, score in enumerate(similarities) 
                       if i != user_idx and score > 0]
        
        # Ordenar por similitud descendente
        user_scores.sort(key=lambda x: x[1], reverse=True)
        
        return user_scores[:top_n]
    
    def get_user_stats(self, user_id: int) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un usuario.
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Diccionario con estadísticas del usuario
        """
        if not self.is_fitted or user_id not in self.user_to_idx:
            return {}
        
        user_idx = self.user_to_idx[user_id]
        user_ratings = self.user_movie_matrix.iloc[user_idx]
        rated_movies = user_ratings[user_ratings > 0]
        
        stats = {
            'user_id': user_id,
            'num_ratings': len(rated_movies),
            'avg_rating': float(rated_movies.mean()) if len(rated_movies) > 0 else 0,
            'min_rating': float(rated_movies.min()) if len(rated_movies) > 0 else 0,
            'max_rating': float(rated_movies.max()) if len(rated_movies) > 0 else 0,
            'rating_std': float(rated_movies.std()) if len(rated_movies) > 1 else 0,
            'favorite_movies': rated_movies.nlargest(5).index.tolist()
        }
        
        return stats
    
    def get_movie_recommendations_from_users(self,
                                             movie_id: int,
                                             top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Encuentra usuarios que dieron alta puntuación a una película 
        y recomienda otras películas que esos usuarios disfrutaron.
        
        Args:
            movie_id: ID de la película de referencia
            top_n: Número de recomendaciones
        
        Returns:
            Lista de tuplas (movie_id, aggregated_score)
        """
        if not self.is_fitted:
            raise RuntimeError("El recomendador no ha sido entrenado.")
        
        if movie_id not in self.movie_to_idx:
            return []
        
        movie_idx = self.movie_to_idx[movie_id]
        
        # Encontrar usuarios que dieron rating alto a esta película (>=4)
        high_ratings = self.user_movie_matrix.iloc[:, movie_idx]
        users_who_liked = high_ratings[high_ratings >= 4].index.tolist()
        
        if not users_who_liked:
            return []
        
        # Agregar recomendaciones de esos usuarios
        movie_scores = defaultdict(float)
        
        for user_id in users_who_liked:
            user_idx = self.user_to_idx[user_id]
            user_ratings = self.user_movie_matrix.iloc[user_idx]
            
            # Obtener películas que este usuario ha rating alto
            high_user_ratings = user_ratings[user_ratings >= 4]
            
            for other_movie_id, rating in high_user_ratings.items():
                if other_movie_id != movie_id:
                    movie_scores[other_movie_id] += rating
        
        # Convertir a lista y ordenar
        recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        return recommendations[:top_n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del recomendador colaborativo.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.is_fitted:
            return {'is_fitted': False}
        
        stats = {
            'is_fitted': True,
            'num_users': len(self.user_to_idx),
            'num_movies': len(self.movie_to_idx),
            'matrix_shape': self.user_movie_matrix.shape,
            'sparsity': self._calculate_sparsity(),
            'avg_similarity': float(np.mean(self.user_similarity_matrix)) if self.user_similarity_matrix is not None else 0,
            'max_similarity': float(np.max(self.user_similarity_matrix)) if self.user_similarity_matrix is not None else 0,
            'similarity_metric': self.similarity_metric
        }
        
        return stats
    
    def _calculate_sparsity(self) -> float:
        """
        Calcula la esparsidad de la matriz usuario-película.
        
        Returns:
            Porcentaje de celdas con rating (0 a 100)
        """
        total_cells = self.user_movie_matrix.shape[0] * self.user_movie_matrix.shape[1]
        non_zero_cells = (self.user_movie_matrix.values > 0).sum()
        
        sparsity = (non_zero_cells / total_cells) * 100
        
        logger.info(f"Esparsidad de matriz: {sparsity:.2f}% de celdas con ratings")
        
        return float(sparsity)
