"""
Módulo de recomendación híbrida.

Implementa:
- Combinación ponderada de recomendaciones de contenido y colaborativo
- Ponderación dinámica según disponibilidad de datos
- Configuración de pesos por parte del usuario
- Normalización de scores entre diferentes métodos
- Estrategias de combinación (promedio, máximo, mínimos cuadrados)
- Soporte para sistema de penalización por ratings bajos
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from collections import defaultdict

# Importar utilidades
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    setup_logging, safe_execute, RecommenderError,
    MovieNotFoundError, InsufficientDataError, get_config
)
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender

logger = setup_logging(log_to_ui=True)


class HybridRecommender:
    """
    Recomendador híbrido que combina contenido y colaborativo.
    
    Estrategias de combinación:
    - weighted: Promedio ponderado con pesos configurables
    - adaptive: Peso dinámico según datos disponibles del usuario
    - max: Toma el máximo score de ambos métodos
    - min_max: Normaliza scores antes de combinar
    
    Attributes:
        content_recommender (ContentBasedRecommender): Recomendador de contenido
        collab_recommender (CollaborativeRecommender): Recomendador colaborativo
        content_weight (float): Peso para recomendaciones de contenido (0-1)
        collab_weight (float): Peso para recomendaciones colaborativas (0-1)
        strategy (str): Estrategia de combinación
        use_fallback (bool): Usar solo contenido si colaborativo falla
    """
    
    def __init__(self,
                 content_recommender: ContentBasedRecommender,
                 collab_recommender: Optional[CollaborativeRecommender] = None,
                 content_weight: float = 0.6,
                 collab_weight: float = 0.4,
                 strategy: str = 'weighted',
                 use_fallback: bool = True):
        """
        Inicializa el recomendador híbrido.
        
        Args:
            content_recommender: Instancia entrenada de ContentBasedRecommender
            collab_recommender: Instancia entrenada de CollaborativeRecommender
            content_weight: Peso para contenido (0-1)
            collab_weight: Peso para colaborativo (0-1)
            strategy: Estrategia ('weighted', 'adaptive', 'max', 'min_max')
            use_fallback: Si True, usa solo contenido si colaborativo no disponible
        """
        self.content_recommender = content_recommender
        self.collab_recommender = collab_recommender
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.strategy = strategy
        self.use_fallback = use_fallback
        
        # Normalizar pesos si no suman 1
        total = content_weight + collab_weight
        if total > 0:
            self.content_weight = content_weight / total
            self.collab_weight = collab_weight / total
        else:
            self.content_weight = 0.5
            self.collab_weight = 0.5
        
        # Verificar si los recomendadores están listos
        content_ready = self.content_recommender.is_fitted if self.content_recommender else False
        collab_ready = self.collab_recommender.is_fitted if self.collab_recommender else False
        
        self.is_ready = content_ready  # Al menos contenido debe estar listo
        
        # Configuración
        self.config = get_config()
        
        logger.info(f"HybridRecommender inicializado: "
                   f"pesos (contenido={self.content_weight:.2f}, "
                   f"colaborativo={self.collab_weight:.2f}), "
                   f"estrategia={strategy}, "
                   f"content_ready={content_ready}, "
                   f"collab_ready={collab_ready}")
    
    def _check_ready(self) -> bool:
        """Verifica si los recomendadores están listos."""
        content_ready = self.content_recommender.is_fitted
        
        collab_ready = False
        if self.collab_recommender:
            collab_ready = self.collab_recommender.is_fitted
        
        ready = content_ready
        if not content_ready:
            logger.warning("Recomendador de contenido no está entrenado")
        
        if not collab_ready and self.collab_recommender:
            logger.warning("Recomendador colaborativo no está entrenado")
        
        if not collab_ready and not self.use_fallback:
            logger.warning("Modo híbrido sin colaborativo y sin fallback")
        
        return ready
    
    def set_weights(self, content_weight: float, collab_weight: float):
        """
        Actualiza los pesos de la combinación híbrida.
        
        Args:
            content_weight: Nuevo peso para contenido (0-1)
            collab_weight: Nuevo peso para colaborativo (0-1)
        """
        total = content_weight + collab_weight
        if total > 0:
            self.content_weight = content_weight / total
            self.collab_weight = collab_weight / total
        else:
            self.content_weight = 0.5
            self.collab_weight = 0.5
        
        logger.info(f"Pesos actualizados: contenido={self.content_weight:.2f}, "
                   f"colaborativo={self.collab_weight:.2f}")
    
    def set_strategy(self, strategy: str):
        """
        Cambia la estrategia de combinación.
        
        Args:
            strategy: 'weighted', 'adaptive', 'max', 'min_max'
        """
        valid_strategies = ['weighted', 'adaptive', 'max', 'min_max']
        if strategy not in valid_strategies:
            raise ValueError(f"Estrategia inválida. Debe ser una de: {valid_strategies}")
        
        self.strategy = strategy
        logger.info(f"Estrategia cambiada a: {strategy}")
    
    def _normalize_scores(self, 
                          scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Normaliza scores al rango [0, 1] usando min-max.
        
        Args:
            scores: Lista de (movie_id, score)
        
        Returns:
            Lista con scores normalizados
        """
        if not scores:
            return scores
        
        min_score = min(score for _, score in scores)
        max_score = max(score for _, score in scores)
        
        if max_score == min_score:
            return [(movie_id, 0.5) for movie_id, _ in scores]
        
        normalized = [(movie_id, (score - min_score) / (max_score - min_score))
                     for movie_id, score in scores]
        
        return normalized
    
    def _combine_weighted(self,
                          content_scores: Dict[int, float],
                          collab_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """
        Combina scores usando promedio ponderado.
        
        Args:
            content_scores: Dict de movie_id -> score de contenido
            collab_scores: Dict de movie_id -> score colaborativo
        
        Returns:
            Lista de (movie_id, combined_score)
        """
        all_movies = set(content_scores.keys()) | set(collab_scores.keys())
        combined = {}
        
        for movie_id in all_movies:
            content_score = content_scores.get(movie_id, 0)
            collab_score = collab_scores.get(movie_id, 0)
            
            combined_score = (self.content_weight * content_score + 
                            self.collab_weight * collab_score)
            
            if combined_score > 0:
                combined[movie_id] = combined_score
        
        # Ordenar por score descendente
        result = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        return result
    
    def _combine_adaptive(self,
                         content_scores: Dict[int, float],
                         collab_scores: Dict[int, float],
                         user_rating_count: int = 0) -> List[Tuple[int, float]]:
        """
        Combina scores con pesos adaptativos según datos del usuario.
        
        Usuarios con más ratings reciben más peso colaborativo.
        
        Args:
            content_scores: Dict de movie_id -> score de contenido
            collab_scores: Dict de movie_id -> score colaborativo
            user_rating_count: Número de ratings del usuario
        
        Returns:
            Lista de (movie_id, combined_score)
        """
        # Calcular peso adaptativo
        # Si usuario tiene >20 ratings, peso colaborativo = 0.7
        # Si tiene <5 ratings, peso colaborativo = 0.2
        max_ratings = 20
        min_ratings = 5
        
        if user_rating_count >= max_ratings:
            adaptive_collab_weight = 0.7
        elif user_rating_count <= min_ratings:
            adaptive_collab_weight = 0.2
        else:
            # Interpolación lineal
            ratio = (user_rating_count - min_ratings) / (max_ratings - min_ratings)
            adaptive_collab_weight = 0.2 + ratio * 0.5
        
        adaptive_content_weight = 1 - adaptive_collab_weight
        
        # Aplicar pesos
        all_movies = set(content_scores.keys()) | set(collab_scores.keys())
        combined = {}
        
        for movie_id in all_movies:
            content_score = content_scores.get(movie_id, 0)
            collab_score = collab_scores.get(movie_id, 0)
            
            combined_score = (adaptive_content_weight * content_score + 
                            adaptive_collab_weight * collab_score)
            
            if combined_score > 0:
                combined[movie_id] = combined_score
        
        result = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Pesos adaptativos: contenido={adaptive_content_weight:.2f}, "
                    f"colaborativo={adaptive_collab_weight:.2f}")
        
        return result
    
    def _combine_max(self,
                    content_scores: Dict[int, float],
                    collab_scores: Dict[int, float]) -> List[Tuple[int, float]]:
        """
        Toma el máximo score entre ambos métodos.
        
        Args:
            content_scores: Dict de movie_id -> score de contenido
            collab_scores: Dict de movie_id -> score colaborativo
        
        Returns:
            Lista de (movie_id, max_score)
        """
        all_movies = set(content_scores.keys()) | set(collab_scores.keys())
        combined = {}
        
        for movie_id in all_movies:
            content_score = content_scores.get(movie_id, 0)
            collab_score = collab_scores.get(movie_id, 0)
            
            combined[movie_id] = max(content_score, collab_score)
        
        result = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        return result
    
    def recommend_for_movie(self,
                           movie_id: int,
                           top_n: int = 10,
                           exclude_self: bool = True) -> List[Tuple[int, float]]:
        """
        Recomienda películas similares usando solo contenido.
        (El híbrido para una película es igual al contenido)
        
        Args:
            movie_id: ID de la película base
            top_n: Número de recomendaciones
            exclude_self: Excluir la misma película
        
        Returns:
            Lista de (movie_id, similarity_score)
        """
        if not self.content_recommender.is_fitted:
            raise RuntimeError("Recomendador de contenido no entrenado")
        
        recommendations = self.content_recommender.get_similar_movies(
            movie_id, top_n=top_n, exclude_self=exclude_self
        )
        
        logger.info(f"Recomendaciones híbridas para película {movie_id}: {len(recommendations)}")
        
        return recommendations
    
    def recommend_for_user(self,
                          user_id: int,
                          watched_movies: List[int],
                          ratings: Optional[List[float]] = None,
                          top_n: int = 10,
                          content_top_n: int = 30,
                          collab_top_n: int = 30,
                          low_rating_penalty: float = 0.1) -> List[Tuple[int, float]]:
        """
        Recomienda películas para un usuario usando enfoque híbrido.
        
        Args:
            user_id: ID del usuario
            watched_movies: Lista de IDs de películas vistas/calificadas
            ratings: Lista de calificaciones correspondientes (1-5)
            top_n: Número de recomendaciones a retornar
            content_top_n: Número de recomendaciones a solicitar al recomendador de contenido
            collab_top_n: Número de recomendaciones a solicitar al recomendador colaborativo
            low_rating_penalty: Factor de penalización para películas con baja calificación
        
        Returns:
            Lista de tuplas (movie_id, score) con scores normalizados [0,1]
        """
        if not self.is_ready:
            raise RuntimeError("Recomendador no está listo. Asegúrate de que content_recommender esté entrenado.")
        
        logger.info(f"Generando recomendaciones híbridas para usuario {user_id}")
        
        # Si no hay ratings, usar lista vacía
        if ratings is None:
            ratings = []
        
        # 1. Obtener recomendaciones de contenido basadas en historial
        content_recs = []
        if self.content_recommender and self.content_recommender.is_fitted:
            try:
                content_recs = self.content_recommender.recommend_for_user_history(
                    watched_movies=watched_movies,
                    ratings=ratings if ratings else None,
                    top_n=content_top_n
                )
                logger.info(f"Contenido generó {len(content_recs)} recomendaciones")
            except Exception as e:
                logger.warning(f"Error en recomendación de contenido: {e}")
        
        # Convertir a dict para fácil acceso
        content_dict = {movie_id: score for movie_id, score in content_recs}
        
        # 2. Obtener recomendaciones colaborativas (si está disponible)
        collab_dict = {}
        collab_available = False
        
        if self.collab_recommender and self.collab_recommender.is_fitted:
            try:
                collab_recs = self.collab_recommender.recommend_for_user(
                    user_id=user_id,
                    top_n=collab_top_n,
                    exclude_watched=True
                )
                collab_dict = {movie_id: score for movie_id, score in collab_recs}
                collab_available = True
                logger.info(f"Colaborativo generó {len(collab_recs)} recomendaciones")
            except Exception as e:
                logger.warning(f"Error en recomendación colaborativa: {e}")
        
        # 3. Si no hay colaborativo y usamos fallback, solo contenido
        if not collab_available and self.use_fallback:
            logger.info("Usando solo contenido (fallback activado)")
            # Normalizar scores de contenido a [0,1]
            if content_recs:
                scores = [s for _, s in content_recs]
                max_score = max(scores) if scores else 1
                min_score = min(scores) if scores else 0
                if max_score > min_score:
                    content_recs = [(m, (s - min_score) / (max_score - min_score)) for m, s in content_recs]
                else:
                    content_recs = [(m, 0.5) for m, _ in content_recs]
            
            # Aplicar penalización por ratings bajos antes de retornar
            if ratings and watched_movies:
                low_rated = {watched_movies[i] for i, r in enumerate(ratings) if r <= 2}
                if low_rated:
                    content_recs = [(m, s * low_rating_penalty) for m, s in content_recs if m not in low_rated]
                    content_recs.sort(key=lambda x: x[1], reverse=True)
            
            return content_recs[:top_n]
        
        # 4. Normalizar scores de contenido a [0,1]
        if content_dict:
            max_content = max(content_dict.values())
            min_content = min(content_dict.values())
            if max_content > min_content:
                content_dict = {m: (s - min_content) / (max_content - min_content) for m, s in content_dict.items()}
            else:
                content_dict = {m: 0.5 for m in content_dict}
        
        # 5. Normalizar scores colaborativos a [0,1]
        if collab_dict:
            max_collab = max(collab_dict.values())
            min_collab = min(collab_dict.values())
            if max_collab > min_collab:
                collab_dict = {m: (s - min_collab) / (max_collab - min_collab) for m, s in collab_dict.items()}
            else:
                collab_dict = {m: 0.5 for m in collab_dict}
        
        # 6. Combinar según estrategia
        all_movies = set(content_dict.keys()) | set(collab_dict.keys())
        combined = {}
        
        for movie_id in all_movies:
            content_score = content_dict.get(movie_id, 0)
            collab_score = collab_dict.get(movie_id, 0)
            
            if self.strategy == 'weighted':
                combined_score = (self.content_weight * content_score + self.collab_weight * collab_score)
            elif self.strategy == 'adaptive':
                # Pesos adaptativos según cantidad de ratings del usuario
                user_rating_count = len(watched_movies)
                if user_rating_count >= 20:
                    adaptive_collab = 0.7
                elif user_rating_count <= 5:
                    adaptive_collab = 0.2
                else:
                    adaptive_collab = 0.2 + ((user_rating_count - 5) / 15) * 0.5
                adaptive_content = 1 - adaptive_collab
                combined_score = (adaptive_content * content_score + adaptive_collab * collab_score)
            elif self.strategy == 'max':
                combined_score = max(content_score, collab_score)
            else:
                combined_score = (self.content_weight * content_score + self.collab_weight * collab_score)
            
            if combined_score > 0:
                combined[movie_id] = combined_score
        
        # 7. Aplicar penalización por ratings bajos (1-2⭐)
        if ratings and watched_movies:
            # Identificar películas con baja calificación
            low_rated_movies = {watched_movies[i] for i, r in enumerate(ratings) if r <= 2}
            
            for movie_id in low_rated_movies:
                if movie_id in combined:
                    combined[movie_id] = combined[movie_id] * low_rating_penalty
                    logger.debug(f"Película {movie_id} penalizada por rating bajo")
        
        # 8. Excluir películas ya vistas/calificadas
        watched_set = set(watched_movies)
        combined = [(m, s) for m, s in combined.items() if m not in watched_set]
        
        # 9. Asegurar que los scores estén entre 0 y 1
        combined = [(m, max(0.0, min(1.0, s))) for m, s in combined]
        
        # 10. Ordenar y tomar top_n
        combined.sort(key=lambda x: x[1], reverse=True)
        recommendations = combined[:top_n]
        
        logger.info(f"Híbrido generó {len(recommendations)} recomendaciones finales")
        if recommendations:
            scores = [s for _, s in recommendations]
            logger.info(f"Rango de scores: [{min(scores):.3f}, {max(scores):.3f}]")
        
        return recommendations
    
    def get_recommendation_details(self, recommendations: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """
        Enriquece las recomendaciones con detalles de películas.
        """
        details = []
        
        for movie_id, score in recommendations:
            movie_details = self.content_recommender.get_movie_details(movie_id)
            
            if movie_details:
                # Manejar géneros
                genres = movie_details.get('genres', [])
                if hasattr(genres, 'tolist'):
                    genres = genres.tolist()
                elif genres is None:
                    genres = []
                
                # Manejar actores
                actors = movie_details.get('actors', [])
                if hasattr(actors, 'tolist'):
                    actors = actors.tolist()
                elif actors is None:
                    actors = []
                
                details.append({
                    'movie_id': movie_id,
                    'title': movie_details.get('title', 'Desconocido'),
                    'year': movie_details.get('year', 0),
                    'genres': genres,
                    'actors': actors,
                    'director': movie_details.get('director', ''),
                    'poster_path': movie_details.get('poster_path', ''),
                    'score': round(score, 4),
                    'score_percentage': round(score * 100, 2)
                })
        
        return details
    
    def explain_recommendation(self,
                               user_id: int,
                               movie_id: int,
                               watched_movies: List[int],
                               ratings: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Explica por qué se recomendó una película.
        
        Args:
            user_id: ID del usuario
            movie_id: ID de la película recomendada
            watched_movies: Lista de películas que el usuario ha visto/calificado
            ratings: Lista de calificaciones correspondientes
        
        Returns:
            Diccionario con explicación detallada
        """
        explanation = {
            'movie_id': movie_id,
            'movie_details': self.content_recommender.get_movie_details(movie_id),
            'content_based_reasons': [],
            'collaborative_reasons': [],
            'blended_score': 0.0,
            'penalty_applied': False
        }
        
        # 1. Razones basadas en contenido
        for idx, watched_id in enumerate(watched_movies[:5]):  # Top 5 películas vistas
            similarity = self.content_recommender.get_similarity_score(watched_id, movie_id)
            if similarity > 0.3:
                watched_details = self.content_recommender.get_movie_details(watched_id)
                rating_info = ""
                if ratings and idx < len(ratings):
                    rating = ratings[idx]
                    if rating <= 2:
                        rating_info = f" (calificaste con {rating}⭐ - bajo)"
                    elif rating >= 4:
                        rating_info = f" (calificaste con {rating}⭐ - alto)"
                
                explanation['content_based_reasons'].append({
                    'watched_movie': watched_details.get('title', ''),
                    'similarity_score': round(similarity, 3),
                    'your_rating': ratings[idx] if ratings and idx < len(ratings) else None,
                    'rating_info': rating_info
                })
        
        # 2. Razones colaborativas (si disponible)
        if self.collab_recommender and self.collab_recommender.is_fitted:
            predicted_rating = self.collab_recommender.predict_rating(user_id, movie_id)
            if predicted_rating > 0:
                explanation['collaborative_reasons'] = {
                    'predicted_rating': round(predicted_rating, 2),
                    'explanation': f"Usuarios similares califican esta película con {predicted_rating:.1f}/5"
                }
        
        # 3. Verificar si hay penalización por rating bajo
        if ratings:
            for watched_id, rating in zip(watched_movies, ratings):
                if rating <= 2:
                    similarity = self.content_recommender.get_similarity_score(watched_id, movie_id)
                    if similarity > 0.2:
                        explanation['penalty_applied'] = True
                        break
        
        # 4. Score combinado
        content_score = max([r['similarity_score'] for r in explanation['content_based_reasons']], default=0)
        collab_score = explanation.get('collaborative_reasons', {}).get('predicted_rating', 0) / 5.0
        
        blended = self.content_weight * content_score + self.collab_weight * collab_score
        
        if explanation['penalty_applied']:
            blended = blended * 0.1  # Penalización por rating bajo
        
        explanation['blended_score'] = round(blended, 3)
        
        return explanation
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del recomendador híbrido.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'strategy': self.strategy,
            'content_weight': self.content_weight,
            'collab_weight': self.collab_weight,
            'use_fallback': self.use_fallback,
            'is_ready': self.is_ready,
            'content_stats': self.content_recommender.get_statistics() if self.content_recommender else {},
            'collab_stats': self.collab_recommender.get_statistics() if self.collab_recommender else {}
        }
        
        return stats
