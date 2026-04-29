"""
Módulo de feedback y aprendizaje online - NUEVO SISTEMA DE RATINGS ⭐1-5.

Implementa:
- Sistema de calificación por estrellas (1-5) para películas
- Almacenamiento en JSON separado (user_ratings.json)
- Historial completo de películas calificadas
- Carga automática de ratings históricos desde MovieLens
- Penalización de películas con baja calificación en recomendaciones
- Exportación/importación de calificaciones
"""

import json
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Importar utilidades
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import setup_logging, safe_execute, get_config, rating_to_category, is_high_rating, is_low_rating

logger = setup_logging(log_to_ui=True)


class RatingEntry:
    """
    Representa una calificación individual de un usuario.
    """
    
    def __init__(self, rating: int, timestamp: str = None, source: str = "manual"):
        """
        Args:
            rating: Calificación de 1 a 5
            timestamp: Fecha y hora de la calificación (ISO format)
            source: Origen ('manual', 'recommendation', 'historical')
        """
        self.rating = max(1, min(5, rating))  # Asegurar rango 1-5
        self.timestamp = timestamp or datetime.now().isoformat()
        self.source = source
    
    def to_dict(self) -> Dict:
        return {
            'rating': self.rating,
            'timestamp': self.timestamp,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RatingEntry':
        return cls(
            rating=data.get('rating', 3),
            timestamp=data.get('timestamp'),
            source=data.get('source', 'manual')
        )
    
    def is_high(self) -> bool:
        """Retorna True si es alta calificación (3-5)."""
        return self.rating >= 3
    
    def is_low(self) -> bool:
        """Retorna True si es baja calificación (1-2)."""
        return self.rating <= 2


class UserRatingsManager:
    """
    Gestor de calificaciones de usuarios con almacenamiento en JSON separado.
    
    Estructura del archivo user_ratings.json:
    {
        "1": {
            "ratings": {
                "123": {"rating": 5, "timestamp": "2024-01-15T10:30:00", "source": "historical"},
                "456": {"rating": 2, "timestamp": "2024-01-15T10:31:00", "source": "recommendation"}
            },
            "last_updated": "2024-01-15T10:31:00"
        }
    }
    """
    
    def __init__(self, storage_file: str = "user_ratings.json"):
        """
        Inicializa el gestor de calificaciones.
        
        Args:
            storage_file: Nombre del archivo JSON para almacenamiento
        """
        self.storage_path = get_config()['data_dir'] / storage_file
        
        # Estructura: user_id -> dict con 'ratings' y 'last_updated'
        self.user_ratings: Dict[int, Dict] = defaultdict(lambda: {
            'ratings': {},
            'last_updated': None
        })
        
        # Cargar datos existentes
        self.load_ratings()
        
        logger.info(f"UserRatingsManager inicializado (archivo: {self.storage_path})")
    
    def load_ratings(self):
        """Carga calificaciones de usuario desde archivo JSON."""
        if not self.storage_path.exists():
            logger.info("No se encontró archivo de ratings previo. Creando nuevo.")
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for user_id_str, user_data in data.items():
                user_id = int(user_id_str)
                ratings_dict = {}
                
                for movie_id_str, rating_data in user_data.get('ratings', {}).items():
                    # Convertir movie_id a int (manejar tanto '1' como '1.0')
                    try:
                        movie_id = int(float(movie_id_str))  # Esto convierte '1.0' a 1
                    except (ValueError, TypeError):
                        movie_id = int(movie_id_str)
                    
                    ratings_dict[movie_id] = RatingEntry.from_dict(rating_data)
                
                self.user_ratings[user_id] = {
                    'ratings': ratings_dict,
                    'last_updated': user_data.get('last_updated')
                }
            
            logger.info(f"Ratings cargados: {len(self.user_ratings)} usuarios con calificaciones")
            
        except Exception as e:
            logger.error(f"Error cargando ratings: {e}")
    
    def save_ratings(self):
        """Guarda calificaciones de usuario a archivo JSON."""
        try:
            data = {}
            for user_id, user_data in self.user_ratings.items():
                # Convertir RatingEntry a dict serializable
                ratings_dict = {}
                for movie_id, rating_entry in user_data['ratings'].items():
                    ratings_dict[str(movie_id)] = rating_entry.to_dict()
                
                data[str(user_id)] = {
                    'ratings': ratings_dict,
                    'last_updated': user_data.get('last_updated')
                }
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Ratings guardados: {len(self.user_ratings)} usuarios")
            
        except Exception as e:
            logger.error(f"Error guardando ratings: {e}")
    
    def get_user_ratings(self, user_id: int) -> Dict[int, RatingEntry]:
        """Obtiene todas las calificaciones de un usuario."""
        return self.user_ratings[user_id]['ratings'].copy()
    
    def get_user_rating(self, user_id: int, movie_id: int) -> Optional[RatingEntry]:
        """Obtiene la calificación de un usuario para una película específica."""
        return self.user_ratings[user_id]['ratings'].get(movie_id)
    
    def get_rating_value(self, user_id: int, movie_id: int) -> Optional[int]:
        """Obtiene solo el valor numérico de la calificación."""
        entry = self.get_user_rating(user_id, movie_id)
        return entry.rating if entry else None
    
    def set_rating(self, user_id: int, movie_id: int, rating: int, source: str = "manual") -> int:
        """
        Establece una calificación para una película.
        
        Args:
            user_id: ID del usuario
            movie_id: ID de la película
            rating: Calificación de 1 a 5
            source: Origen ('manual', 'recommendation', 'historical')
        
        Returns:
            int: Rating asignado (1-5)
        """
        rating = max(1, min(5, rating))
        
        # Asegurar que movie_id sea int, no float
        movie_id_int = int(movie_id) if isinstance(movie_id, float) else movie_id
        
        rating_entry = RatingEntry(rating, source=source)
        self.user_ratings[user_id]['ratings'][movie_id_int] = rating_entry
        self.user_ratings[user_id]['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Usuario {user_id} calificó película {movie_id_int} con {rating}⭐ (fuente: {source})")
        
        # Guardar cambios
        self.save_ratings()
        
        return rating
    
    def remove_rating(self, user_id: int, movie_id: int) -> bool:
        """
        Elimina la calificación de una película.
        
        Args:
            user_id: ID del usuario
            movie_id: ID de la película
        
        Returns:
            bool: True si se eliminó, False si no existía
        """
        if movie_id in self.user_ratings[user_id]['ratings']:
            del self.user_ratings[user_id]['ratings'][movie_id]
            self.user_ratings[user_id]['last_updated'] = datetime.now().isoformat()
            self.save_ratings()
            logger.info(f"Calificación eliminada para usuario {user_id}, película {movie_id}")
            return True
        return False
    
    def get_high_rated_movies(self, user_id: int) -> List[int]:
        """
        Obtiene lista de películas con alta calificación (3-5⭐).
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de movie_id con rating >= 3
        """
        ratings = self.get_user_ratings(user_id)
        return [movie_id for movie_id, entry in ratings.items() if entry.rating >= 3]
    
    def get_low_rated_movies(self, user_id: int) -> List[int]:
        """
        Obtiene lista de películas con baja calificación (1-2⭐).
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de movie_id con rating <= 2
        """
        ratings = self.get_user_ratings(user_id)
        return [movie_id for movie_id, entry in ratings.items() if entry.rating <= 2]
    
    def get_all_rated_movies(self, user_id: int) -> List[Tuple[int, int]]:
        """
        Obtiene todas las películas calificadas con sus ratings.
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de tuplas (movie_id, rating)
        """
        ratings = self.get_user_ratings(user_id)
        return [(movie_id, entry.rating) for movie_id, entry in ratings.items()]
    
    def load_historical_ratings_from_movielens(self, user_id: int, ratings_df: pd.DataFrame) -> int:
        """
        Carga las calificaciones históricas del usuario desde MovieLens.
        Solo carga ratings 3-5⭐ como favoritos (alta calificación).
        
        Args:
            user_id: ID del usuario
            ratings_df: DataFrame con ratings de MovieLens
        
        Returns:
            int: Número de ratings cargados
        """
        if ratings_df is None or ratings_df.empty:
            logger.warning("No hay datos de ratings de MovieLens para cargar")
            return 0
        
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        if user_ratings.empty:
            logger.info(f"Usuario {user_id} no tiene ratings históricos en MovieLens")
            return 0
        
        loaded_count = 0
        for _, row in user_ratings.iterrows():
            movie_id = row['movie_id']
            rating = int(round(row['rating']))  # Redondear rating (puede ser decimal en algunos datasets)
            
            # Solo cargar si no existe ya una calificación (no sobrescribir calificaciones manuales)
            if movie_id not in self.user_ratings[user_id]['ratings']:
                self.set_rating(user_id, movie_id, rating, source="historical")
                loaded_count += 1
        
        logger.info(f"Cargados {loaded_count} ratings históricos para usuario {user_id}")
        return loaded_count
    
    def clear_user_ratings(self, user_id: int, confirm: bool = False) -> bool:
        """
        Elimina todas las calificaciones de un usuario.
        
        Args:
            user_id: ID del usuario
            confirm: Debe ser True para confirmar eliminación
        
        Returns:
            bool: True si se eliminó correctamente
        """
        if not confirm:
            logger.warning(f"Eliminación de ratings para usuario {user_id} no confirmada")
            return False
        
        if user_id in self.user_ratings:
            self.user_ratings[user_id] = {
                'ratings': {},
                'last_updated': datetime.now().isoformat()
            }
            self.save_ratings()
            logger.info(f"Ratings eliminados para usuario {user_id}")
            return True
        
        return False
    
    def get_ratings_summary(self, user_id: int) -> Dict[str, Any]:
        """
        Obtiene resumen de calificaciones de un usuario.
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Dict con estadísticas
        """
        ratings = self.get_user_ratings(user_id)
        
        if not ratings:
            return {
                'user_id': user_id,
                'total_ratings': 0,
                'high_ratings': 0,
                'low_ratings': 0,
                'avg_rating': 0,
                'last_updated': None
            }
        
        rating_values = [entry.rating for entry in ratings.values()]
        high_count = sum(1 for r in rating_values if r >= 3)
        low_count = sum(1 for r in rating_values if r <= 2)
        
        return {
            'user_id': user_id,
            'total_ratings': len(ratings),
            'high_ratings': high_count,
            'low_ratings': low_count,
            'avg_rating': sum(rating_values) / len(rating_values),
            'last_updated': self.user_ratings[user_id].get('last_updated')
        }
    
    def export_ratings_to_excel(self, user_id: int, movies_df: pd.DataFrame, output_path: Path) -> bool:
        """
        Exporta todas las calificaciones del usuario a Excel.
        
        Args:
            user_id: ID del usuario
            movies_df: DataFrame con información de películas
            output_path: Ruta donde guardar el archivo Excel
        
        Returns:
            bool: True si se exportó correctamente
        """
        ratings = self.get_user_ratings(user_id)
        
        if not ratings:
            logger.warning(f"Usuario {user_id} no tiene calificaciones para exportar")
            return False
        
        # Obtener detalles de las películas
        movie_ids = list(ratings.keys())
        rated_movies = movies_df[movies_df['movie_id'].isin(movie_ids)].copy()
        
        if rated_movies.empty:
            logger.warning("No se encontraron detalles para las películas calificadas")
            return False
        
        # Preparar DataFrame para exportación
        export_data = []
        for _, row in rated_movies.iterrows():
            movie_id = row['movie_id']
            rating_entry = ratings[movie_id]
            
            # Obtener géneros como string
            genres = row.get('genres_processed', [])
            if hasattr(genres, 'tolist'):
                genres = genres.tolist()
            genres_str = ', '.join(str(g) for g in genres) if genres else ''
            
            export_data.append({
                'ID': movie_id,
                'Título': row.get('title_clean', row.get('title', 'Desconocido')),
                'Año': row.get('year', 0),
                'Géneros': genres_str,
                'Mi Calificación ⭐': rating_entry.rating,
                'Categoría': 'Favorita' if rating_entry.rating >= 3 else 'No me gustó',
                'Fecha Calificación': rating_entry.timestamp,
                'Fuente': rating_entry.source
            })
        
        export_df = pd.DataFrame(export_data)
        export_df = export_df.sort_values('Mi Calificación ⭐', ascending=False)
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Mis Calificaciones', index=False)
                
                # Añadir hoja de estadísticas
                summary = self.get_ratings_summary(user_id)
                stats_df = pd.DataFrame({
                    'Métrica': [
                        'Total de películas calificadas',
                        'Películas favoritas (3-5⭐)',
                        'Películas que no me gustaron (1-2⭐)',
                        'Calificación promedio',
                        'Última actualización'
                    ],
                    'Valor': [
                        summary['total_ratings'],
                        summary['high_ratings'],
                        summary['low_ratings'],
                        f"{summary['avg_rating']:.2f} ⭐",
                        summary['last_updated'] or 'Nunca'
                    ]
                })
                stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
            
            logger.info(f"Exportadas {len(export_data)} calificaciones a {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exportando a Excel: {e}")
            return False


# ==================== CLASE PRINCIPAL FEEDBACK SYSTEM (COMPATIBILIDAD) ====================

class FeedbackSystem:
    """
    Sistema de feedback actualizado para usar el nuevo sistema de ratings ⭐1-5.
    Mantiene compatibilidad con métodos antiguos (add_like, add_dislike) convirtiéndolos.
    """
    
    def __init__(self, storage_file: str = "user_ratings.json"):
        """
        Inicializa el sistema de feedback.
        
        Args:
            storage_file: Nombre del archivo JSON para almacenamiento
        """
        self.ratings_manager = UserRatingsManager(storage_file)
        
        logger.info("FeedbackSystem inicializado con sistema de ratings ⭐1-5")
    
    # ============ MÉTODOS NUEVOS (RATINGS ⭐1-5) ============
    
    def set_rating(self, user_id: int, movie_id: int, rating: int, source: str = "manual") -> int:
        """Establece una calificación de 1 a 5 estrellas."""
        return self.ratings_manager.set_rating(user_id, movie_id, rating, source)
    
    def get_rating(self, user_id: int, movie_id: int) -> Optional[int]:
        """Obtiene la calificación de un usuario para una película."""
        return self.ratings_manager.get_rating_value(user_id, movie_id)
    
    def get_user_ratings(self, user_id: int) -> Dict[int, int]:
        """Obtiene todas las calificaciones del usuario (movie_id -> rating)."""
        ratings = self.ratings_manager.get_user_ratings(user_id)
        return {movie_id: entry.rating for movie_id, entry in ratings.items()}
    
    def get_user_likes(self, user_id: int) -> Set[int]:
        """Compatibilidad: retorna películas con alta calificación (3-5⭐)."""
        return set(self.ratings_manager.get_high_rated_movies(user_id))
    
    def get_user_dislikes(self, user_id: int) -> Set[int]:
        """Compatibilidad: retorna películas con baja calificación (1-2⭐)."""
        return set(self.ratings_manager.get_low_rated_movies(user_id))
    
    def get_high_rated_movies(self, user_id: int) -> List[int]:
        """Obtiene películas con calificación alta (3-5⭐)."""
        return self.ratings_manager.get_high_rated_movies(user_id)
    
    def get_low_rated_movies(self, user_id: int) -> List[int]:
        """Obtiene películas con calificación baja (1-2⭐)."""
        return self.ratings_manager.get_low_rated_movies(user_id)
    
    def get_all_rated_movies(self, user_id: int) -> List[Tuple[int, int]]:
        """Obtiene todas las películas calificadas con sus ratings."""
        return self.ratings_manager.get_all_rated_movies(user_id)
    
    def load_historical_ratings(self, user_id: int, ratings_df: pd.DataFrame) -> int:
        """Carga ratings históricos desde MovieLens."""
        return self.ratings_manager.load_historical_ratings_from_movielens(user_id, ratings_df)
    
    def remove_rating(self, user_id: int, movie_id: int) -> bool:
        """Elimina una calificación."""
        return self.ratings_manager.remove_rating(user_id, movie_id)
    
    def clear_user_ratings(self, user_id: int, confirm: bool = False) -> bool:
        """Elimina todas las calificaciones del usuario."""
        return self.ratings_manager.clear_user_ratings(user_id, confirm)
    
    def export_ratings_to_excel(self, user_id: int, movies_df: pd.DataFrame, output_path: Path) -> bool:
        """Exporta calificaciones a Excel."""
        return self.ratings_manager.export_ratings_to_excel(user_id, movies_df, output_path)
    
    def get_ratings_summary(self, user_id: int) -> Dict[str, Any]:
        """Obtiene resumen de calificaciones."""
        return self.ratings_manager.get_ratings_summary(user_id)
    
    # ============ MÉTODOS DE COMPATIBILIDAD (DEPRECATED) ============
    
    def add_like(self, user_id: int, movie_id: int, movie_title: str = ""):
        """Compatibilidad: convierte like a ⭐4."""
        logger.info(f"add_like convertido a set_rating(user={user_id}, movie={movie_id}, rating=4)")
        self.set_rating(user_id, movie_id, 4, source="legacy_like")
    
    def add_dislike(self, user_id: int, movie_id: int, movie_title: str = ""):
        """Compatibilidad: convierte dislike a ⭐2."""
        logger.info(f"add_dislike convertido a set_rating(user={user_id}, movie={movie_id}, rating=2)")
        self.set_rating(user_id, movie_id, 2, source="legacy_dislike")
    
    def add_personal_rating(self, user_id: int, movie_id: int, rating: float, movie_title: str = ""):
        """Compatibilidad: añade rating personal."""
        self.set_rating(user_id, movie_id, int(round(rating)), source="legacy_rating")
    
    def remove_feedback(self, user_id: int, movie_id: int):
        """Compatibilidad: elimina cualquier feedback."""
        self.remove_rating(user_id, movie_id)
    
    def get_feedback_summary(self, user_id: int = None) -> Dict[str, Any]:
        """Compatibilidad: obtiene resumen de feedback."""
        if user_id:
            summary = self.get_ratings_summary(user_id)
            return {
                'user_id': summary['user_id'],
                'total_likes': summary['high_ratings'],
                'total_dislikes': summary['low_ratings'],
                'total_personal_ratings': summary['total_ratings'],
                'total_feedback_actions': summary['total_ratings'],
                'last_updated': summary['last_updated'],
                'avg_personal_rating': summary['avg_rating']
            }
        else:
            # Resumen global (simplificado)
            return {
                'total_users': len(self.ratings_manager.user_ratings),
                'total_likes': 0,  # Se calcularía iterando
                'total_dislikes': 0,
                'total_feedback_actions': 0,
                'users_with_feedback': sum(1 for u in self.ratings_manager.user_ratings.values() if u['ratings'])
            }
    
    def adjust_recommendations(self, user_id: int, recommendations: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Ajusta scores de recomendaciones según calificaciones del usuario.
        
        - Películas con alta calificación (3-5⭐): se mantienen o potencian
        - Películas con baja calificación (1-2⭐): se penalizan fuertemente
        
        Args:
            user_id: ID del usuario
            recommendations: Lista de (movie_id, score)
        
        Returns:
            Lista ajustada de (movie_id, adjusted_score)
        """
        ratings = self.get_user_ratings(user_id)
        
        if not ratings:
            return recommendations
        
        adjusted = []
        for movie_id, score in recommendations:
            rating = ratings.get(movie_id)
            
            if rating is None:
                adjusted_score = score
            elif rating <= 2:
                # Penalizar fuertemente películas con baja calificación
                penalty = 0.1  # Reducir al 10% del score original
                adjusted_score = score * penalty
                logger.debug(f"Película {movie_id} penalizada (rating={rating}): {score:.3f} -> {adjusted_score:.3f}")
            elif rating >= 4:
                # Potenciar ligeramente películas con alta calificación
                boost = 1.1
                adjusted_score = min(1.0, score * boost)
                logger.debug(f"Película {movie_id} potenciada (rating={rating}): {score:.3f} -> {adjusted_score:.3f}")
            else:
                # Rating 3 es neutral
                adjusted_score = score
            
            adjusted.append((movie_id, adjusted_score))
        
        # Re-ordenar por nuevo score
        adjusted.sort(key=lambda x: x[1], reverse=True)
        
        return adjusted
