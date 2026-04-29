"""
Módulo de utilidades para el sistema de recomendación de películas.

Proporciona funciones auxiliares para:
- Logging con salida a archivo y captura para Streamlit
- Normalización de texto (nombres de actores/directores)
- Carga de configuración desde .env
- Manejo de errores comunes
- Gestión de calificaciones de usuarios (nuevo sistema ⭐1-5)
"""

import os
import re
import logging
import unicodedata
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración global de directorios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / os.getenv('DATA_DIR', './data')
MODELS_DIR = BASE_DIR / os.getenv('MODELS_DIR', './models')
LOGS_DIR = BASE_DIR / os.getenv('LOGS_DIR', './logs')
EXPORTS_DIR = BASE_DIR / 'exports'

# Crear directorios si no existen
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== LOGGING CON CAPTURA PARA STREAMLIT ====================

class StreamlitLogHandler(logging.Handler):
    """Handler personalizado que captura logs y los almacena para mostrar en Streamlit."""
    
    def __init__(self, log_storage: list):
        super().__init__()
        self.log_storage = log_storage
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    def emit(self, record):
        """Almacena el log formateado en la lista."""
        try:
            msg = self.format(record)
            self.log_storage.append(msg)
            # Limitar tamaño para evitar memory leak
            if len(self.log_storage) > 1000:
                self.log_storage.pop(0)
        except Exception:
            self.handleError(record)

# Almacenamiento global de logs para la UI
LOG_BUFFER = []

def setup_logging(log_to_ui: bool = True) -> logging.Logger:
    """
    Configura el sistema de logging dual: archivo + UI (opcional).
    
    Args:
        log_to_ui (bool): Si es True, también envía logs a un buffer para Streamlit.
    
    Returns:
        logging.Logger: Logger configurado.
    """
    logger = logging.getLogger('MovieRecommender')
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicación de handlers
    if logger.handlers:
        return logger
    
    # Formato detallado
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo
    log_file = LOGS_DIR / f'app_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para consola (Streamlit lo captura si se ejecuta en terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    # Handler para UI (buffer)
    if log_to_ui:
        ui_handler = StreamlitLogHandler(LOG_BUFFER)
        ui_handler.setLevel(logging.INFO)
        logger.addHandler(ui_handler)
    
    return logger

def get_log_buffer() -> list:
    """Retorna el buffer actual de logs para mostrar en Streamlit."""
    return LOG_BUFFER

def clear_log_buffer():
    """Limpia el buffer de logs."""
    LOG_BUFFER.clear()

# ==================== NORMALIZACIÓN DE TEXTOS ====================

def normalize_text(text: str, capitalize: bool = True) -> str:
    """
    Normaliza texto: elimina acentos, convierte a minúsculas, limpia espacios.
    
    Args:
        text (str): Texto a normalizar.
        capitalize (bool): Si es True, aplica mayúscula inicial a cada palabra.
    
    Returns:
        str: Texto normalizado.
    
    Examples:
        >>> normalize_text("Robert Downey Jr.")
        'Robert Downey Jr.'
        >>> normalize_text("  José   García  ")
        'Jose Garcia'
    """
    if not isinstance(text, str):
        return "Desconocido"
    
    # Eliminar acentos
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Reemplazar caracteres especiales
    text = re.sub(r'[^\w\s\-\.]', '', text)
    
    # Limpiar espacios múltiples y trim
    text = re.sub(r'\s+', ' ', text).strip()
    
    if capitalize and text:
        # Capitalizar cada palabra (excepto Jr., Sr., etc. que ya están bien)
        words = text.split()
        words = [w.capitalize() if w.upper() not in ['JR.', 'SR.', 'II', 'III', 'IV'] else w for w in words]
        text = ' '.join(words)
    else:
        text = text.lower()
    
    return text if text else "Desconocido"

def normalize_actor_name(name: str) -> str:
    """
    Normaliza nombres de actores para matching consistente.
    
    Args:
        name (str): Nombre del actor.
    
    Returns:
        str: Nombre normalizado (primer apellido + nombre inicial).
    
    Examples:
        >>> normalize_actor_name("Robert Downey Jr.")
        'downey robert'
    """
    if not name or name == "Desconocido":
        return "Desconocido"
    
    # Normalizar y pasar a minúsculas para matching
    normalized = normalize_text(name, capitalize=False)
    
    # Para actores, a veces queremos solo apellido principal
    parts = normalized.split()
    if len(parts) >= 2:
        # Formato: apellido_principal + nombre_inicial
        return f"{parts[-1]} {parts[0]}"
    return normalized

# ==================== MANEJO DE ERRORES ====================

class RecommenderError(Exception):
    """Excepción base para errores del sistema de recomendación."""
    pass

class DataNotFoundError(RecommenderError):
    """Error cuando no se encuentran datos necesarios."""
    pass

class MovieNotFoundError(RecommenderError):
    """Error cuando una película no está en el dataset."""
    pass

class InsufficientDataError(RecommenderError):
    """Error cuando no hay suficientes datos para una recomendación."""
    pass

def safe_execute(logger: logging.Logger, default_return=None):
    """
    Decorador para ejecutar funciones con manejo de errores consistente.
    
    Args:
        logger: Logger para registrar errores.
        default_return: Valor a retornar si falla la función.
    
    Returns:
        Decorador configurado.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error en {func.__name__}: {str(e)}", exc_info=True)
                return default_return
        return wrapper
    return decorator

# ==================== CONFIGURACIÓN ====================

def get_config() -> Dict[str, Any]:
    """
    Carga configuración desde variables de entorno.
    
    Returns:
        Dict con valores de configuración.
    """
    return {
        'tmdb_api_key': os.getenv('TMDB_API_KEY'),
        'default_top_n': int(os.getenv('DEFAULT_TOP_N', 10)),
        'default_content_weight': float(os.getenv('DEFAULT_CONTENT_WEIGHT', 0.6)),
        'default_collab_weight': float(os.getenv('DEFAULT_COLLAB_WEIGHT', 0.4)),
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'logs_dir': LOGS_DIR,
        'exports_dir': EXPORTS_DIR
    }

def validate_tmdb_key(logger: logging.Logger) -> bool:
    """
    Valida si la API key de TMDB está presente y tiene formato correcto.
    
    Args:
        logger: Logger para registrar advertencias.
    
    Returns:
        bool: True si la key existe y tiene formato válido.
    """
    key = os.getenv('TMDB_API_KEY')
    if not key:
        logger.warning("No se encontró TMDB_API_KEY en .env. Los pósters no funcionarán.")
        return False
    if len(key) < 32:
        logger.warning("TMDB_API_KEY parece tener formato incorrecto.")
        return False
    return True

# ==================== FUNCIONES PARA RATINGS (NUEVO SISTEMA) ====================

def rating_to_category(rating: int) -> str:
    """
    Convierte una calificación numérica a categoría.
    
    Args:
        rating: Calificación de 1 a 5
    
    Returns:
        str: 'favorite' para 3-5, 'disliked' para 1-2, 'none' para otros
    """
    if rating >= 3:
        return 'favorite'
    elif rating >= 1:
        return 'disliked'
    return 'none'

def is_high_rating(rating: int) -> bool:
    """Retorna True si la calificación es alta (3-5)."""
    return rating >= 3 and rating <= 5

def is_low_rating(rating: int) -> bool:
    """Retorna True si la calificación es baja (1-2)."""
    return rating >= 1 and rating <= 2
