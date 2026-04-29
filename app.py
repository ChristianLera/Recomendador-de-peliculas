"""
Aplicación principal de Streamlit para el Sistema de Recomendación de Películas.
Con pósters de TMDB y diseño profesional tipo Netflix.
SISTEMA DE CALIFICACIÓN POR ESTRELLAS ⭐1-5
HISTORIAL EN VENTANA MODAL CON BÚSQUEDA, ORDENAMIENTO Y PAGINACIÓN
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
import requests
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO

# Configurar página
st.set_page_config(
    page_title="Sistema de Recomendación de Películas",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Añadir src al path
sys.path.append(str(Path(__file__).parent))

# Importar módulos
from src.utils import (
    setup_logging, get_log_buffer, clear_log_buffer,
    get_config, validate_tmdb_key, LOG_BUFFER
)
from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.feedback import FeedbackSystem
from src.unified_data import load_unified_data, load_ratings_data, enrich_movie_with_tmdb, get_user_historical_ratings

# Configurar logging
logger = setup_logging(log_to_ui=True)


# ==================== FUNCIONES AUXILIARES ====================

def get_poster_url(poster_path: str, size: str = 'w342') -> str:
    """Construye la URL completa del póster de TMDB."""
    if not poster_path or poster_path == '' or pd.isna(poster_path):
        return None
    return f"https://image.tmdb.org/t/p/{size}{poster_path}"


def enrich_selected_movie(movie_id: int, force_refresh: bool = False):
    """Enriquece UNA película con datos de TMDB bajo demanda."""
    if st.session_state.movies_df is None:
        return None
    
    movie_rows = st.session_state.movies_df[st.session_state.movies_df['movie_id'] == movie_id]
    if movie_rows.empty:
        return None
    
    movie = movie_rows.iloc[0].copy()
    
    if not force_refresh and movie.get('tmdb_loaded', False):
        return movie
    
    try:
        enriched = enrich_movie_with_tmdb(movie, force_refresh=force_refresh)
        
        idx = st.session_state.movies_df[st.session_state.movies_df['movie_id'] == movie_id].index[0]
        for key, value in enriched.items():
            if key in st.session_state.movies_df.columns:
                st.session_state.movies_df.at[idx, key] = value
        
        return enriched
    except Exception as e:
        logger.error(f"Error enriqueciendo: {e}")
        return movie


# ==================== CSS PERSONALIZADO ====================

st.markdown("""
<style>
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    
    /* Estilos para estrellas en botones */
    .stButton button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4CAF50;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #45a049;
    }
</style>
""", unsafe_allow_html=True)


# ==================== INICIALIZACIÓN DE SESIÓN ====================

def init_session_state():
    """Inicializa todas las variables de estado de sesión."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.data_loaded = False
        st.session_state.content_rec = None
        st.session_state.collab_rec = None
        st.session_state.hybrid_rec = None
        st.session_state.feedback = None
        st.session_state.movies_df = None
        st.session_state.ratings_df = None
        st.session_state.current_user_id = 1
        st.session_state.selected_movie_id = None
        st.session_state.recommendations = []
        st.session_state.recommendation_details = []
        st.session_state.active_tab = "Híbrido"
        st.session_state.log_expanded = False
        st.session_state.data_loading_in_progress = False
        st.session_state.historical_ratings_loaded = {}
        st.session_state.show_history_modal = False
        st.session_state.history_search = ""
        st.session_state.history_sort_by = "rating"
        st.session_state.history_sort_ascending = False
        st.session_state.history_page = 1
        st.session_state.current_tab_for_sidebar = "Híbrido"
        st.session_state.show_history_popover = False  


def load_data(source: str = 'movielens_small', include_tmdb: bool = True, force_reload: bool = False):
    """Carga SOLO MovieLens (rápido)."""
    with st.spinner("Cargando MovieLens..."):
        try:
            unified_df = load_unified_data(force_reload=force_reload)
            ratings_df = load_ratings_data(force_reload=force_reload)
            
            st.session_state.movies_df = unified_df
            st.session_state.ratings_df = ratings_df
            st.session_state.data_loaded = True
            
            st.success(f"✅ MovieLens cargado: {len(unified_df):,} películas | {len(ratings_df):,} ratings")
            st.info("💡 Los pósters se cargarán bajo demanda desde TMDB")
            return True
        except Exception as e:
            st.error(f"Error: {e}")
            return False


def train_models():
    """Entrena los modelos de recomendación."""
    if not st.session_state.data_loaded:
        st.warning("Primero carga los datos")
        return False
    
    with st.spinner("Entrenando modelos..."):
        try:
            preprocessor = DataPreprocessor(
                min_ratings_per_user=5,
                min_ratings_per_movie=3
            )
            
            st.session_state.content_rec = ContentBasedRecommender(preprocessor)
            st.session_state.content_rec.fit(st.session_state.movies_df)
            
            if st.session_state.ratings_df is not None and not st.session_state.ratings_df.empty:
                st.session_state.collab_rec = CollaborativeRecommender(preprocessor)
                st.session_state.collab_rec.fit(
                    st.session_state.ratings_df,
                    st.session_state.movies_df
                )
            else:
                st.warning("No hay datos de ratings.")
                st.session_state.collab_rec = None
            
            st.session_state.hybrid_rec = HybridRecommender(
                content_recommender=st.session_state.content_rec,
                collab_recommender=st.session_state.collab_rec,
                content_weight=0.6,
                collab_weight=0.4,
                strategy='weighted'
            )
            
            st.session_state.feedback = FeedbackSystem()
            
            logger.info("Modelos entrenados exitosamente")
            return True
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error: {e}", exc_info=True)
            return False


def load_user_historical_ratings(user_id: int):
    """Carga los ratings históricos del usuario desde MovieLens."""
    if st.session_state.feedback is None:
        return 0
    
    if st.session_state.historical_ratings_loaded.get(user_id, False):
        return 0
    
    historical_ratings = get_user_historical_ratings(user_id, min_rating=3)
    
    if not historical_ratings:
        logger.info(f"Usuario {user_id} no tiene ratings históricos altos (3-5⭐)")
        return 0
    
    loaded_count = 0
    for movie_id, rating in historical_ratings:
        existing = st.session_state.feedback.get_rating(user_id, movie_id)
        if existing is None:
            st.session_state.feedback.set_rating(user_id, movie_id, rating, source="historical")
            loaded_count += 1
    
    st.session_state.historical_ratings_loaded[user_id] = True
    logger.info(f"Cargados {loaded_count} ratings históricos para usuario {user_id}")
    
    return loaded_count


def auto_load_and_train():
    """
    Carga datos y entrena modelos automáticamente al inicio.
    Exactamente igual que el botón "Recargar MovieLens".
    """
    if not st.session_state.initialized and not st.session_state.data_loading_in_progress:
        st.session_state.data_loading_in_progress = True
        
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info("🚀 Inicializando el Sistema de Recomendación...")
            
            progress_text = st.text("📀 Cargando MovieLens...")
            
            # Cargar datos (sin forzar recarga para inicio rápido, pero igual que el botón)
            if load_data(force_reload=False):
                progress_text.text("✅ MovieLens cargado correctamente")
                
                progress_text.text("🤖 Entrenando modelos de recomendación...")
                if train_models():
                    progress_text.text("✅ Modelos entrenados correctamente")
                    
                    progress_text.text("📊 Cargando tu historial de calificaciones...")
                    loaded = load_user_historical_ratings(st.session_state.current_user_id)
                    if loaded > 0:
                        progress_text.text(f"✅ Cargadas {loaded} películas de tu historial")
                    
                    st.session_state.initialized = True
                    st.success("🎉 Sistema listo para usar!")
                    time.sleep(1.5)
                else:
                    st.error("❌ Error al entrenar los modelos")
            else:
                st.error("❌ Error al cargar MovieLens")
            
            progress_text.empty()
        
        progress_placeholder.empty()
        st.session_state.data_loading_in_progress = False
        st.rerun()


def get_user_history():
    """Obtiene el historial de películas calificadas del usuario actual."""
    user_id = st.session_state.current_user_id
    watched_movies = []
    ratings = []
    
    if st.session_state.feedback:
        all_ratings = st.session_state.feedback.get_user_ratings(user_id)
        for movie_id, rating in all_ratings.items():
            watched_movies.append(movie_id)
            ratings.append(rating)
    
    return watched_movies, ratings


def generate_recommendations():
    """Genera recomendaciones según el método seleccionado."""
    user_id = st.session_state.current_user_id
    watched_movies, ratings = get_user_history()
    
    if not watched_movies:
        st.warning("No hay historial. Califica algunas películas primero (⭐3-5).")
        return []
    
    top_n = st.session_state.get('top_n', 10)
    method = st.session_state.active_tab
    
    with st.spinner(f"Generando recomendaciones y cargando pósters de TMDB..."):
        try:
            if method == "Contenido":
                recs = st.session_state.content_rec.recommend_for_user_history(
                    watched_movies, ratings, top_n=top_n
                )
            elif method == "Colaborativo":
                if st.session_state.collab_rec and st.session_state.collab_rec.is_fitted:
                    recs = st.session_state.collab_rec.recommend_for_user(
                        user_id, top_n=top_n
                    )
                else:
                    st.warning("Recomendador colaborativo no disponible.")
                    recs = []
            else:
                if st.session_state.hybrid_rec:
                    content_weight = st.session_state.get('content_weight', 0.6)
                    collab_weight = st.session_state.get('collab_weight', 0.4)
                    st.session_state.hybrid_rec.set_weights(content_weight, collab_weight)
                    
                    recs = st.session_state.hybrid_rec.recommend_for_user(
                        user_id, watched_movies, ratings, top_n=top_n
                    )
                else:
                    recs = []
            
            if st.session_state.feedback and recs:
                recs = st.session_state.feedback.adjust_recommendations(user_id, recs)
            
            st.session_state.recommendations = recs
            
            details = []
            total_recs = len(recs[:top_n])
            
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            for idx, (movie_id, score) in enumerate(recs[:top_n]):
                progress_value = (idx + 1) / total_recs if total_recs > 0 else 1
                progress_bar.progress(progress_value)
                progress_text.text(f"Cargando póster {idx + 1} de {total_recs}...")
                
                movie_data = st.session_state.movies_df[st.session_state.movies_df['movie_id'] == movie_id]
                
                if not movie_data.empty:
                    movie = movie_data.iloc[0].copy()
                    
                    poster_path = movie.get('poster_path', '')
                    if not poster_path or poster_path == '' or pd.isna(poster_path):
                        try:
                            enriched = enrich_movie_with_tmdb(movie, force_refresh=False)
                            movie_idx = st.session_state.movies_df[st.session_state.movies_df['movie_id'] == movie_id].index[0]
                            for key, value in enriched.items():
                                if key in st.session_state.movies_df.columns:
                                    st.session_state.movies_df.at[movie_idx, key] = value
                            movie = enriched
                            poster_path = movie.get('poster_path', '')
                        except Exception as e:
                            print(f"Error enriqueciendo {movie_id}: {e}")
                    
                    genres = movie.get('genres_processed', [])
                    if hasattr(genres, 'tolist'):
                        genres = genres.tolist()
                    elif isinstance(genres, np.ndarray):
                        genres = genres.tolist()
                    elif genres is None:
                        genres = []
                    
                    actors = movie.get('actors', [])
                    if hasattr(actors, 'tolist'):
                        actors = actors.tolist()
                    elif isinstance(actors, np.ndarray):
                        actors = actors.tolist()
                    elif actors is None:
                        actors = []
                    
                    director = movie.get('director', '')
                    if director is None or pd.isna(director):
                        director = ''
                    
                    score_raw = float(score)
                    if method == "Contenido":
                        score_normalized = max(0.0, min(1.0, score_raw))
                    elif method == "Colaborativo":
                        if score_raw > 1:
                            score_normalized = (score_raw - 1) / 4
                        else:
                            score_normalized = max(0.0, min(1.0, score_raw))
                    else:
                        if score_raw > 1:
                            score_normalized = (score_raw - 1) / 4
                        else:
                            score_normalized = max(0.0, min(1.0, score_raw))
                    
                    score_percentage = score_normalized * 100
                    score_percentage = max(0, min(100, score_percentage))
                    
                    current_rating = None
                    if st.session_state.feedback:
                        current_rating = st.session_state.feedback.get_rating(user_id, movie_id)
                    
                    detail = {
                        'movie_id': movie_id,
                        'title': movie.get('title_clean', movie.get('title', 'Desconocido')),
                        'year': movie.get('year', 0),
                        'genres': genres,
                        'actors': actors,
                        'director': director,
                        'poster_path': poster_path,
                        'score': score_normalized,
                        'score_percentage': score_percentage,
                        'user_rating': current_rating
                    }
                    
                    details.append(detail)
            
            progress_bar.empty()
            progress_text.empty()
            
            st.session_state.recommendation_details = details
            
            if details:
                scores = [d['score_percentage'] for d in details]
                st.info(f"📊 Rango de Match: {min(scores):.0f}% - {max(scores):.0f}% | Promedio: {sum(scores)/len(scores):.0f}%")
            
            return recs
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error: {e}", exc_info=True)
            return []


# ==================== MODAL DEL HISTORIAL (USANDO st.dialog) ====================

@st.dialog("📋 Mi Historial de Calificaciones", width="large")
def history_modal():
    """Ventana modal del historial usando st.dialog."""
    user_id = st.session_state.current_user_id
    
    if st.session_state.feedback is None:
        st.info("Sistema de calificaciones no disponible")
        return
    
    all_ratings = st.session_state.feedback.get_user_ratings(user_id)
    
    if not all_ratings:
        st.info("📭 No has calificado ninguna película todavía.")
        st.caption("💡 Califica películas con ⭐1-5 para verlas aquí.")
        if st.button("Cerrar"):
            st.rerun()
        return
    
    # Inicializar estado de paginación dentro del modal
    if 'modal_page_all' not in st.session_state:
        st.session_state.modal_page_all = 1
        st.session_state.modal_page_fav = 1
        st.session_state.modal_page_dis = 1
    
    # Filtros y ordenamiento (estos SÍ requieren rerun)
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        search_term = st.text_input("🔍 Buscar por título, año, director o actor:", 
                                    key="modal_search",
                                    placeholder="Escribe para buscar...")
    
    with col2:
        sort_by = st.selectbox("Ordenar por:", 
                               options=["rating", "title", "year"],
                               format_func=lambda x: "⭐ Rating" if x == "rating" else "📝 Título" if x == "title" else "📅 Año",
                               key="modal_sort_by")
        
        sort_ascending = st.checkbox("Ascendente", key="modal_sort_asc")
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Exportar filtrado", key="modal_export_btn"):
            export_filtered_ratings(search_term, sort_by, sort_ascending)
    
    # Clasificar ratings
    high_ratings = {m: r for m, r in all_ratings.items() if r >= 3}
    low_ratings = {m: r for m, r in all_ratings.items() if r <= 2}
    
    # Pestañas
    tab_all, tab_favorites, tab_disliked = st.tabs([
        f"📋 Todas ({len(all_ratings)})",
        f"⭐ Favoritas (3-5⭐) ({len(high_ratings)})",
        f"💔 No gustaron (1-2⭐) ({len(low_ratings)})"
    ])
    
    # Función para renderizar cada pestaña (sin rerun para paginación)
    def render_tab_content(ratings_dict, tab_id, page_state_key):
        if not ratings_dict:
            st.info("No hay películas en esta categoría")
            return
        
        movie_ids = list(ratings_dict.keys())
        movies_subset = st.session_state.movies_df[st.session_state.movies_df['movie_id'].isin(movie_ids)].copy()
        movies_subset['user_rating'] = movies_subset['movie_id'].map(ratings_dict)
        
        # Filtrar por búsqueda
        if search_term:
            search_lower = search_term.lower()
            def matches_search(row):
                title = str(row.get('title_clean', '')).lower()
                year = str(row.get('year', '')).lower()
                director = str(row.get('director', '')).lower()
                actors = str(row.get('actors', '')).lower()
                return (search_lower in title or search_lower in year or 
                       search_lower in director or search_lower in actors)
            movies_subset = movies_subset[movies_subset.apply(matches_search, axis=1)]
        
        # Ordenar
        if sort_by == 'title':
            movies_subset = movies_subset.sort_values('title_clean', ascending=sort_ascending)
        elif sort_by == 'year':
            movies_subset = movies_subset.sort_values('year', ascending=sort_ascending)
        elif sort_by == 'rating':
            movies_subset = movies_subset.sort_values('user_rating', ascending=sort_ascending)
        
        # Paginación
        items_per_page = 10
        total_items = len(movies_subset)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        # Obtener página actual desde session_state
        current_page = st.session_state.get(page_state_key, 1)
        
        # Asegurar que la página está dentro de los límites
        if current_page > total_pages:
            current_page = total_pages
            st.session_state[page_state_key] = current_page
        
        # Controles de paginación (usando botones que actualizan session_state SIN rerun)
        if total_pages > 1:
            col_prev, col_page_info, col_next = st.columns([1, 2, 1])
            with col_prev:
                if st.button("◀ Anterior", key=f"prev_{tab_id}"):
                    if current_page > 1:
                        st.session_state[page_state_key] = current_page - 1
                        st.rerun()
            with col_page_info:
                st.markdown(f"<div style='text-align: center'>Página {current_page} de {total_pages}</div>", unsafe_allow_html=True)
            with col_next:
                if st.button("Siguiente ▶", key=f"next_{tab_id}"):
                    if current_page < total_pages:
                        st.session_state[page_state_key] = current_page + 1
                        st.rerun()
        
        start_idx = (current_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_movies = movies_subset.iloc[start_idx:end_idx]
        
        # Mostrar películas
        for _, movie in page_movies.iterrows():
            movie_id = movie['movie_id']
            rating = ratings_dict[movie_id]
            
            col_poster, col_info, col_actions = st.columns([1, 3, 1.5])
            
            with col_poster:
                poster_path = movie.get('poster_path', '')
                if not poster_path or poster_path == '':
                    try:
                        enriched = enrich_movie_with_tmdb(movie, force_refresh=False)
                        poster_path = enriched.get('poster_path', '')
                    except:
                        pass
                
                if poster_path and poster_path != '':
                    st.image(f"https://image.tmdb.org/t/p/w92{poster_path}", width=70)
                else:
                    st.markdown("🎬")
            
            with col_info:
                st.markdown(f"**{movie.get('title_clean', 'Desconocido')}** ({movie.get('year', '?')})")
                st.caption(f"{'⭐' * rating} ({rating}/5)")
                
                genres = movie.get('genres_processed', [])
                if hasattr(genres, 'tolist'):
                    genres = genres.tolist()
                if genres and len(genres) > 0:
                    st.caption(f"🎭 {', '.join([str(g) for g in genres[:2] if g])}")
                
                director = movie.get('director', '')
                if director and director not in ['Pendiente de TMDB', 'No disponible', '', 'Desconocido']:
                    st.caption(f"🎬 {director[:40]}")
            
            with col_actions:
                # Selector de estrellas
                rating_cols = st.columns(5)
                for i in range(1, 6):
                    with rating_cols[i-1]:
                        btn_key = f"modal_rating_{tab_id}_{movie_id}_{i}"
                        if st.button(f"{i}⭐", key=btn_key, use_container_width=True):
                            st.session_state.feedback.set_rating(user_id, movie_id, i, source="modal")
                            st.rerun()
                
                if st.button("🗑️ Eliminar", key=f"modal_remove_{tab_id}_{movie_id}", use_container_width=True):
                    st.session_state.feedback.remove_rating(user_id, movie_id)
                    st.rerun()
            
            st.divider()
    
    with tab_all:
        render_tab_content(all_ratings, "all", "modal_page_all")
    with tab_favorites:
        render_tab_content(high_ratings, "fav", "modal_page_fav")
    with tab_disliked:
        render_tab_content(low_ratings, "dis", "modal_page_dis")


def export_filtered_ratings(search_term, sort_by, sort_ascending):
    """Exporta el historial completo, enriqueciendo películas bajo demanda."""
    user_id = st.session_state.current_user_id
    
    if st.session_state.feedback is None:
        st.warning("Sistema de calificaciones no disponible")
        return
    
    # Obtener TODAS las calificaciones del usuario
    all_ratings = st.session_state.feedback.get_user_ratings(user_id)
    
    if not all_ratings:
        st.warning("No hay calificaciones para exportar")
        return
    
    # Obtener datos de películas para esas calificaciones
    movie_ids = list(all_ratings.keys())
    movies_df = st.session_state.movies_df[st.session_state.movies_df['movie_id'].isin(movie_ids)].copy()
    movies_df['user_rating'] = movies_df['movie_id'].map(all_ratings)
    
    # Aplicar filtro de búsqueda si existe
    if search_term and search_term.strip():
        search_lower = search_term.lower().strip()
        
        def matches(row):
            title = str(row.get('title_clean', row.get('title', ''))).lower()
            if search_lower in title:
                return True
            year = str(row.get('year', '')).lower()
            if search_lower in year:
                return True
            return False
        
        movies_df = movies_df[movies_df.apply(matches, axis=1)]
    
    # Ordenar
    if sort_by == 'title':
        movies_df = movies_df.sort_values('title_clean', ascending=sort_ascending)
    elif sort_by == 'year':
        movies_df = movies_df.sort_values('year', ascending=sort_ascending)
    elif sort_by == 'rating':
        movies_df = movies_df.sort_values('user_rating', ascending=sort_ascending)
    
    # Función para enriquecer una película si no tiene datos
    def enrich_if_needed(movie_row):
        movie_id = movie_row['movie_id']
        
        # Verificar si ya tiene datos de TMDB
        director = movie_row.get('director', '')
        actors = movie_row.get('actors', [])
        
        # Si no tiene director o está pendiente, enriquecer
        if not director or director in ['Pendiente de TMDB', 'No disponible', 'Desconocido', None, '']:
            try:
                # Enriquecer bajo demanda
                movie_series = pd.Series(movie_row)
                enriched = enrich_movie_with_tmdb(movie_series, force_refresh=False)
                director = enriched.get('director', 'No disponible')
                actors = enriched.get('actors', [])
                
                # Actualizar en session_state para futuras exportaciones
                idx = st.session_state.movies_df[st.session_state.movies_df['movie_id'] == movie_id].index
                if len(idx) > 0:
                    st.session_state.movies_df.at[idx[0], 'director'] = director
                    st.session_state.movies_df.at[idx[0], 'actors'] = actors
                    st.session_state.movies_df.at[idx[0], 'tmdb_loaded'] = True
            except Exception as e:
                logger.warning(f"Error enriqueciendo película {movie_id}: {e}")
                director = 'No disponible'
                actors = []
        
        return director, actors
    
    # Procesar cada película con barra de progreso
    export_data = []
    total = len(movies_df)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, movie) in enumerate(movies_df.iterrows()):
        progress = (idx + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Procesando película {idx + 1} de {total}: {movie.get('title_clean', 'Unknown')[:30]}...")
        
        movie_id = movie['movie_id']
        rating = movie['user_rating']
        
        # Enriquecer si es necesario
        director, actors = enrich_if_needed(movie)
        
        # Procesar actores
        if isinstance(actors, (list, np.ndarray)):
            actors_clean = [str(a) for a in actors if a and str(a) != 'nan']
            actors_str = ', '.join(actors_clean[:5]) if actors_clean else 'No disponible'
        else:
            actors_str = str(actors) if actors and str(actors) != 'nan' else 'No disponible'
        
        # Procesar director
        if isinstance(director, (list, np.ndarray)):
            director_clean = [str(d) for d in director if d and str(d) != 'nan']
            director_str = ', '.join(director_clean) if director_clean else 'No disponible'
        else:
            director_str = str(director) if director and str(director) not in ['nan', 'None', ''] else 'No disponible'
        
        # Procesar géneros
        genres_raw = movie.get('genres_processed', [])
        if isinstance(genres_raw, (list, np.ndarray)):
            genres_clean = [str(g) for g in genres_raw if g and str(g) != 'nan']
            genres_str = ', '.join(genres_clean) if genres_clean else 'No especificado'
        else:
            genres_str = str(genres_raw) if genres_raw and str(genres_raw) != 'nan' else 'No especificado'
        
        # Título
        title = movie.get('title_clean', movie.get('title', 'Desconocido'))
        if isinstance(title, (list, np.ndarray)):
            title = ', '.join([str(t) for t in title if t]) if len(title) > 0 else 'Desconocido'
        elif str(title) in ['nan', 'None', '']:
            title = 'Desconocido'
        
        # Año
        year = movie.get('year', 0)
        try:
            year = int(float(str(year))) if year not in [None, 'nan', 'None', ''] else 0
        except:
            year = 0
        
        export_data.append({
            'ID': int(movie_id),
            'Título': str(title),
            'Año': year,
            'Mi Calificación ⭐': int(rating),
            'Categoría': 'Favorita (3-5⭐)' if rating >= 3 else 'No me gustó (1-2⭐)',
            'Director': director_str,
            'Actores Principales': actors_str,
            'Géneros': genres_str
        })
    
    progress_bar.empty()
    status_text.empty()
    
    if not export_data:
        st.warning("No hay datos para exportar después del filtrado")
        return
    
    export_df = pd.DataFrame(export_data)
    
    # Asegurar orden de columnas
    column_order = ['ID', 'Título', 'Año', 'Mi Calificación ⭐', 'Categoría', 'Director', 'Actores Principales', 'Géneros']
    export_df = export_df[column_order]
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Mi Historial', index=False)
        
        # Estadísticas
        avg_rating = export_df['Mi Calificación ⭐'].mean()
        fav_count = len(export_df[export_df['Mi Calificación ⭐'] >= 3])
        disliked_count = len(export_df[export_df['Mi Calificación ⭐'] <= 2])
        
        stats_df = pd.DataFrame({
            'Métrica': [
                'Total de películas calificadas',
                'Películas favoritas (3-5⭐)',
                'Películas que no gustaron (1-2⭐)',
                'Calificación promedio',
                'Fecha de exportación',
                'Usuario ID'
            ],
            'Valor': [
                len(export_df),
                fav_count,
                disliked_count,
                f"{avg_rating:.2f} ⭐",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                user_id
            ]
        })
        stats_df.to_excel(writer, sheet_name='Estadísticas', index=False)
    
    output.seek(0)
    st.download_button(
        label="📥 Descargar Excel",
        data=output,
        file_name=f"historial_usuario_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"export_download_{user_id}_{datetime.now().timestamp()}"
    )
    
    st.success(f"✅ Exportadas {len(export_df)} películas")


# ==================== BARRA LATERAL ====================

def render_sidebar():
    """Renderiza la barra lateral."""
    with st.sidebar:
        st.title("⚙️ Controles")
        
        st.subheader("📊 Datos")
        
        if st.button("🔄 Recargar MovieLens", type="secondary", key="reload_data_button"):
            with st.spinner("Recargando datos..."):
                if load_data(force_reload=True):
                    if train_models():
                        st.session_state.historical_ratings_loaded[st.session_state.current_user_id] = False
                        load_user_historical_ratings(st.session_state.current_user_id)
                        st.success("Datos y modelos recargados")
                        st.rerun()
        
        if st.session_state.data_loaded:
            st.success("✅ Datos cargados")
            st.caption(f"📊 {len(st.session_state.movies_df):,} películas disponibles")
            
            st.divider()
            st.subheader("👤 Usuario")
            
            old_user_id = st.session_state.current_user_id
            user_id = st.number_input(
                "ID de usuario",
                min_value=1,
                max_value=10000,
                value=st.session_state.current_user_id,
                step=1,
                key="user_id_input"
            )
            
            if user_id != old_user_id:
                st.session_state.current_user_id = user_id
                st.session_state.historical_ratings_loaded[user_id] = False
                load_user_historical_ratings(user_id)
                st.rerun()
            
            if st.session_state.feedback:
                summary = st.session_state.feedback.get_ratings_summary(user_id)
                if summary and summary['total_ratings'] > 0:
                    st.info(f"📊 {summary['total_ratings']} películas calificadas")
                    st.info(f"⭐ Promedio: {summary['avg_rating']:.1f}")
            
            st.divider()
            st.subheader("Recomendación")
            
            st.session_state.top_n = st.slider(
                "Número de recomendaciones",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                key="top_n_slider"
            )
            
            # ========== PONDERACIÓN HÍBRIDA ==========
            st.markdown("#### ⚖️ Ponderación Híbrida")
            
            # Obtener valores actuales
            if 'content_weight' not in st.session_state:
                st.session_state.content_weight = 0.6
                st.session_state.collab_weight = 0.4
            
            current_content = st.session_state.content_weight
            
            # Slider de Contenido
            new_content = st.slider(
                "🎬 Peso de Contenido",
                min_value=0.0,
                max_value=1.0,
                value=current_content,
                step=0.05,
                key="content_weight_slider",
                help="Recomienda películas similares a las que te gustan"
            )
            
            # Calcular Colaborativo
            new_collab = 1.0 - new_content
            
            # Mostrar el valor de Colaborativo
            st.metric("👥 Peso Colaborativo", f"{new_collab:.0%}")
            
            # Actualizar si cambió
            if new_content != current_content:
                st.session_state.content_weight = new_content
                st.session_state.collab_weight = new_collab
                
                if st.session_state.hybrid_rec:
                    st.session_state.hybrid_rec.set_weights(new_content, new_collab)
                    st.success(f"✅ Ponderación actualizada: Contenido {new_content:.0%} | Colaborativo {new_collab:.0%}")
                    time.sleep(0.5)
                    st.rerun()
            
            # Explicación breve
            with st.expander("ℹ️ ¿Qué significa esto?"):
                st.markdown("""
                - **Contenido**: Recomienda películas similares a las que te gustan (mismos géneros, actores, director)
                - **Colaborativo**: Recomienda películas que a otros usuarios similares a ti les gustaron
                
                **Consejos:**
                - Si tienes pocas calificaciones: aumenta el peso de **Contenido**
                - Si tienes muchas calificaciones: prueba más **Colaborativo**
                - 60%/40% es un buen punto de partida
                """)
            
        else:
            if st.session_state.data_loading_in_progress:
                st.info("⏳ Cargando datos automáticamente...")
            else:
                st.warning("⚠️ Esperando carga automática...")


# ==================== PESTAÑA DE RECOMENDACIÓN ====================

def render_recommendation_tab(method: str):
    """Renderiza una pestaña de recomendación con selector de estrellas."""
    st.session_state.active_tab = method
    
    st.subheader(f"🔍 Recomendación por {method}")
    
    if st.session_state.movies_df is not None:
        total_movies = len(st.session_state.movies_df)
        st.caption(f"📊 Base de datos: {total_movies:,} películas disponibles")
    
    st.markdown("### 🎬 Buscar y calificar películas")
    
    # Selector de películas
    if st.session_state.movies_df is not None:
        movies_selector = st.session_state.movies_df.copy()
        
        def get_genres_safe(row):
            genres = row.get('genres_processed', [])
            if genres is None or isinstance(genres, float):
                return 'Sin género'
            if hasattr(genres, 'tolist'):
                genres = genres.tolist()
            if isinstance(genres, list) and len(genres) > 0:
                return ', '.join([str(g) for g in genres[:2] if g])
            return 'Sin género'
        
        movies_selector['display_text'] = movies_selector.apply(
            lambda row: f"{row.get('title_clean', row.get('title', 'Desconocido'))} ({row.get('year', 0)}) - {get_genres_safe(row)}",
            axis=1
        )
        
        movie_options = movies_selector['display_text'].tolist()
        movie_ids = movies_selector['movie_id'].tolist()
        movie_map = dict(zip(movie_options, movie_ids))
        
        selected_movie_display = st.selectbox(
            "🔎 Buscar película para calificar",
            options=movie_options,
            index=0,
            key=f"movie_selector_{method}"
        )
        
        selected_movie_id = movie_map.get(selected_movie_display)
    
    # Película seleccionada
    if selected_movie_id is not None:
        with st.spinner("🔍 Cargando datos de TMDB..."):
            enriched_movie = enrich_selected_movie(selected_movie_id, force_refresh=False)
        
        if enriched_movie is not None:
            user_id = st.session_state.current_user_id
            current_rating = None
            if st.session_state.feedback:
                current_rating = st.session_state.feedback.get_rating(user_id, selected_movie_id)
            
            st.markdown("### 📌 Calificar esta película")
            
            poster_path = enriched_movie.get('poster_path', '')
            
            col_poster, col_info = st.columns([1, 2])
            
            with col_poster:
                if poster_path and poster_path != '' and not pd.isna(poster_path):
                    poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown('''
                    <div style="background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%); border-radius: 12px; padding: 60px 20px; text-align: center;">
                        <div style="font-size: 48px;">🎬</div>
                        <div style="margin-top: 10px; color: #888;">Sin póster disponible</div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            with col_info:
                st.markdown(f"#### 🎬 {enriched_movie.get('title_clean', enriched_movie.get('title', 'Desconocido'))}")
                st.markdown(f"**📅 Año:** {enriched_movie.get('year', 'Desconocido')}")
                
                genres = enriched_movie.get('genres_processed', [])
                if hasattr(genres, 'tolist'):
                    genres = genres.tolist()
                if genres and len(genres) > 0:
                    genre_str = ', '.join([str(g) for g in genres if g])
                    st.markdown(f"**🎭 Géneros:** {genre_str}")
                
                director = enriched_movie.get('director', '')
                if director and director not in ['Pendiente de TMDB', 'No disponible', '', 'Desconocido', 'None']:
                    st.markdown(f"**🎬 Director:** {director}")
                
                actors = enriched_movie.get('actors', [])
                if hasattr(actors, 'tolist'):
                    actors = actors.tolist()
                if actors and len(actors) > 0:
                    st.markdown(f"**⭐ Actores:** {', '.join([str(a) for a in actors[:3] if a])}")
                
                st.markdown("---")
                st.markdown("**Tu calificación:**")
                
                col_stars = st.columns(5)
                
                for i in range(1, 6):
                    with col_stars[i-1]:
                        unique_key = f"detail_rating_{method}_{selected_movie_id}_{i}"
                        if st.button(f"{'⭐' * i} {i}", key=unique_key):
                            if st.session_state.feedback:
                                st.session_state.feedback.set_rating(user_id, selected_movie_id, i, source="manual")
                                st.rerun()
                
                if current_rating:
                    st.success(f"Tu calificación actual: {'⭐' * current_rating} ({current_rating}/5)")
                else:
                    st.info("Selecciona una calificación (1-5 estrellas)")
    
    # ========== HISTORIAL CON EXPANDER (SIMPLE Y FUNCIONAL) ==========
    with st.expander("📋 ABRIR HISTORIAL", expanded=False):
        render_history_popover_content(method)
    
    # Generar recomendaciones
    st.divider()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎯 GENERAR RECOMENDACIONES", type="primary", use_container_width=True, key=f"generate_btn_{method}"):
            user_likes = st.session_state.feedback.get_user_likes(st.session_state.current_user_id) if st.session_state.feedback else set()
            if len(user_likes) > 0:
                recommendations = generate_recommendations()
                if recommendations:
                    st.success(f"✅ {len(recommendations)} recomendaciones generadas")
                    st.rerun()
            else:
                st.warning("⚠️ Primero califica algunas películas con ⭐3, 4 o 5")
    
    # Mostrar recomendaciones
    if st.session_state.recommendation_details:
        st.markdown("### 🎬 Recomendaciones para ti")
        st.caption("💡 Califica las recomendaciones con ⭐1-5 para mejorar futuras sugerencias")
        
        total_recs = len(st.session_state.recommendation_details)
        st.caption(f"📊 Mostrando {total_recs} recomendaciones")
        
        cols = st.columns(3)
        
        for idx, movie in enumerate(st.session_state.recommendation_details):
            with cols[idx % 3]:
                poster_path = movie.get('poster_path', '')
                
                if poster_path and poster_path != '' and not pd.isna(poster_path):
                    poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%); border-radius: 12px; height: 200px; display: flex; align-items: center; justify-content: center;">
                        <div style="font-size: 48px;">🎬</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"**{movie['title']}** ({movie['year']})")
                
                genres = movie.get('genres', [])
                if isinstance(genres, list) and len(genres) > 0:
                    st.caption(f"🎭 {', '.join(genres[:2])}")
                
                score_percent = movie.get('score_percentage', 0)
                score_percent = max(0, min(100, score_percent))
                st.progress(score_percent / 100, text=f"Match: {score_percent:.1f}%")
                
                st.markdown("---")
                st.markdown("**Tu calificación:**")
                
                current_rating = movie.get('user_rating')
                col_stars = st.columns(5)
                
                for i in range(1, 6):
                    with col_stars[i-1]:
                        unique_key = f"rec_rating_{method}_{idx}_{movie['movie_id']}_{i}"
                        if st.button(f"{i}⭐", key=unique_key):
                            if st.session_state.feedback:
                                st.session_state.feedback.set_rating(
                                    st.session_state.current_user_id, 
                                    movie['movie_id'], 
                                    i, 
                                    source="recommendation"
                                )
                                st.rerun()
                
                if current_rating:
                    st.caption(f"Actual: {'⭐' * current_rating}")
                else:
                    st.caption("Sin calificar")
                
                st.divider()

def render_history_popover_content(method: str):
    """Contenido del popover del historial con selector de calificación horizontal ancho."""
    user_id = st.session_state.current_user_id
    
    if st.session_state.feedback is None:
        st.info("Sistema de calificaciones no disponible")
        return
    
    all_ratings = st.session_state.feedback.get_user_ratings(user_id)
    
    if not all_ratings:
        st.info("📭 No has calificado ninguna película todavía.")
        return
    
    base_key = f"{method}"
    
    # Inicializar páginas
    if f'popover_page_all_{base_key}' not in st.session_state:
        st.session_state[f'popover_page_all_{base_key}'] = 1
        st.session_state[f'popover_page_fav_{base_key}'] = 1
        st.session_state[f'popover_page_dis_{base_key}'] = 1
    
    # Filtros
    col_search, col_sort, col_asc = st.columns([2, 1.5, 0.8])
    with col_search:
        search_term = st.text_input("🔍 Buscar:", key=f"popover_search_{base_key}", placeholder="Título, año...")
    with col_sort:
        sort_by = st.selectbox("Ordenar por:", options=["rating", "title", "year"],
                               format_func=lambda x: "⭐ Rating" if x == "rating" else "📝 Título" if x == "title" else "📅 Año",
                               key=f"popover_sort_by_{base_key}")
    with col_asc:
        st.markdown("<br>", unsafe_allow_html=True)
        sort_ascending = st.checkbox("⬆️ Ascendente", key=f"popover_sort_asc_{base_key}")
    
    # Clasificar
    high_ratings = {m: r for m, r in all_ratings.items() if r >= 3}
    low_ratings = {m: r for m, r in all_ratings.items() if r <= 2}
    
    tab_all, tab_favorites, tab_disliked = st.tabs([
        f"📋 Todas ({len(all_ratings)})",
        f"⭐ Favoritas (3-5⭐) ({len(high_ratings)})",
        f"💔 No gustaron (1-2⭐) ({len(low_ratings)})"
    ])
    
    def render_tab(ratings_dict, page_key_suffix):
        if not ratings_dict:
            st.info("No hay películas en esta categoría")
            return
        
        movie_ids = list(ratings_dict.keys())
        movies_subset = st.session_state.movies_df[st.session_state.movies_df['movie_id'].isin(movie_ids)].copy()
        movies_subset['user_rating'] = movies_subset['movie_id'].map(ratings_dict)
        
        # Filtrar por búsqueda
        if search_term:
            search_lower = search_term.lower()
            movies_subset = movies_subset[
                movies_subset['title_clean'].str.lower().str.contains(search_lower, na=False) |
                movies_subset['year'].astype(str).str.contains(search_lower, na=False)
            ]
        
        # Ordenar
        if sort_by == 'title':
            movies_subset = movies_subset.sort_values('title_clean', ascending=sort_ascending)
        elif sort_by == 'year':
            movies_subset = movies_subset.sort_values('year', ascending=sort_ascending)
        elif sort_by == 'rating':
            movies_subset = movies_subset.sort_values('user_rating', ascending=sort_ascending)
        
        # Paginación
        items_per_page = 12
        total_items = len(movies_subset)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        page_key = f"popover_{page_key_suffix}_{base_key}"
        current_page = st.session_state.get(page_key, 1)
        if current_page > total_pages:
            current_page = total_pages
            st.session_state[page_key] = current_page
        
        start = (current_page - 1) * items_per_page
        end = start + items_per_page
        page_movies = movies_subset.iloc[start:end]
        
        # Forzar enriquecimiento
        for _, movie in page_movies.iterrows():
            movie_id = movie['movie_id']
            poster_path = movie.get('poster_path', '')
            if not poster_path or poster_path == '' or pd.isna(poster_path):
                try:
                    movie_series = pd.Series(movie)
                    enriched = enrich_movie_with_tmdb(movie_series, force_refresh=False)
                    poster_path = enriched.get('poster_path', '')
                    if poster_path:
                        idx = st.session_state.movies_df[st.session_state.movies_df['movie_id'] == movie_id].index
                        if len(idx) > 0:
                            st.session_state.movies_df.at[idx[0], 'poster_path'] = poster_path
                            movies_subset.loc[movies_subset['movie_id'] == movie_id, 'poster_path'] = poster_path
                except:
                    pass
        
        page_movies = movies_subset.iloc[start:end]
        
        # Controles de paginación
        if total_pages > 1:
            col_prev, col_info, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("◀ Anterior", key=f"prev_{page_key}"):
                    if current_page > 1:
                        st.session_state[page_key] = current_page - 1
                        st.rerun()
            with col_info:
                st.markdown(f"<div style='text-align:center'>📄 Página {current_page} de {total_pages}</div>", unsafe_allow_html=True)
            with col_next:
                if st.button("Siguiente ▶", key=f"next_{page_key}"):
                    if current_page < total_pages:
                        st.session_state[page_key] = current_page + 1
                        st.rerun()
        
        # Grid de 4 columnas
        for idx in range(0, len(page_movies), 4):
            cols = st.columns(4)
            for col_idx in range(4):
                if idx + col_idx < len(page_movies):
                    movie = page_movies.iloc[idx + col_idx]
                    movie_id = movie['movie_id']
                    rating = ratings_dict[movie_id]
                    
                    with cols[col_idx]:
                        poster_path = movie.get('poster_path', '')
                        
                        if poster_path and poster_path != '' and not pd.isna(poster_path):
                            clean_path = poster_path.replace('https://image.tmdb.org/t/p/w342', '')
                            clean_path = clean_path.replace('https://image.tmdb.org/t/p/w185', '')
                            clean_path = clean_path.replace('https://image.tmdb.org/t/p/w92', '')
                            poster_url = f"https://image.tmdb.org/t/p/w185{clean_path}"
                            st.image(poster_url, use_container_width=True)
                        else:
                            st.markdown("""
                            <div style="background:#2d2d3d; border-radius:8px; height:180px; display:flex; align-items:center; justify-content:center;">
                                <span style="font-size:40px;">🎬</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        title = movie.get('title_clean', movie.get('title', '?'))
                        st.markdown(f"**{title[:35]}**")
                        st.markdown(f"{'⭐' * rating} ({rating}/5)")
                        
                        director = movie.get('director', '')
                        if director and director not in ['Pendiente de TMDB', 'No disponible', 'Desconocido', None, '']:
                            if isinstance(director, (list, np.ndarray)):
                                director = ', '.join([str(d) for d in director[:2]])
                            st.caption(f"🎬 {str(director)[:25]}")
                        
                        st.markdown("---")
                        st.markdown("**Calificar:**")
                        
                        # Radio horizontal SIN COLUMNAS (ocupa todo el ancho disponible)
                        current_rating_val = rating
                        selected_rating = st.radio(
                            "",
                            options=[1, 2, 3, 4, 5],
                            format_func=lambda x: f"{x}⭐",
                            index=[1, 2, 3, 4, 5].index(current_rating_val) if current_rating_val in [1, 2, 3, 4, 5] else 2,
                            key=f"pop_radio_{page_key_suffix}_{base_key}_{movie_id}_{idx}_{col_idx}",
                            horizontal=True,
                            label_visibility="collapsed"
                        )
                        
                        if selected_rating != current_rating_val:
                            st.session_state.feedback.set_rating(user_id, movie_id, selected_rating, source="popover")
                            st.rerun()
                        
                        # Botón eliminar (ocupa todo el ancho)
                        if st.button("🗑️ Eliminar", key=f"pop_del_{page_key_suffix}_{base_key}_{movie_id}_{idx}_{col_idx}", use_container_width=True):
                            st.session_state.feedback.remove_rating(user_id, movie_id)
                            st.rerun()
                        
                        st.divider()
    
    with tab_all:
        render_tab(all_ratings, "page_all")
    with tab_favorites:
        render_tab(high_ratings, "page_fav")
    with tab_disliked:
        render_tab(low_ratings, "page_dis")
    
    # Exportar (sin botón de cerrar)
    col_exp1, col_exp2, col_exp3 = st.columns([1, 2, 1])
    with col_exp2:
        if st.button("📥 Exportar filtrado a Excel", key=f"popover_export_{base_key}", use_container_width=True):
            export_filtered_ratings(search_term, sort_by, sort_ascending)

# ==================== CONTENIDO PRINCIPAL ====================

def render_main_content():
    """Renderiza el contenido principal."""
    if not st.session_state.data_loaded:
        if not st.session_state.data_loading_in_progress:
            auto_load_and_train()
        else:
            st.info("⏳ Cargando MovieLens y entrenando modelos...")
            st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <div style="font-size: 48px;">🎬</div>
                <h3>Preparando el Sistema de Recomendación</h3>
                <p>Esto puede tomar unos segundos la primera vez...</p>
            </div>
            """, unsafe_allow_html=True)
        return
    
    # Crear las pestañas
    tab1, tab2, tab3 = st.tabs(["🎯 Híbrido", "📝 Contenido", "👥 Colaborativo"])
    
    with tab1:
        # Actualizar active_tab sin rerun
        st.session_state.active_tab = "Híbrido"
        render_recommendation_tab("Híbrido")
    
    with tab2:
        st.session_state.active_tab = "Contenido"
        render_recommendation_tab("Contenido")
    
    with tab3:
        st.session_state.active_tab = "Colaborativo"
        render_recommendation_tab("Colaborativo")

# ==================== MAIN ====================

def main():
    st.title("🎬 Sistema de Recomendación de Películas")
    st.caption("⭐ Califica películas con 1-5 estrellas y recibe recomendaciones personalizadas")
    
    init_session_state()
    
    # Inicializar la variable de pestaña si no existe
    if 'current_tab_for_sidebar' not in st.session_state:
        st.session_state.current_tab_for_sidebar = "Híbrido"
    
    render_sidebar()  # Sin parámetros ahora
    render_main_content()
    
    st.divider()
    st.caption("🚀 MovieLens + TMDB bajo demanda | Calificaciones ⭐1-5")


if __name__ == "__main__":
    main()
