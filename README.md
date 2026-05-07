# 🎬 Sistema de Recomendación de Películas

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Code style](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Sistema profesional de recomendación de películas que combina **filtrado basado en contenido**, **filtrado colaborativo** y **recomendación híbrida** con pesos configurables por el usuario. Incluye interfaz gráfica con Streamlit, sistema de calificación por estrellas (⭐1-5), historial persistente, exportación a Excel y carga bajo demanda de pósters desde TMDB.

---

## ✨ Características Principales

| Característica | Descripción |
|----------------|-------------|
| 🎯 **Tres Métodos** | Contenido, Colaborativo e Híbrido con pesos ajustables |
| ⭐ **Calificación 1-5** | Sistema completo de estrellas con persistencia |
| 📊 **Historial Visual** | Ver todas tus películas calificadas con filtros y búsqueda |
| 🖼️ **Pósters TMDB** | Carga bajo demanda de imágenes desde TMDB |
| 📥 **Exportación Excel** | Guarda tu historial con director, actores y estadísticas |
| 🔄 **Feedback Online** | Las calificaciones mejoran recomendaciones en tiempo real |
| 💾 **Caché Inteligente** | Matrices de similitud guardadas en disco |
| 📝 **Logs Expandibles** | Depuración en tiempo real desde la interfaz |

---

## 🏗️ Estructura del Proyecto

```
movie-recommender/
│
├── app.py                      # Aplicación principal Streamlit
├── run.ps1                     # Script de gestión para Windows
├── requirements.txt            # Dependencias Python
├── .env.example                # Plantilla de variables de entorno
├── .gitignore                  # Archivos ignorados por git
│
├── src/                        # Módulos del sistema
│   ├── utils.py                # Logging, normalización, utilidades
│   ├── data_loader.py          # Carga de datasets (MovieLens)
│   ├── preprocess.py           # Limpieza y preparación de datos
│   ├── content_based.py        # Similitud coseno por contenido
│   ├── collaborative.py        # Filtro colaborativo usuario-usuario
│   ├── hybrid.py               # Combinación híbrida ponderada
│   ├── feedback.py             # Sistema de calificaciones ⭐1-5
│   ├── tmdb_enricher.py        # Enriquecimiento bajo demanda
│   └── unified_data.py         # Unificación de fuentes de datos
│
├── tests/                      # Tests unitarios
│   ├── test_content_based.py
│   ├── test_collaborative.py
│   └── test_feedback.py
│
├── data/                       # Datasets descargados (MovieLens)
├── models/                     # Matrices de similitud precalculadas
├── logs/                       # Archivos de log de la aplicación
└── exports/                    # Exportaciones Excel de usuarios
```

---

## 🚀 Instalación Rápida

### Requisitos Previos
- **Python 3.9 o superior**
- **Git** (opcional, para clonar)
- **Conexión a internet** (para descargar MovieLens)

### Windows

```powershell
# 1. Clonar o descargar el repositorio
git clone https://github.com/tuusuario/movie-recommender.git
cd movie-recommender

# 2. Crear estructura de directorios
.\run.ps1 setup-data

# 3. Instalar dependencias
.\run.ps1 install

# 4. (Opcional) Configurar TMDB API Key
copy .env.example .env
# Editar .env y añadir TMDB_API_KEY=tu_clave

# 5. Iniciar la aplicación
.\run.ps1 run
```

### Linux / macOS

```bash
# 1. Clonar el repositorio
git clone https://github.com/tuusuario/movie-recommender.git
cd movie-recommender

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Crear directorios
mkdir -p data models logs exports

# 5. (Opcional) Configurar TMDB
cp .env.example .env

# 6. Iniciar la aplicación
streamlit run app.py
```

---

## 🎮 Uso de la Aplicación

### 1. ⭐ Calificar Películas

```python
# Desde la interfaz:
# - Busca una película en el selector
# - Haz clic en 1-5 estrellas para calificarla
# - Las calificaciones 3-5⭐ se consideran "favoritas"
# - Las calificaciones 1-2⭐ se penalizan en recomendaciones
```

### 2. 🎯 Generar Recomendaciones

```python
# Según el método seleccionado:
# - Contenido: Basado en similitud de géneros/actores/director
# - Colaborativo: Basado en usuarios con gustos similares
# - Híbrido: Combinación ponderada (ajustable en sidebar)
```

### 3. ⚖️ Ajustar Ponderación Híbrida

```python
# En la barra lateral:
# - Desliza "Peso de Contenido" (0% a 100%)
# - El peso colaborativo se calcula automáticamente
# - Consejo: Si tienes pocas calificaciones, aumenta contenido
```

### 4. 📋 Ver Historial

```python
# Haz clic en "📋 Mi Historial" para:
# - Ver todas tus películas calificadas
# - Filtrar por búsqueda (título, año, director, actor)
# - Ordenar por rating, título o año
# - Exportar a Excel con todos los detalles
```

### 5. 📥 Exportar a Excel

```python
# El archivo Excel incluye:
# - Hoja "Mi Historial": ID, Título, Año, Rating, Director, Actores, Géneros
# - Hoja "Estadísticas": Resumen con métricas agregadas
# - Las películas se enriquecen con datos de TMDB bajo demanda
```

---

## 🔬 Algoritmos y Técnicas

### 📝 Content-Based (Basado en Contenido)

```python
# Características utilizadas:
# - Géneros (ponderación 2x)
# - Actores principales (top 5)
# - Director
# - Año de lanzamiento

# Implementación:
# 1. CountVectorizer para convertir texto a vectores TF
# 2. Similitud coseno entre vectores
# 3. Ponderación por rating del usuario (1-5⭐)
# 4. Penalización de ratings bajos (1-2⭐)
```

### 👥 Collaborative (Filtro Colaborativo)

```python
# Matriz usuario-película:
# - Dimensiones: N_usuarios × N_películas
# - Valores: ratings de 1 a 5 (0 si no visto)

# Cálculo de similitud:
# - Métrica: Pearson o Coseno
# - Mínimo de películas en común: 3

# Predicción:
# rating_predicho = Σ(similitud × rating) / Σ(similitud)
```

### 🎯 Hybrid (Recomendación Híbrida)

```python
# Estrategias disponibles:
# - weighted: Promedio ponderado (configurable)
# - adaptive: Peso dinámico según cantidad de ratings
# - max: Toma el máximo score de ambos métodos

# Fórmula (weighted):
# score_hibrido = w_content × score_content + w_collab × score_collab
# donde w_content + w_collab = 1
```

---

## 📊 Fuentes de Datos

| Fuente | Descripción | Tamaño | Uso |
|--------|-------------|--------|-----|
| **MovieLens Small** | Ratings de usuarios reales | ~10MB | Principal (recomendaciones) |
| **TMDB API** | Metadatos y pósters | Bajo demanda | Enriquecimiento y visuales |

### MovieLens Small (default)
- **9742** películas
- **100,836** ratings
- **610** usuarios
- Escala de ratings: 0.5 a 5.0 (convertido a ⭐1-5)

---

## 🛠️ Comandos Útiles (Windows)

```powershell
.\run.ps1 install     # Instalar dependencias
.\run.ps1 run         # Iniciar aplicación
.\run.ps1 test        # Ejecutar tests
.\run.ps1 clean       # Limpiar archivos temporales
.\run.ps1 status      # Ver estado del proyecto
.\run.ps1 reset-data  # Reiniciar datos de MovieLens
.\run.ps1 reset-models# Reiniciar matrices precalculadas
.\run.ps1 help        # Mostrar ayuda completa
```

---

## 🔧 Configuración Avanzada

### Variables de entorno (`.env`)

```env
# TMDB API Key (opcional, para pósters)
# Obtener en: https://www.themoviedb.org/signup
TMDB_API_KEY=tu_api_key_aqui

# Configuración por defecto
DEFAULT_TOP_N=10
DEFAULT_CONTENT_WEIGHT=0.6
DEFAULT_COLLAB_WEIGHT=0.4

# Directorios
DATA_DIR=./data
MODELS_DIR=./models
LOGS_DIR=./logs
```

### Parámetros de los Algoritmos

```python
# En collaborative.py
min_common_movies = 3      # Mínimo películas en común
similarity_metric = 'cosine'  # 'cosine' o 'pearson'

# En content_based.py  
low_rating_penalty = 0.1   # Penalización para ratings 1-2⭐
decay_factor = 0.8         # Decaimiento temporal

# En hybrid.py
strategy = 'weighted'      # 'weighted', 'adaptive', 'max'
use_fallback = True        # Usar solo contenido si falla colaborativo
```

---

## 📈 Rendimiento

| Operación | Tiempo (aprox) | Memoria |
|-----------|---------------|---------|
| Carga inicial MovieLens | 2-5 segundos | ~50MB |
| Entrenamiento contenido | 10-30 segundos | ~200MB |
| Entrenamiento colaborativo | 5-10 segundos | ~100MB |
| Recomendación (cacheado) | <100ms | - |
| Enriquecimiento TMDB | 1-2 segundos/película | - |

---

## 🧪 Tests

```powershell
# Ejecutar todos los tests
.\run.ps1 test

# Con cobertura
pytest tests/ --cov=src --cov-report=html

# Test específico
pytest tests/test_feedback.py -v
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. **Fork** el proyecto
2. **Crea una rama** (`git checkout -b feature/nueva-funcionalidad`)
3. **Commit** los cambios (`git commit -m 'feat: añadir nueva funcionalidad'`)
4. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
5. **Abre un Pull Request**

### Convención de Commits

```
feat: Nueva funcionalidad
fix: Corrección de bug
docs: Documentación
test: Tests
refactor: Refactorización
perf: Mejora de rendimiento
style: Formato de código
chore: Mantenimiento
```

---

## 📄 Licencia

MIT License - Ver archivo [LICENSE](LICENSE)

---

## 🙏 Agradecimientos

- **[GroupLens](https://grouplens.org/datasets/movielens/)** - Dataset MovieLens
- **[TMDB](https://www.themoviedb.org/)** - API de películas y pósters
- **[Streamlit](https://streamlit.io/)** - Framework de interfaz
- **[scikit-learn](https://scikit-learn.org/)** - Implementaciones de ML

---

## 📧 Contacto

**Autor**: Tu Nombre
- **GitHub**: [@tusuario](https://github.com/tusuario)
- **Email**: tuemail@ejemplo.com

**Proyecto**: [github.com/tusuario/movie-recommender](https://github.com/tusuario/movie-recommender)

---

## ⭐️ ¡No olvides darle una estrella al repo si te gustó!



---
? **Desarrollado por Christian Lera** | [GitHub](https://github.com/ChristianLera) | Proyecto de portfolio personal
