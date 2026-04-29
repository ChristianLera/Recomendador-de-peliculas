# run.ps1 - Script de gestion para Windows
# Uso: .\run.ps1 [comando]

param(
    [Parameter(Position=0)]
    [ValidateSet("install", "test", "run", "clean", "setup-data", "reset-data", "reset-models", "status", "requirements", "help")]
    [string]$command = "help"
)

$ErrorActionPreference = "Continue"

# Funciones de colores
function Write-Step { Write-Host "`n===> $($args[0])" -ForegroundColor Green }
function Write-Warning { Write-Host "WARNING: $($args[0])" -ForegroundColor Yellow }
function Write-Error { Write-Host "ERROR: $($args[0])" -ForegroundColor Red }
function Write-Success { Write-Host "SUCCESS: $($args[0])" -ForegroundColor Green }
function Write-Info { Write-Host "INFO: $($args[0])" -ForegroundColor Cyan }

function Test-Command {
    param($cmd)
    $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

function Test-ProjectRoot {
    if (-not (Test-Path "app.py")) {
        Write-Error "No se encuentra app.py"
        Write-Info "Ubicacion actual: $(Get-Location)"
        return $false
    }
    return $true
}

switch ($command) {
    "install" {
        Write-Step "Instalando dependencias..."
        
        if (-not (Test-Command "python")) {
            Write-Error "Python no esta instalado"
            Write-Info "Descarga Python desde: https://www.python.org/downloads/"
            exit 1
        }
        
        Write-Info "Python version: $(python --version)"
        Write-Step "Actualizando pip..."
        python -m pip install --upgrade pip
        Write-Step "Instalando dependencias desde requirements.txt..."
        pip install -r requirements.txt
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Dependencias instaladas correctamente"
        } else {
            Write-Error "Error instalando dependencias"
            exit 1
        }
    }
    
    "test" {
        Write-Step "Ejecutando tests..."
        
        if (-not (Test-Command "pytest")) {
            Write-Warning "pytest no esta instalado. Instalando..."
            pip install pytest pytest-cov
        }
        
        pytest tests/ -v --cov=src --cov-report=term-missing
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Tests completados exitosamente"
        } else {
            Write-Error "Algunos tests fallaron"
            exit 1
        }
    }
    
    "run" {
        if (-not (Test-ProjectRoot)) { exit 1 }
        
        Write-Step "Iniciando aplicacion Streamlit..."
        
        if (-not (Test-Path ".env")) {
            Write-Warning "No se encontro archivo .env"
            if (Test-Path ".env.example") {
                Copy-Item ".env.example" ".env"
                Write-Success "Archivo .env creado desde .env.example"
            } else {
                Write-Info "Creando archivo .env basico..."
                $envLines = @(
                    "# TMDB API Key - opcional para posters",
                    "TMDB_API_KEY=",
                    "",
                    "# Configuracion por defecto",
                    "DEFAULT_TOP_N=10",
                    "DEFAULT_CONTENT_WEIGHT=0.6",
                    "DEFAULT_COLLAB_WEIGHT=0.4",
                    "",
                    "# Directorios",
                    "DATA_DIR=./data",
                    "MODELS_DIR=./models",
                    "LOGS_DIR=./logs"
                )
                $envLines -join "`r`n" | Out-File -FilePath ".env" -Encoding utf8
                Write-Success "Archivo .env creado"
            }
        }
        
        $dirs = @("data", "models", "logs", "exports")
        foreach ($dir in $dirs) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Force -Path $dir | Out-Null
                Write-Info "Directorio creado: $dir"
            }
        }
        
        Write-Info "Abriendo http://localhost:8501 en tu navegador..."
        Write-Warning "Presiona Ctrl+C para detener la aplicacion"
        Write-Host ""
        
        if (Test-Command "streamlit") {
            streamlit run app.py --server.port 8501 --server.address localhost
        } else {
            python -m streamlit run app.py --server.port 8501 --server.address localhost
        }
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Error al iniciar Streamlit"
            exit 1
        }
    }
    
    "clean" {
        Write-Step "Limpiando archivos temporales..."
        
        Get-ChildItem -Path . -Include "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue | 
            Where-Object { $_.FullName -notlike "*\venv\*" } | 
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Info "Eliminados: __pycache__"
        
        $tempItems = @(".pytest_cache", ".coverage", "htmlcov", ".mypy_cache")
        foreach ($item in $tempItems) {
            if (Test-Path $item) {
                Remove-Item -Recurse -Force $item -ErrorAction SilentlyContinue
                Write-Info "Eliminado: $item"
            }
        }
        
        Write-Success "Limpieza completada"
    }
    
    "setup-data" {
        Write-Step "Creando estructura de directorios..."
        
        $directories = @("data", "models", "logs", "exports")
        foreach ($dir in $directories) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Force -Path $dir | Out-Null
                Write-Info "Directorio creado: $dir"
            } else {
                Write-Info "Directorio ya existe: $dir"
            }
        }
        
        Write-Success "Estructura de directorios lista"
    }
    
    "reset-data" {
        Write-Step "Reiniciando datos de MovieLens..."
        Write-Warning "Esto eliminara los datos descargados de MovieLens"
        
        $confirm = Read-Host "¿Estas seguro? s/N"
        if ($confirm -ne "s" -and $confirm -ne "S") {
            Write-Info "Operacion cancelada"
            exit 0
        }
        
        $dataFiles = @(
            "data/movielens_movies.parquet",
            "data/movielens_ratings.parquet",
            "data/ml-latest-small.zip",
            "data/tmdb_cache.json",
            "data/tmdb_enrichment_cache.json",
            "data/user_ratings.json"
        )
        
        foreach ($file in $dataFiles) {
            if (Test-Path $file) {
                Remove-Item -Force $file -ErrorAction SilentlyContinue
                Write-Info "Eliminado: $file"
            }
        }
        
        if (Test-Path "data/ml-latest-small") {
            Remove-Item -Recurse -Force "data/ml-latest-small" -ErrorAction SilentlyContinue
            Write-Info "Eliminado: data/ml-latest-small"
        }
        
        Write-Success "Datos reiniciados"
    }
    
    "reset-models" {
        Write-Step "Reiniciando modelos matrices de similitud..."
        Write-Warning "Esto eliminara las matrices precalculadas"
        
        $confirm = Read-Host "¿Estas seguro? s/N"
        if ($confirm -ne "s" -and $confirm -ne "S") {
            Write-Info "Operacion cancelada"
            exit 0
        }
        
        Get-ChildItem -Path "models" -Filter "*.joblib" -ErrorAction SilentlyContinue | ForEach-Object {
            Remove-Item -Force $_.FullName -ErrorAction SilentlyContinue
            Write-Info "Eliminado: models/$($_.Name)"
        }
        
        Write-Success "Modelos reiniciados"
    }
    
    "status" {
        Write-Step "Estado del proyecto"
        
        if (Test-Command "python") {
            Write-Success "Python: $(python --version)"
        } else {
            Write-Error "Python: No instalado"
        }
        
        Write-Step "Directorios:"
        @("data", "models", "logs", "exports") | ForEach-Object {
            if (Test-Path $_) { Write-Success "  $_/ existe" }
            else { Write-Warning "  $_/ no existe ejecuta setup-data" }
        }
        
        Write-Step "Configuracion:"
        if (Test-Path ".env") {
            Write-Success "  .env existe"
        } else {
            Write-Warning "  .env no existe"
        }
        
        Write-Success "Estado verificado"
    }
    
    "requirements" {
        Write-Step "Exportando requirements.txt actual..."
        pip freeze > requirements.freeze.txt
        Write-Success "Requirements exportados a requirements.freeze.txt"
    }
    
    "help" {
        Write-Host ""
        Write-Host "SISTEMA DE RECOMENDACION DE PELICULAS" -ForegroundColor Magenta
        Write-Host "==================================================" -ForegroundColor Magenta
        Write-Host ""
        Write-Host "COMANDOS DISPONIBLES:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  .\run.ps1 install     - Instalar todas las dependencias" -ForegroundColor Green
        Write-Host "  .\run.ps1 run         - Iniciar la aplicacion Streamlit" -ForegroundColor Green
        Write-Host "  .\run.ps1 test        - Ejecutar los tests unitarios" -ForegroundColor Green
        Write-Host "  .\run.ps1 clean       - Limpiar archivos temporales" -ForegroundColor Green
        Write-Host "  .\run.ps1 setup-data  - Crear estructura de directorios" -ForegroundColor Green
        Write-Host "  .\run.ps1 reset-data  - Eliminar datos descargados" -ForegroundColor Green
        Write-Host "  .\run.ps1 reset-models- Eliminar matrices precalculadas" -ForegroundColor Green
        Write-Host "  .\run.ps1 status      - Mostrar estado del proyecto" -ForegroundColor Green
        Write-Host "  .\run.ps1 requirements- Exportar requirements actuales" -ForegroundColor Green
        Write-Host "  .\run.ps1 help        - Mostrar esta ayuda" -ForegroundColor Green
        Write-Host ""
        Write-Host "EJEMPLOS:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  Instalacion inicial:" -ForegroundColor Cyan
        Write-Host "  .\run.ps1 setup-data" -ForegroundColor White
        Write-Host "  .\run.ps1 install" -ForegroundColor White
        Write-Host ""
        Write-Host "  Ejecutar la aplicacion:" -ForegroundColor Cyan
        Write-Host "  .\run.ps1 run" -ForegroundColor White
        Write-Host ""
        Write-Host "  Verificar estado:" -ForegroundColor Cyan
        Write-Host "  .\run.ps1 status" -ForegroundColor White
        Write-Host ""
        Write-Host "NOTAS:" -ForegroundColor Yellow
        Write-Host "  - La primera ejecucion descargara MovieLens unos 10MB" -ForegroundColor White
        Write-Host "  - TMDB_API_KEY es opcional solo para posters" -ForegroundColor White
        Write-Host "  - Los modelos se guardan en carpeta models para reutilizacion" -ForegroundColor White
        Write-Host ""
    }
    
    default {
        Write-Host "Comando no reconocido: $command" -ForegroundColor Red
        Write-Host "Ejecuta '.\\run.ps1 help' para ver los comandos disponibles" -ForegroundColor Yellow
        exit 1
    }
}