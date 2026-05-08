# run.ps1 - Script de gestion simple

# Verificar y solicitar administrador
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Solicitando permisos de Administrador..." -ForegroundColor Yellow
    Start-Process powershell.exe "-NoExit -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

Clear-Host
Write-Host ""
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host "  SISTEMA DE RECOMENDACION DE PELICULAS" -ForegroundColor Magenta
Write-Host "  Modo Administrador Activado" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

# Menu principal
$opcion = ""
while ($opcion -ne "0") {
    Write-Host ""
    Write-Host "QUE DESEAS HACER?" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " 1. Instalar dependencias" -ForegroundColor Green
    Write-Host " 2. Iniciar aplicacion" -ForegroundColor Green
    Write-Host " 3. Ejecutar tests" -ForegroundColor Green
    Write-Host " 4. Ver estado" -ForegroundColor Green
    Write-Host " 5. Crear directorios (setup-data)" -ForegroundColor Green
    Write-Host " 6. Limpiar archivos temporales" -ForegroundColor Green
    Write-Host " 7. Reiniciar datos" -ForegroundColor Green
    Write-Host " 8. Reiniciar modelos" -ForegroundColor Green
    Write-Host " 0. Salir" -ForegroundColor Red
    Write-Host ""
    
    $opcion = Read-Host "Elige una opcion (0-8)"
    
    if ($opcion -eq "1") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Instalando dependencias..." -ForegroundColor Cyan
        python --version
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        Write-Host "Instalacion completada" -ForegroundColor Green
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "2") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Iniciando aplicacion..." -ForegroundColor Cyan
        $dirs = @("data", "models", "logs", "exports")
        foreach ($dir in $dirs) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Force -Path $dir | Out-Null
            }
        }
        Write-Host "Abriendo http://localhost:8501" -ForegroundColor Green
        streamlit run app.py --server.port 8501 --server.address localhost
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "3") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Ejecutando tests..." -ForegroundColor Cyan
        pip install pytest pytest-cov
        pytest tests/ -v
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "4") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Estado del proyecto:" -ForegroundColor Cyan
        python --version
        Write-Host "" -ForegroundColor Yellow
        Write-Host "Directorios:" -ForegroundColor Yellow
        $dirs = @("data", "models", "logs", "exports")
        foreach ($dir in $dirs) {
            if (Test-Path $dir) {
                Write-Host "  [OK] $dir/ existe" -ForegroundColor Green
            } else {
                Write-Host "  [NO] $dir/ no existe" -ForegroundColor Red
            }
        }
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "5") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Creando directorios..." -ForegroundColor Cyan
        $dirs = @("data", "models", "logs", "exports")
        foreach ($dir in $dirs) {
            New-Item -ItemType Directory -Force -Path $dir | Out-Null
            Write-Host "  Creado: $dir/" -ForegroundColor Green
        }
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "6") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Limpiando archivos temporales..." -ForegroundColor Cyan
        Get-ChildItem -Path . -Include "__pycache__" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Limpieza completada" -ForegroundColor Green
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "7") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Reiniciando datos..." -ForegroundColor Cyan
        $confirm = Read-Host "Esta seguro? (s/N)"
        if ($confirm -eq "s") {
            Remove-Item -Path "data/*" -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "Datos eliminados" -ForegroundColor Green
        }
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "8") {
        Write-Host "" -ForegroundColor Cyan
        Write-Host "Reiniciando modelos..." -ForegroundColor Cyan
        $confirm = Read-Host "Esta seguro? (s/N)"
        if ($confirm -eq "s") {
            Remove-Item -Path "models/*.joblib" -Force -ErrorAction SilentlyContinue
            Write-Host "Modelos eliminados" -ForegroundColor Green
        }
        Read-Host "`nPresiona Enter para continuar"
        Clear-Host
    }
    
    if ($opcion -eq "0") {
        Write-Host "" -ForegroundColor Magenta
        Write-Host "Hasta luego!" -ForegroundColor Magenta
        break
    }
    
    if ($opcion -ne "0" -and $opcion -ne "1" -and $opcion -ne "2" -and $opcion -ne "3" -and $opcion -ne "4" -and $opcion -ne "5" -and $opcion -ne "6" -and $opcion -ne "7" -and $opcion -ne "8") {
        Write-Host "Opcion no valida" -ForegroundColor Red
        Start-Sleep -Seconds 1
        Clear-Host
    }
}
