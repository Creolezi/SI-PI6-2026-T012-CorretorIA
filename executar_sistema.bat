@echo off
chcp 65001 > nul
title CorretorIA - Sistema de Precificação Imobiliária (SI-PI6)
color 0A

echo ======================================================================
echo       CorretorIA - Inteligência de Precificação Imobiliária
echo                   Projeto Integrador VI (SI-PI6)
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Verificando ambiente Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERRO] Python não foi encontrado no sistema!
    echo Por favor, instale o Python 3.10 ou superior e marque a opcao "Add Python to PATH".
    echo.
    pause
    exit /b
)

echo [2/3] Verificando dependências necessárias...
python -c "import fastapi, uvicorn, sklearn, pandas, numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo Instalando bibliotecas do requirements.txt...
    python -m pip install -r requirements.txt
) else (
    echo Ambiente e bibliotecas prontos!
)

echo.
echo [3/3] Iniciando a aplicação...
echo.
echo ======================================================================
echo  Painel do Corretor : http://localhost:8000
echo  Documentacao Swagger: http://localhost:8000/docs
echo ======================================================================
echo.
echo Abrindo o navegador automaticamente em http://localhost:8000...
echo Para encerrar o sistema, pressione CTRL + C ou feche esta janela.
echo.

start "" "http://localhost:8000"

python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
