@echo off
REM ==============================================================================
REM Script de Execucao do Sistema de Contagem Aerea de Bovinos
REM ==============================================================================

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual .venv nao encontrado!
    echo Execute: python -m venv .venv e instale os pacotes.
    pause
    exit /b 1
)

IF NOT EXIST "input_video.mp4" (
    echo [AVISO] O arquivo 'input_video.mp4' nao foi encontrado na pasta.
    echo Rodando com o video de teste sintetico 'test_drone_video.mp4'...
    echo.
    .\.venv\Scripts\python.exe main.py --input test_drone_video.mp4 --output test_annotated_output.mp4
) ELSE (
    echo [INICIANDO] Processando 'input_video.mp4'...
    echo.
    .\.venv\Scripts\python.exe main.py --input input_video.mp4 --output output_annotated_video.mp4 --show
)

echo.
echo [CONCLUIDO] Pressione qualquer tecla para sair...
pause >nul
