@echo off
REM ==============================================================================
REM Script de Execucao do Sistema de Contagem Aerea de Bovinos (modo apresentacao)
REM Uso: run.bat [video.mp4]
REM Padrao: input_video.mp4, video-gado.mp4 ou gado-teste.mp4 (o primeiro que existir)
REM ==============================================================================

cd /d "%~dp0"

IF NOT EXIST ".venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual .venv nao encontrado!
    echo Execute: python -m venv .venv e .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

SET "VIDEO=%~1"
IF "%VIDEO%"=="" IF EXIST "input_video.mp4" SET "VIDEO=input_video.mp4"
IF "%VIDEO%"=="" IF EXIST "video-gado.mp4" SET "VIDEO=video-gado.mp4"
IF "%VIDEO%"=="" IF EXIST "gado-teste.mp4" SET "VIDEO=gado-teste.mp4"
IF "%VIDEO%"=="" IF EXIST "gado.webm" SET "VIDEO=gado.webm"

IF "%VIDEO%"=="" (
    echo [ERRO] Nenhum video encontrado.
    echo Coloque 'input_video.mp4' na pasta ou rode: run.bat caminho\do\video.mp4
    pause
    exit /b 1
)

echo [INICIANDO] Processando '%VIDEO%' com exibicao ao vivo...
echo [ESPACO] pausa ^| [F] tela cheia ^| [Q]/[ESC] encerra
echo.
.\.venv\Scripts\python.exe main.py --input "%VIDEO%" --output output_annotated_video.mp4 --show

echo.
echo [CONCLUIDO] Video anotado salvo em output_annotated_video.mp4
echo Pressione qualquer tecla para sair...
pause >nul
