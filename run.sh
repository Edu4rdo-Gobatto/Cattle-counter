#!/usr/bin/env bash
# ==============================================================================
# Execução do Sistema de Contagem Aérea de Bovinos (Linux) em modo apresentação.
# Uso: ./run.sh [video.mp4]   (padrão: input_video.mp4, video-gado.mp4 ou gado-teste.mp4)
# ==============================================================================
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "[ERRO] Ambiente virtual .venv nao encontrado!"
    echo "Execute: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

VIDEO="$1"
if [ -z "$VIDEO" ]; then
    for candidato in input_video.mp4 video-gado.mp4 gado-teste.mp4; do
        if [ -f "$candidato" ]; then VIDEO="$candidato"; break; fi
    done
fi
if [ -z "$VIDEO" ] || [ ! -f "$VIDEO" ]; then
    echo "[ERRO] Nenhum video encontrado. Uso: ./run.sh caminho/do/video.mp4"
    exit 1
fi

echo "[INICIANDO] Processando '$VIDEO' com exibicao ao vivo..."
.venv/bin/python main.py --input "$VIDEO" --output output_annotated_video.mp4 --show
