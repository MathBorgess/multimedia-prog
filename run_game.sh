#!/bin/bash

# Script para executar o jogo de memória e coordenação
# Este script ativa o ambiente virtual e executa o jogo

echo "🎮 Iniciando Jogo de Memória e Coordenação"
echo "=========================================="

# Verificar se o ambiente virtual existe
if [ ! -d ".venv" ]; then
    echo "❌ Ambiente virtual não encontrado!"
    echo "Execute: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source .venv/bin/activate

# Verificar se as dependências estão instaladas
echo "📦 Verificando dependências..."
python -c "import cv2, mediapipe, numpy, pygame" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependências não encontradas! Instalando..."
    pip install -r requirements.txt
fi

# Verificar se os sons existem
if [ ! -f "sounds/success.wav" ]; then
    echo "🔊 Gerando arquivos de som..."
    python generate_sounds.py
fi

echo "🚀 Iniciando o jogo..."
echo "💡 Instruções:"
echo "   - Memorize a sequência de cores nas bordas"
echo "   - Toque nas áreas coloridas na ordem correta"
echo "   - Use a mão indicada (L = esquerda, R = direita)"
echo "   - Pressione 'q' para sair"
echo ""

# Executar o jogo
python __main__.py
