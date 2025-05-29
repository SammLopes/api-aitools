#!/bin/sh
if [ "$FIREBASE_STUDIO"="studio" ]; then 
    echo "🔧 Ambiente Firebase Studio detectado"
    
    source ./.venv/bin/activate
    pip install -r requirements.txt
    python -m flask --app main run --debug

else
    echo "🖥️ Ambiente local detectado"
     # Ativa o virtualenv se existir
    if [ -f ".venv/bin/activate" ]; then
        echo "👉 Ativando ambiente virtual .venv"
        . .venv/bin/activate
    else
        echo "⚠️ Ambiente virtual .venv não encontrado! Criando um novo..."
        python3 -m venv .venv && . .venv/bin/activate
    fi
    # Instala dependências
    if [ -f "requirements.txt" ]; then
      echo "📦 Instalando dependências do requirements.txt"
        pip install -r requirements.txt
    else
        echo "❌ Arquivo requirements.txt não encontrado!"
        exit 1
    fi
    # Inicia o servidor Flask
    echo "🚀 Iniciando servidor Flask local"
    python -m flask --app main run --debug
fi