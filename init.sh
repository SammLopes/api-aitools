#!/bin/bash

APP_NAME="backend"
APP_PORT=3000
APP_MODULE="main:app"
LOG_FILE="logs.txt"
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    source $VENV_DIR/bin/activate
else
    python3 -m venv .venv
    source $VENV_DIR/bin/activate
fi

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else 
    exit 1
fi

[ ! -f "$LOG_FILE" ] && touch "$LOG_FILE"

gunicorn --workers 2 --bind 0.0.0.0:$APP_PORT $APP_MODULE > "$LOG_FILE" 2>&1 &
echo "✅ API disponível em http://<seu-ip>:$APP_PORT"
