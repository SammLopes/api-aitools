#!/bin/sh
source .venv/bin/activate
python install -r requirements.txt
python -m flask --app main run --debug