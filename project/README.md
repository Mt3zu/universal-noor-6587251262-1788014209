# SeoraAI-1

A real, trainable decoder-only Transformer starter model.

## Run
python -m venv .venv
# activate the environment
pip install -r requirements.txt
python train.py
uvicorn api.server:app --reload --port 8000

## API
GET /health
GET /v1/models
POST /v1/chat/completions

This is a starter model architecture, not a GPT-5-scale model. Training quality depends on dataset size, compute, tokenizer, hyperparameters, and training time.
