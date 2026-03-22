# CLAUDE.md - Project Context for Claude Code

## Project Overview
AI automation platform combining n8n workflows, a Flask ML API, and Docker-based infrastructure (PostgreSQL, Qdrant, Ollama).

## Key Commands
- Start infrastructure: `cd infrastructure-ia-locale && docker compose --profile cpu up -d`
- Run Flask API: `python app.py` (downloads model from Google Drive on first run)
- Install Python deps: `pip install -r requirements.txt`

## Architecture
- `app.py` - Flask REST API serving a scikit-learn fraud detection model via `/predict` endpoint
- `workflows/` - n8n workflow JSON exports (import via n8n UI or CLI)
- `infrastructure-ia-locale/` - Docker Compose stack: n8n, PostgreSQL 16, Qdrant, Ollama
- `FraudDetection (1).ipynb` - Training notebook for the fraud detection model

## Important Notes
- The `.env` file in `infrastructure-ia-locale/` contains secrets and is NOT committed (see `.env.example`)
- `model.pkl` is downloaded at runtime by `app.py` from Google Drive, not stored in the repo
- The project uses French in comments and some file names
- Docker Compose uses profiles: `cpu`, `gpu-nvidia`, `gpu-amd`
- n8n runs on port 5678, Flask API on port 5000, PostgreSQL on 5432, Qdrant on 6333, Ollama on 11434

## Code Style
- Python: standard Flask patterns, no type hints currently
- Comments are in French
- Workflow files are n8n JSON exports -- do not edit manually
