# AI Build - AI Automation & ML Platform

A collection of AI automation workflows and ML services built on n8n, Ollama, and Flask.

## Components

- **Flask Prediction API** (`app.py`) - Fraud detection ML model served via REST API
- **n8n Workflows** (`workflows/`) - 17 automation workflows for AI agents, RAG, SEO, data analysis, and more
- **Infrastructure** (`infrastructure-ia-locale/`) - Docker Compose stack with n8n, PostgreSQL, Qdrant, and Ollama

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- (Optional) NVIDIA GPU for accelerated inference

### Infrastructure Setup

1. Copy the environment template:
   ```bash
   cp infrastructure-ia-locale/.env.example infrastructure-ia-locale/.env
   # Edit .env with your own credentials
   ```
2. Start the stack:
   ```bash
   cd infrastructure-ia-locale
   docker compose --profile cpu up -d
   ```
3. Access n8n at http://localhost:5678

### Flask API Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API:
   ```bash
   python app.py
   ```
3. Test: `POST http://localhost:5000/predict` with JSON body `{"features": [...]}`

## Workflows

| Workflow | Description |
|----------|-------------|
| Agent_IA_Prediction_Fraudes_Bancaires | AI agent for bank fraud prediction |
| Detection_des_Fraudes_Input | Fraud detection input processing |
| Machine_Learning | ML workflow integration |
| Data_Insight_Rapide | Quick data analytics agent |
| RAG_pdf_qdrant | RAG pipeline with PDF and Qdrant |
| Eleven_Labs_RAG_Agent | RAG with ElevenLabs voice |
| SEO_IA | SEO automation with AI |
| Multi_Agent_IA_Recherche | Multi-agent research system |
| Agent_Visual_Scraping_HTML | Visual web scraping agent |
| Apollo_Leads_Building | Lead generation workflow |
| Personnal_Agent_on_Steroid | Enhanced personal assistant |
| Workflow_Youtube | YouTube automation |
| data_analyst_ai_agent | Data analysis agent |
| mixture_of_agent | Multi-agent orchestration |
| generation_doc | Document generation |

See `workflows/` for all workflow files.

## Project Structure

```
ai-build/
  app.py                          # Flask ML prediction API
  requirements.txt                # Python dependencies
  FraudDetection (1).ipynb        # Fraud detection notebook
  workflows/                      # n8n workflow exports (JSON)
  infrastructure-ia-locale/       # Docker Compose infrastructure
    docker-compose.yml
    .env.example
    README.md
```

## License

Apache License 2.0
