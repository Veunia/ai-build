# Workflow Generator

This is a minimal prototype for generating n8n workflows from an image, text description or YouTube URL.

## Backend

A FastAPI server exposes a `/generate` endpoint. Real OCR, language model and video processing should be added where noted.

```
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Frontend

Open `frontend/index.html` in a browser. Submit an image, text or YouTube URL to test the API.

This project is a skeleton and does not implement real AI processing yet.
