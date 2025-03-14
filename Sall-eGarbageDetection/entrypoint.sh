#!/bin/bash
echo "🚀 Starting FastAPI server with Uvicorn..."
exec python -m uvicorn app:app --host 0.0.0.0 --port 7860
