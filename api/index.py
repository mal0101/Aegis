"""Vercel serverless function entry point for the FastAPI backend."""
import sys
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, "backend"))

from app.main import app
