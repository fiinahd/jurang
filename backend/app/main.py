import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import analysis

# Membuat direktori yang dibutuhkan jika belum ada
os.makedirs("data", exist_ok=True)
os.makedirs("models_trained", exist_ok=True)

app = FastAPI(
    title="Aspect-Based Sentiment Analysis API",
    description="API untuk pipeline analisis sentimen berbasis aspek.",
    version="1.0.0"
)

# Konfigurasi CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Menyertakan router dari file analysis.py
app.include_router(analysis.router)

@app.get("/", tags=["Root"])
async def read_root():
    """Endpoint utama untuk mengecek apakah server berjalan."""
    return {"message": "Welcome to the ABSA API! Backend is running correctly."}