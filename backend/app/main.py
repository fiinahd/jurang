from fastapi import FastAPI
from .routers import upload, select_rows, clean, extract, label, train, sentiment

app = FastAPI(title="Jurang ABSA Backend")

app.include_router(upload.router)
app.include_router(select_rows.router)
app.include_router(clean.router)
app.include_router(extract.router)
app.include_router(label.router)
app.include_router(train.router)
app.include_router(sentiment.router)

@app.get("/", tags=["root"] )
def root():
    return {"message": "Jurang ABSA API"}