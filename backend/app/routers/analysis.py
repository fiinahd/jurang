import os
import uuid
import pandas as pd
from fastapi import (
    APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
)
from fastapi.responses import FileResponse
from typing import List
import json

# Mengimpor modul dengan nama yang benar dari folder logic.
from ..logic import (
    l_1_preprocess as preprocess,
    l_2_postag as postag,
    l_3_extraction as extraction,
    l_4_training as training
)
from ..models import AspectSelection, LabelingPayload

router = APIRouter(
    prefix="/api/process",
    tags=["Analysis Pipeline"]
)

# --- TAHAP 1: UPLOAD & PREPROCESSING ---
@router.post("/start")
async def start_process(
    background_tasks: BackgroundTasks,
    review_column: str = Form(...),
    product_column: str = Form(...),
    file: UploadFile = File(...)
):
    process_id = str(uuid.uuid4())
    raw_file_path = os.path.join("data", f"raw_{process_id}.xlsx")
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")

    try:
        with open(raw_file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {e}")

    background_tasks.add_task(
        preprocess.run_preprocessing,
        input_path=raw_file_path,
        output_path=cleaned_file_path,
        review_column=review_column,
        product_column=product_column
    )
    return {"process_id": process_id, "message": "Preprocessing started."}

@router.get("/{process_id}/preprocess_result")
async def get_preprocess_result(process_id: str):
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    if not os.path.exists(cleaned_file_path):
        raise HTTPException(status_code=404, detail="File hasil preprocessing belum siap atau tidak ditemukan.")
    
    df = pd.read_csv(cleaned_file_path)
    df_preview = df.head(10).fillna('')
    preview = df_preview.to_dict(orient='records')
    columns = list(df.columns)

    return {"preview": {"columns": columns, "rows": preview}}


# --- TAHAP 2: POS TAGGING & PEMILIHAN ASPEK ---
@router.post("/{process_id}/postag")
async def run_pos_tagging_endpoint(process_id: str):
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    if not os.path.exists(cleaned_file_path):
        raise HTTPException(status_code=404, detail="File cleaned tidak ditemukan.")

    try:
        top_aspects = postag.run_postagging(cleaned_file_path, top_n=30)
        return {"aspects": top_aspects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal saat POS Tagging: {e}")


# --- TAHAP 3: EKSTRAKSI BERBASIS ATURAN & PERSIAPAN LABELING ---
@router.post("/{process_id}/extract")
async def run_extraction_endpoint(process_id: str, payload: AspectSelection):
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    extracted_file_path = os.path.join("data", f"extracted_{process_id}.csv")
    
    if not os.path.exists(cleaned_file_path):
        raise HTTPException(status_code=404, detail="File cleaned tidak ditemukan.")

    extraction.run_extraction(
        input_path=cleaned_file_path,
        output_path=extracted_file_path,
        selected_aspects=payload.aspects
    )

    if not os.path.exists(extracted_file_path):
        raise HTTPException(status_code=404, detail="File hasil ekstraksi tidak ditemukan.")

    df = pd.read_csv(extracted_file_path)
    if df.empty:
        raise HTTPException(status_code=404, detail="Tidak ada aspek yang ditemukan di dalam data. Coba pilih aspek yang berbeda.")

    sample_size = min(500, len(df))
    labeling_sample = df.sample(n=sample_size, random_state=42)
    labeling_sample['id'] = range(len(labeling_sample))
    
    return {"labeling_data": labeling_sample.to_dict(orient='records')}


# --- TAHAP 4 & 5: MENERIMA LABEL, MELATIH MODEL, EVALUASI & PREDIKSI ---
@router.post("/{process_id}/train")
async def train_model_endpoint(process_id: str, payload: LabelingPayload, background_tasks: BackgroundTasks):
    labeled_data_path = os.path.join("data", f"labeled_{process_id}.csv")
    
    labels_df = pd.DataFrame([item.model_dump() for item in payload.labels])
    labels_df.to_csv(labeled_data_path, index=False)

    background_tasks.add_task(
        training.run_training_pipeline,
        process_id=process_id,
        labeled_data_path=labeled_data_path
    )
    return {"message": "Proses training, evaluasi, dan prediksi telah dimulai."}


@router.get("/{process_id}/results")
async def get_final_results(process_id: str):
    eval_path = os.path.join("models_trained", f"evaluation_{process_id}.json")
    prediction_path = os.path.join("data", f"final_predictions_{process_id}.csv")

    if not os.path.exists(eval_path) or not os.path.exists(prediction_path):
        raise HTTPException(status_code=404, detail="Hasil belum siap. Proses training mungkin masih berjalan.")

    with open(eval_path, 'r') as f:
        evaluation_results = json.load(f)

    df_pred = pd.read_csv(prediction_path)
    display_cols = ['product_name', 'cleaned_review', 'aspect', 'predicted_sentiment']
    df_display = df_pred[[col for col in display_cols if col in df_pred.columns]]
    df_display_preview = df_display.head(100).fillna('')
    
    prediction_preview = df_display_preview.to_dict(orient='records')
    
    final_results = {
        "columns": list(df_display.columns),
        "rows": prediction_preview
    }
    return {"evaluation": evaluation_results, "predictions": final_results}


# --- ENDPOINT UNTUK UNDUH FILE ---
@router.get("/{process_id}/download/{stage}")
async def download_file(process_id: str, stage: str):
    file_map = {
        "preprocessed": os.path.join("data", f"cleaned_{process_id}.csv"),
        "final_results": os.path.join("data", f"final_predictions_{process_id}.csv")
    }
    file_path = file_map.get(stage)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    return FileResponse(path=file_path, filename=f"{stage}_{process_id}.csv", media_type='text/csv')