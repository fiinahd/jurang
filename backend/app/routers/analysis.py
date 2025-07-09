import os
import uuid
import pandas as pd
from fastapi import (
    APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
)
from fastapi.responses import FileResponse, StreamingResponse
import json
from collections import Counter
import io

from ..logic import (
    l_1_preprocess as preprocess,
    l_2_postag as postag,
    l_3_extraction as extraction,
    l_4_training as training
)
from ..models import AspectSelection, LabelingPayload, VisualizationData

router = APIRouter(
    prefix="/api/process",
    tags=["Analysis Pipeline"]
)

def cleanup_files(process_id: str):
    """Menghapus file status sementara setelah proses selesai."""
    print(f"[{process_id}] Cleaning up temporary files...")
    status_file = os.path.join("data", f"status_{process_id}.json")
    if os.path.exists(status_file): os.remove(status_file)
    aspect_file = os.path.join("data", f"aspects_{process_id}.json")
    if os.path.exists(aspect_file): os.remove(aspect_file)

def _generate_visualization_data(df: pd.DataFrame) -> dict:
    """Memproses DataFrame hasil prediksi untuk menghasilkan data visualisasi."""
    if df.empty or 'aspect' not in df.columns:
        return {"net_sentiment_scores": [], "aspect_details": {}, "all_aspects": []}
    
    print("Generating visualization data...")
    sentiment_scores, all_aspects = [], sorted(df['aspect'].unique())

    for aspect in all_aspects:
        aspect_df = df[df['aspect'] == aspect]
        counts, total = aspect_df['predicted_sentiment'].value_counts(), len(aspect_df)
        pos, neg, netral = counts.get('positif', 0), counts.get('negatif', 0), counts.get('netral', 0)
        score = (pos - neg) / total if total > 0 else 0
        sentiment_scores.append({"aspect": aspect, "score": score, "positif": int(pos), "negatif": int(neg), "netral": int(netral)})
    
    sentiment_scores = sorted(sentiment_scores, key=lambda x: x['score'], reverse=True)
    aspect_details = {}
    for aspect in all_aspects:
        aspect_df = df[df['aspect'] == aspect]
        dist_counts = aspect_df['predicted_sentiment'].value_counts()
        distribution = {"positif": int(dist_counts.get('positif', 0)), "negatif": int(dist_counts.get('negatif', 0)), "netral": int(dist_counts.get('netral', 0))}
        word_clouds = {"positif": [], "negatif": [], "netral": []}
        for sentiment in ['positif', 'negatif', 'netral']:
            sentiment_df = aspect_df[aspect_df['predicted_sentiment'] == sentiment]
            if not sentiment_df.empty:
                text = ' '.join(sentiment_df['cleaned_review'].astype(str))
                word_counts = Counter(text.split())
                word_clouds[sentiment] = word_counts.most_common(20)
        aspect_details[aspect] = {"sentiment_distribution": distribution, "word_clouds": word_clouds}
    
    print("Visualization data generated successfully.")
    return {"net_sentiment_scores": sentiment_scores, "aspect_details": aspect_details, "all_aspects": all_aspects}

@router.post("/start")
async def start_process(background_tasks: BackgroundTasks, review_column: str = Form(...), file: UploadFile = File(...)):
    process_id = str(uuid.uuid4())
    print(f"[{process_id}] New process started.")
    raw_file_path = os.path.join("data", f"raw_{process_id}.xlsx")
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    try:
        with open(raw_file_path, "wb") as buffer: buffer.write(await file.read())
        print(f"[{process_id}] Raw file saved to {raw_file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {e}")
    background_tasks.add_task(preprocess.run_preprocessing, process_id=process_id, input_path=raw_file_path, output_path=cleaned_file_path, review_column=review_column)
    return {"process_id": process_id, "message": "Preprocessing started."}

@router.get("/{process_id}/progress")
async def get_progress(process_id: str):
    status_file = os.path.join("data", f"status_{process_id}.json")
    if not os.path.exists(status_file): return {"status": "Memulai..."}
    try:
        with open(status_file, "r") as f: data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {"status": "Memulai..."}

@router.get("/{process_id}/preprocess_result")
async def get_preprocess_result(process_id: str):
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    if not os.path.exists(cleaned_file_path):
        raise HTTPException(status_code=404, detail="File hasil preprocessing belum siap atau tidak ditemukan.")
    cleanup_files(process_id)
    df = pd.read_csv(cleaned_file_path)
    df_preview = df.head(10).fillna('')
    preview = df_preview.to_dict(orient='records')
    columns = list(df.columns)
    print(f"[{process_id}] Preprocessing result fetched.")
    return {"preview": {"columns": columns, "rows": preview}}

@router.post("/{process_id}/postag")
async def run_pos_tagging_endpoint(process_id: str, background_tasks: BackgroundTasks):
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    if not os.path.exists(cleaned_file_path):
        raise HTTPException(status_code=404, detail="File cleaned tidak ditemukan.")
    print(f"[{process_id}] Starting POS Tagging task.")
    background_tasks.add_task(postag.run_postagging, process_id=process_id, input_csv=cleaned_file_path, top_n=30)
    return {"message": "POS Tagging started."}

@router.get("/{process_id}/postag_result")
async def get_postag_result(process_id: str):
    result_file = os.path.join("data", f"aspects_{process_id}.json")
    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="Hasil POS Tagging belum siap.")
    with open(result_file, 'r') as f: data = json.load(f)
    cleanup_files(process_id)
    print(f"[{process_id}] POS Tagging result fetched.")
    return data

@router.post("/{process_id}/extract")
async def run_extraction_endpoint(process_id: str, payload: AspectSelection):
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    extracted_file_path = os.path.join("data", f"extracted_{process_id}.csv")
    if not os.path.exists(cleaned_file_path):
        raise HTTPException(status_code=404, detail="File cleaned tidak ditemukan.")
    extraction.run_extraction(input_path=cleaned_file_path, output_path=extracted_file_path, selected_aspects=payload.aspects)
    if not os.path.exists(extracted_file_path):
        raise HTTPException(status_code=404, detail="File hasil ekstraksi tidak ditemukan.")
    df = pd.read_csv(extracted_file_path)
    if df.empty:
        raise HTTPException(status_code=404, detail="Tidak ada aspek yang ditemukan di dalam data. Coba pilih aspek yang berbeda.")
    total_rows, percentage = len(df), payload.sampling_percentage / 100.0
    sample_size_from_percentage = int(total_rows * percentage)
    sample_size = min(sample_size_from_percentage, 500) if total_rows > 2000 else sample_size_from_percentage
    if total_rows > 0 and sample_size == 0: sample_size = 1
    sample_size = min(sample_size, total_rows)
    print(f"[{process_id}] Total rows with aspects: {total_rows}")
    print(f"[{process_id}] User choice: {payload.sampling_percentage}% ({sample_size_from_percentage} rows)")
    print(f"[{process_id}] Final sample size after smart rule: {sample_size} rows")
    labeling_sample = df.sample(n=sample_size, random_state=42)
    labeling_sample['id'] = range(len(labeling_sample))
    return {"labeling_data": labeling_sample.to_dict(orient='records')}

@router.post("/{process_id}/train")
async def train_model_endpoint(process_id: str, payload: LabelingPayload, background_tasks: BackgroundTasks):
    labeled_data_path = os.path.join("data", f"labeled_{process_id}.csv")
    labels_df = pd.DataFrame([item.model_dump() for item in payload.labels])
    labels_df.to_csv(labeled_data_path, index=False)
    print(f"[{process_id}] Labeled data saved. Starting training pipeline.")
    background_tasks.add_task(training.run_training_pipeline, process_id=process_id, labeled_data_path=labeled_data_path)
    return {"message": "Proses training, evaluasi, dan prediksi telah dimulai."}

@router.get("/{process_id}/results")
async def get_final_results(process_id: str):
    eval_path = os.path.join("models_trained", f"evaluation_{process_id}.json")
    prediction_path = os.path.join("data", f"final_predictions_{process_id}.csv")
    if not os.path.exists(eval_path) or not os.path.exists(prediction_path):
        raise HTTPException(status_code=404, detail="Hasil belum siap. Proses training mungkin masih berjalan.")
    cleanup_files(process_id)
    with open(eval_path, 'r') as f: evaluation_results = json.load(f)
    df_pred = pd.read_csv(prediction_path)
    visualization_data = _generate_visualization_data(df_pred)
    display_cols = ['cleaned_review', 'aspect', 'predicted_sentiment']
    df_display = df_pred[[col for col in display_cols if col in df_pred.columns]]
    df_display_preview = df_display.head(100).fillna('')
    prediction_preview = {"columns": list(df_display.columns), "rows": df_display_preview.to_dict(orient='records')}
    print(f"[{process_id}] Final results fetched.")
    return {"evaluation": evaluation_results, "predictions": prediction_preview, "visualization": visualization_data}
    
@router.get("/{process_id}/download/visualization")
async def download_visualization_data(process_id: str):
    prediction_path = os.path.join("data", f"final_predictions_{process_id}.csv")
    if not os.path.exists(prediction_path):
        raise HTTPException(status_code=404, detail="File prediksi tidak ditemukan untuk membuat laporan.")
    print(f"[{process_id}] Generating visualization report for download...")
    df_pred = pd.read_csv(prediction_path)
    viz_data = _generate_visualization_data(df_pred)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_scores = pd.DataFrame(viz_data['net_sentiment_scores'])
        df_scores.rename(columns={'aspect': 'Aspek', 'score': 'Skor Sentimen Bersih', 'positif': 'Jumlah Positif', 'negatif': 'Jumlah Negatif', 'netral': 'Jumlah Netral'}, inplace=True)
        df_scores.to_excel(writer, sheet_name='Ringkasan Peringkat Aspek', index=False)
        dist_global = df_pred['predicted_sentiment'].value_counts().reset_index()
        dist_global.columns = ['Sentimen', 'Jumlah']
        dist_global.to_excel(writer, sheet_name='Distribusi Sentimen Global', index=False)
        for aspect, details in viz_data['aspect_details'].items():
            sheet_name = f"Kata Kunci - {aspect[:20]}"
            df_pos = pd.DataFrame(details['word_clouds']['positif'], columns=['Kata Positif', 'Frekuensi'])
            df_neg = pd.DataFrame(details['word_clouds']['negatif'], columns=['Kata Negatif', 'Frekuensi'])
            df_net = pd.DataFrame(details['word_clouds']['netral'], columns=['Kata Netral', 'Frekuensi'])
            df_pos.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=0, index=False)
            df_neg.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=3, index=False)
            df_net.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=6, index=False)
    output.seek(0)
    headers = {'Content-Disposition': f'attachment; filename="laporan_visualisasi_{process_id}.xlsx"'}
    print(f"[{process_id}] Sending visualization report.")
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@router.get("/{process_id}/download/{stage}")
async def download_file(process_id: str, stage: str):
    file_map = {"preprocessed": os.path.join("data", f"cleaned_{process_id}.csv"), "final_results": os.path.join("data", f"final_predictions_{process_id}.csv")}
    file_path = file_map.get(stage)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    return FileResponse(path=file_path, filename=f"{stage}_{process_id}.csv", media_type='text/csv')