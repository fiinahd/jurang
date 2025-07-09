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

# Perbaikan untuk matplotlib agar tidak membuka GUI di server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from fpdf import FPDF

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

# --- KELAS BANTUAN UNTUK PDF DENGAN HEADER & FOOTER ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Laporan Analisis Sentimen', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
        
    def add_image_from_bytes(self, image_bytes, x=None, y=None, w=0, h=0):
        image_bytes.seek(0)
        self.image(image_bytes, x=x, y=y, w=w, h=h)

# --- FUNGSI INTERNAL ---
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

# --- ENDPOINT API ---
@router.post("/start")
async def start_process(background_tasks: BackgroundTasks, review_column: str = Form(...), file: UploadFile = File(...)):
    process_id = str(uuid.uuid4())
    raw_file_path = os.path.join("data", f"raw_{process_id}.xlsx")
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    try:
        with open(raw_file_path, "wb") as buffer: buffer.write(await file.read())
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
    if not os.path.exists(cleaned_file_path): raise HTTPException(status_code=404, detail="File hasil preprocessing belum siap atau tidak ditemukan.")
    cleanup_files(process_id)
    df = pd.read_csv(cleaned_file_path)
    preview = df.head(10).fillna('').to_dict(orient='records')
    return {"preview": {"columns": list(df.columns), "rows": preview}}

@router.post("/{process_id}/postag")
async def run_pos_tagging_endpoint(process_id: str, background_tasks: BackgroundTasks):
    cleaned_file_path = os.path.join("data", f"cleaned_{process_id}.csv")
    if not os.path.exists(cleaned_file_path): raise HTTPException(status_code=404, detail="File cleaned tidak ditemukan.")
    background_tasks.add_task(postag.run_postagging, process_id=process_id, input_csv=cleaned_file_path, top_n=30)
    return {"message": "POS Tagging started."}

@router.get("/{process_id}/postag_result")
async def get_postag_result(process_id: str):
    result_file = os.path.join("data", f"aspects_{process_id}.json")
    if not os.path.exists(result_file): raise HTTPException(status_code=404, detail="Hasil POS Tagging belum siap.")
    with open(result_file, 'r') as f: data = json.load(f)
    cleanup_files(process_id)
    return data

@router.post("/{process_id}/extract")
async def run_extraction_endpoint(process_id: str, payload: AspectSelection):
    cleaned_file_path, extracted_file_path = os.path.join("data", f"cleaned_{process_id}.csv"), os.path.join("data", f"extracted_{process_id}.csv")
    if not os.path.exists(cleaned_file_path): raise HTTPException(status_code=404, detail="File cleaned tidak ditemukan.")
    extraction.run_extraction(input_path=cleaned_file_path, output_path=extracted_file_path, selected_aspects=payload.aspects)
    if not os.path.exists(extracted_file_path): raise HTTPException(status_code=404, detail="File hasil ekstraksi tidak ditemukan.")
    df = pd.read_csv(extracted_file_path)
    if df.empty: raise HTTPException(status_code=404, detail="Tidak ada aspek yang ditemukan di dalam data.")
    total_rows, percentage = len(df), payload.sampling_percentage / 100.0
    sample_size = min(int(total_rows * percentage), 500) if total_rows > 2000 else int(total_rows * percentage)
    if total_rows > 0 and sample_size == 0: sample_size = 1
    labeling_sample = df.sample(n=min(sample_size, total_rows), random_state=42)
    labeling_sample['id'] = range(len(labeling_sample))
    return {"labeling_data": labeling_sample.to_dict(orient='records')}

@router.post("/{process_id}/train")
async def train_model_endpoint(process_id: str, payload: LabelingPayload, background_tasks: BackgroundTasks):
    labeled_data_path = os.path.join("data", f"labeled_{process_id}.csv")
    pd.DataFrame([item.model_dump() for item in payload.labels]).to_csv(labeled_data_path, index=False)
    background_tasks.add_task(training.run_training_pipeline, process_id=process_id, labeled_data_path=labeled_data_path)
    return {"message": "Proses training, evaluasi, dan prediksi telah dimulai."}

@router.get("/{process_id}/results")
async def get_final_results(process_id: str):
    eval_path, prediction_path = os.path.join("models_trained", f"evaluation_{process_id}.json"), os.path.join("data", f"final_predictions_{process_id}.csv")
    if not os.path.exists(eval_path) or not os.path.exists(prediction_path): raise HTTPException(status_code=404, detail="Hasil belum siap.")
    cleanup_files(process_id)
    with open(eval_path, 'r') as f: evaluation_results = json.load(f)
    df_pred = pd.read_csv(prediction_path)
    visualization_data, display_cols = _generate_visualization_data(df_pred), ['cleaned_review', 'aspect', 'predicted_sentiment']
    df_display = df_pred[[col for col in display_cols if col in df_pred.columns]].head(100).fillna('')
    prediction_preview = {"columns": list(df_display.columns), "rows": df_display.to_dict(orient='records')}
    return {"evaluation": evaluation_results, "predictions": prediction_preview, "visualization": visualization_data}
    
# --- VERSI 1: UNDUH EXCEL DENGAN CHART PER ASPEK ---
@router.get("/{process_id}/download/visualization")
async def download_visualization_data(process_id: str):
    prediction_path = os.path.join("data", f"final_predictions_{process_id}.csv")
    if not os.path.exists(prediction_path): raise HTTPException(status_code=404, detail="File prediksi tidak ditemukan.")
    
    df_pred = pd.read_csv(prediction_path)
    viz_data = _generate_visualization_data(df_pred)
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        df_scores = pd.DataFrame(viz_data['net_sentiment_scores'])
        dist_global = df_pred['predicted_sentiment'].value_counts()
        
        # Buat Gambar Global
        fig, ax = plt.subplots(figsize=(10, len(df_scores) * 0.5)); ax.barh(df_scores['aspect'], df_scores['score'], color=['#4ade80' if s>=0 else '#f87171' for s in df_scores['score']]); ax.set_title('Peringkat Aspek Berdasarkan Sentimen'); plt.tight_layout(); chart_rank_io = io.BytesIO(); plt.savefig(chart_rank_io, format='PNG'); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 6)); ax.pie(dist_global, labels=dist_global.index, autopct='%1.1f%%', colors=['#4ade80', '#f87171', '#facc15']); ax.set_title('Distribusi Sentimen Global'); chart_global_dist_io = io.BytesIO(); plt.savefig(chart_global_dist_io, format='PNG'); plt.close(fig)

        # Tulis Sheet Dashboard
        dashboard_sheet = workbook.add_worksheet('Dashboard'); dashboard_sheet.set_column('B:B', 60); dashboard_sheet.set_column('J:J', 60); dashboard_sheet.write('B1', 'Dashboard Laporan', workbook.add_format({'bold': True, 'font_size': 20})); dashboard_sheet.insert_image('B3', 'rank.png', {'image_data': chart_rank_io}); dashboard_sheet.insert_image('J3', 'dist.png', {'image_data': chart_global_dist_io})

        # Tulis Sheet Data Mentah
        df_scores.rename(columns={'aspect': 'Aspek', 'score': 'Skor Sentimen', 'positif': 'Jml Positif', 'negatif': 'Jml Negatif', 'netral': 'Jml Netral'}, inplace=True); df_scores.to_excel(writer, sheet_name='Data Peringkat Aspek', index=False)
        dist_global.reset_index().rename(columns={'index': 'Sentimen', 'predicted_sentiment': 'Jumlah'}).to_excel(writer, sheet_name='Data Distribusi Global', index=False)

        # Tulis Sheet per Aspek dengan Chart dan Word Cloud
        for aspect, details in viz_data['aspect_details'].items():
            sheet_name = f"Aspek - {aspect[:20]}"
            aspect_sheet = workbook.add_worksheet(sheet_name)
            
            # Buat & sisipkan chart distribusi per aspek
            dist_data = details['sentiment_distribution']
            sizes = list(dist_data.values())
            if sum(sizes) > 0:
                fig, ax = plt.subplots(figsize=(5, 4)); ax.pie(sizes, labels=list(dist_data.keys()), autopct='%1.1f%%', colors=['#4ade80', '#f87171', '#facc15']); ax.set_title(f'Distribusi: {aspect}'); plt.tight_layout(); chart_aspect_io = io.BytesIO(); plt.savefig(chart_aspect_io, format='PNG'); plt.close(fig)
                aspect_sheet.insert_image('J2', 'dist_aspect.png', {'image_data': chart_aspect_io})

            # Tulis data kata kunci
            for i, sentiment in enumerate(['positif', 'negatif', 'netral']):
                df_wc = pd.DataFrame(details['word_clouds'][sentiment], columns=[f'Kata {sentiment.capitalize()}', 'Frekuensi']); df_wc.to_excel(writer, sheet_name=sheet_name, startrow=0, startcol=i*3, index=False)

            # Buat & sisipkan word cloud
            pos_words, neg_words = dict(details['word_clouds']['positif']), dict(details['word_clouds']['negatif'])
            if pos_words:
                wc_pos_io = io.BytesIO(); WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(pos_words).to_image().save(wc_pos_io, format='PNG'); aspect_sheet.insert_image('J20', 'wc_pos.png', {'image_data': wc_pos_io})
            if neg_words:
                wc_neg_io = io.BytesIO(); WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(neg_words).to_image().save(wc_neg_io, format='PNG'); aspect_sheet.insert_image('J32', 'wc_neg.png', {'image_data': wc_neg_io})

    output.seek(0)
    return StreamingResponse(output, headers={'Content-Disposition': f'attachment; filename="laporan_visualisasi_{process_id}.xlsx"'}, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- VERSI 2: UNDUH PDF DENGAN CHART PER ASPEK ---
@router.get("/{process_id}/download/visualization-pdf")
async def download_visualization_pdf(process_id: str):
    prediction_path = os.path.join("data", f"final_predictions_{process_id}.csv")
    if not os.path.exists(prediction_path): raise HTTPException(status_code=404, detail="File prediksi tidak ditemukan.")
    
    df_pred = pd.read_csv(prediction_path)
    viz_data = _generate_visualization_data(df_pred)
    
    # Buat Gambar Global
    df_scores = pd.DataFrame(viz_data['net_sentiment_scores']); dist_global = df_pred['predicted_sentiment'].value_counts()
    fig, ax = plt.subplots(figsize=(8, len(df_scores) * 0.4)); ax.barh(df_scores['aspect'], df_scores['score'], color=['#4ade80' if s>=0 else '#f87171' for s in df_scores['score']]); ax.set_title('Peringkat Aspek'); plt.tight_layout(); chart_rank_io = io.BytesIO(); plt.savefig(chart_rank_io, format='PNG'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.pie(dist_global, labels=dist_global.index, autopct='%1.1f%%', colors=['#4ade80', '#f87171', '#facc15']); ax.set_title('Distribusi Global'); chart_global_dist_io = io.BytesIO(); plt.savefig(chart_global_dist_io, format='PNG'); plt.close(fig)

    # Buat PDF
    pdf = PDF(); pdf.add_page(); pdf.chapter_title('Dasbor Visualisasi Utama')
    pdf.add_image_from_bytes(chart_rank_io, x=10, y=35, w=190)
    pdf.ln(len(df_scores) * 12); pdf.add_image_from_bytes(chart_global_dist_io, x=pdf.get_x() + 50, w=100)

    for aspect, details in viz_data['aspect_details'].items():
        pdf.add_page(); pdf.chapter_title(f'Analisis Mendalam: Aspek "{aspect}"')
        
        # Buat & sisipkan chart distribusi per aspek
        dist_data = details['sentiment_distribution']
        sizes = list(dist_data.values())
        if sum(sizes) > 0:
            fig, ax = plt.subplots(figsize=(5,4)); ax.pie(sizes, labels=list(dist_data.keys()), autopct='%1.1f%%', colors=['#4ade80','#f87171','#facc15']); ax.set_title(f'Distribusi: {aspect}'); plt.tight_layout(); chart_aspect_io = io.BytesIO(); plt.savefig(chart_aspect_io, format='PNG'); plt.close(fig)
            pdf.add_image_from_bytes(chart_aspect_io, x=10, y=pdf.get_y(), w=90)

        # Buat & sisipkan word cloud
        pos_words, neg_words = dict(details['word_clouds']['positif']), dict(details['word_clouds']['negatif'])
        if pos_words:
            wc_pos_io = io.BytesIO(); WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(pos_words).to_image().save(wc_pos_io, format='PNG'); pdf.set_font('Arial', 'B', 12); pdf.set_xy(110, pdf.get_y()); pdf.cell(0, 10, 'Word Cloud Positif', 0, 1); pdf.add_image_from_bytes(wc_pos_io, x=110, y=pdf.get_y(), w=90)
        if neg_words:
            wc_neg_io = io.BytesIO(); WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(neg_words).to_image().save(wc_neg_io, format='PNG'); pdf.set_font('Arial', 'B', 12); pdf.set_xy(110, pdf.get_y()+50); pdf.cell(0, 10, 'Word Cloud Negatif', 0, 1); pdf.add_image_from_bytes(wc_neg_io, x=110, y=pdf.get_y(), w=90)
    pdf_output = io.BytesIO(pdf.output())
    return StreamingResponse(pdf_output, headers={'Content-Disposition': f'attachment; filename="laporan_pdf_{process_id}.pdf"'}, media_type='application/pdf')
    # pdf_output = io.BytesIO(pdf.output(dest='S').encode('latin-1'))
    # return StreamingResponse(pdf_output, headers={'Content-Disposition': f'attachment; filename="laporan_pdf_{process_id}.pdf"'}, media_type='application/pdf')

@router.get("/{process_id}/download/{stage}")
async def download_file(process_id: str, stage: str):
    file_map = {"preprocessed": os.path.join("data", f"cleaned_{process_id}.csv"), "final_results": os.path.join("data", f"final_predictions_{process_id}.csv")}
    file_path = file_map.get(stage)
    if not file_path or not os.path.exists(file_path): raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    return FileResponse(path=file_path, filename=f"{stage}_{process_id}.csv", media_type='text/csv')