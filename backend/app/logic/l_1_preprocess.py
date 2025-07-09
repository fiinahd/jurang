import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
import os

def write_progress(process_id: str, message: str):
    """Helper function to write progress to a status file."""
    status_file = os.path.join("data", f"status_{process_id}.json")
    with open(status_file, "w") as f:
        json.dump({"status": message}, f)

def _preprocess_text_internal(text: str, stemmer) -> str:
    # ... (fungsi ini tidak berubah) ...
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"\btp\b", "tapi", text)
    text = re.sub(r"\bgk\b", "tidak", text)
    text = re.sub(r"\bga\b", "tidak", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z\s]", ' ', text)
    tokens = text.split()
    stopwords = {'yang','dan','di','ke','dari','pada','ini','itu','untuk','dengan','tidak','tapi','sangat','sekali','juga','adalah','atau','dalam','tp','gk','ga','aja','sih','nih','kok'}
    tokens = [t for t in tokens if t not in stopwords]
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)

def run_preprocessing(process_id: str, input_path: str, output_path: str, review_column: str, product_column: str):
    try:
        write_progress(process_id, "Membaca file Excel...")
        df = pd.read_excel(input_path)
        df = df.dropna(subset=[product_column, review_column])
        df = df[~df[product_column].astype(str).str.lower().isin(['tidak tersedia'])]

        total_rows = len(df)
        write_progress(process_id, f"0/{total_rows} baris diproses")

        stemmer = StemmerFactory().create_stemmer()
        
        cleaned_reviews = []
        for i, text in enumerate(df[review_column]):
            cleaned_reviews.append(_preprocess_text_internal(text, stemmer))
            if (i + 1) % 20 == 0 or (i + 1) == total_rows: # Update progress setiap 20 baris
                 write_progress(process_id, f"{i + 1}/{total_rows} baris diproses")

        df['cleaned_review'] = cleaned_reviews
        
        out_df = pd.DataFrame({
            'product_name': df[product_column],
            'cleaned_review': df['cleaned_review']
        })
        
        out_df = out_df[out_df['cleaned_review'].str.split().str.len() > 1]
        
        out_df.to_csv(output_path, index=False)
        print(f"Preprocessing selesai untuk {process_id}. Hasil disimpan di {output_path}")
    except Exception as e:
        print(f"ERROR DALAM PROSES PREPROCESSING ({process_id}): {e}")
        write_progress(process_id, f"Error: {e}")