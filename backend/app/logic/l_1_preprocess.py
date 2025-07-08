import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def _preprocess_text_internal(text: str, stemmer) -> str:
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

def run_preprocessing(input_path: str, output_path: str, review_column: str, product_column: str):
    try:
        df = pd.read_excel(input_path)
        df = df.dropna(subset=[product_column, review_column])
        df = df[~df[product_column].astype(str).str.lower().isin(['tidak tersedia'])]

        stemmer = StemmerFactory().create_stemmer()
        df['cleaned_review'] = df[review_column].apply(lambda text: _preprocess_text_internal(text, stemmer))
        
        # PERBAIKAN KUNCI: Menggunakan variabel `product_column` bukan string "product_name"
        # dan menamainya kembali menjadi 'product_name' untuk konsistensi di tahap selanjutnya.
        out_df = pd.DataFrame({
            'product_name': df[product_column],
            'cleaned_review': df['cleaned_review']
        })
        
        out_df = out_df[out_df['cleaned_review'].str.split().str.len() > 1]
        
        out_df.to_csv(output_path, index=False)
        print(f"Preprocessing selesai. Hasil disimpan di {output_path}")
    except Exception as e:
        # Menampilkan error di console server untuk debugging
        print(f"ERROR DALAM PROSES PREPROCESSING: {e}")