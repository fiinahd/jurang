import pandas as pd
import stanza
from collections import Counter
from typing import List
import json
import os

def write_progress(process_id: str, message: str):
    status_file = os.path.join("data", f"status_{process_id}.json")
    with open(status_file, "w") as f:
        json.dump({"status": message}, f)

def run_postagging(process_id: str, input_csv: str, top_n: int = 30):
    try:
        write_progress(process_id, "Memuat data bersih...")
        df = pd.read_csv(input_csv)
        df = df.dropna(subset=['cleaned_review'])
        texts = df['cleaned_review'].astype(str).tolist()
        total_docs = len(texts)

        write_progress(process_id, "Inisialisasi model Stanza...")
        nlp = stanza.Pipeline(lang='id', processors='tokenize,pos,lemma', use_gpu=False, verbose=False)

        counter = Counter()
        for i, doc_text in enumerate(texts):
            doc = nlp(doc_text)
            for sent in doc.sentences:
                for word in sent.words:
                    lemma = word.lemma
                    if word.upos == 'NOUN' and lemma is not None and len(lemma) > 2:
                        counter[lemma] += 1
            if (i + 1) % 20 == 0 or (i + 1) == total_docs:
                write_progress(process_id, f"Mengekstrak aspek: {i + 1}/{total_docs} ulasan")

        top_nouns = [term for term, _ in counter.most_common(top_n)]
        
        result_file = os.path.join("data", f"aspects_{process_id}.json")
        with open(result_file, 'w') as f:
            json.dump({"aspects": top_nouns}, f)

        print(f"POS Tagging selesai untuk {process_id}.")
    except Exception as e:
        print(f"ERROR DALAM PROSES POS TAGGING ({process_id}): {e}")
        write_progress(process_id, f"Error: {e}")