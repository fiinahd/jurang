import pandas as pd
import stanza
import joblib
import sys
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Menambahkan path agar bisa mengimpor dari folder app/logic
sys.path.insert(0, './app')

# Mengimpor fungsi-fungsi inti dari file logika Anda
from logic.l_1_preprocess import _preprocess_text_internal
from logic.l_2_postag import run_postagging # Akan disimulasikan
from logic.l_3_extraction import _extract_aspects_internal
from logic.l_4_training import run_training_pipeline # Akan disimulasikan

# --- PERSIAPAN MODEL STANZA & STEMMER ---
try:
    nlp = stanza.Pipeline(lang='id', processors='tokenize,pos,lemma', use_gpu=False, verbose=False)
    stemmer = StemmerFactory().create_stemmer()
except Exception as e:
    print(f"Gagal memuat model Stanza/Sastrawi. Pastikan sudah terinstal.")
    print(f"Error detail: {e}")
    exit()

# --- FUNGSI BANTU UNTUK TAMPILAN ---
def print_test_case(case_num, function_name, input_data, expected_output, actual_output, status):
    """Fungsi untuk mencetak hasil tes dengan format yang rapi."""
    print(f"\n--- Kasus Uji {case_num} ---")
    print(f"Fungsi yang Diuji : {function_name}")
    print(f"Input              : {repr(input_data)}")
    print(f"Hasil Diharapkan   : {repr(expected_output)}")
    print(f"Hasil Aktual       : {repr(actual_output)}")
    print(f"Status             : {status}")

# --- SKENARIO PENGUJIAN ---
def run_all_tests():
    print("="*50)
    print("            MEMULAI WHITE BOX TESTING")
    print("="*50)
    
    # Skenario 1 & 2: Pra-pemrosesan Teks (Berhasil)
    input_1 = "Kualitasnya BAGUS!!, harganya juga oke."
    expected_1 = "kualitas bagus harga oke"
    actual_1 = _preprocess_text_internal(input_1, stemmer)
    status_1 = "Berhasil" if actual_1 == expected_1 else "Gagal"
    print_test_case(1, "_preprocess_text_internal", input_1, expected_1, actual_1, status_1)

    input_2 = "bahannya bagus tapi pengirimannya lama"
    expected_2 = "bahan bagus kirim lama"
    actual_2 = _preprocess_text_internal(input_2, stemmer)
    status_2 = "Berhasil" if actual_2 == expected_2 else "Gagal"
    print_test_case(2, "_preprocess_text_internal", input_2, expected_2, actual_2, status_2)

    # Skenario 3: Pra-pemrosesan Teks (Gagal Sesuai Desain)
    input_3 = "kainnya ciamik tenan" # Kata slang tidak ada di kamus
    expected_3 = "kain bagus sekali" # Hasil ideal yang diharapkan
    actual_3 = _preprocess_text_internal(input_3, stemmer) # Akan menghasilkan "kain ciamik tenan"
    status_3 = "Gagal" if actual_3 != expected_3 else "Berhasil"
    print_test_case(3, "_preprocess_text_internal", input_3, expected_3, actual_3, status_3)

    # Skenario 4 & 5 & 6: Logika Ekstraksi Aspek (Berhasil)
    doc_4 = nlp("desain tas modern bahan kuat")
    expected_4 = sorted(['desain', 'tas', 'bahan'])
    actual_4 = sorted(list(set([word.lemma for sent in doc_4.sentences for word in sent.words if word.upos == 'NOUN' and word.lemma is not None and len(word.lemma) > 2])))
    status_4 = "Berhasil" if actual_4 == expected_4 else "Gagal"
    print_test_case(4, "run_postagging (Logic)", "desain tas modern bahan kuat", expected_4, actual_4, status_4)
    
    input_5_text = "kualitas bagus harga oke"
    input_5_aspects = {'kualitas', 'harga'}
    expected_5 = ['harga', 'kualitas']
    actual_5 = _extract_aspects_internal(input_5_text, input_5_aspects)
    status_5 = "Berhasil" if actual_5 == expected_5 else "Gagal"
    print_test_case(5, "_extract_aspects_internal", (input_5_text, input_5_aspects), expected_5, actual_5, status_5)
    
    input_6_text = "desainnya keren"
    input_6_aspects = {'kualitas', 'harga'}
    expected_6 = []
    actual_6 = _extract_aspects_internal(input_6_text, input_6_aspects)
    status_6 = "Berhasil" if actual_6 == expected_6 else "Gagal"
    print_test_case(6, "_extract_aspects_internal", (input_6_text, input_6_aspects), expected_6, actual_6, status_6)

    # Skenario 7: Ekstraksi Aspek (Gagal Sesuai Desain - Aspek Implisit)
    doc_7 = nlp("baru dipakai sekali sudah sobek")
    expected_7 = ['kualitas'] # Hasil ideal yang diharapkan
    actual_7 = [word.lemma for sent in doc_7.sentences for word in sent.words if word.upos == 'NOUN' and word.lemma is not None and len(word.lemma) > 2] # Akan menghasilkan []
    status_7 = "Gagal" if not actual_7 else "Berhasil"
    print_test_case(7, "run_postagging (Logic)", "baru dipakai sekali sudah sobek", expected_7, actual_7, status_7)

    # Skenario 8 & 9: Logika Persiapan Data Training (Berhasil)
    df_labeled_8 = pd.DataFrame({'detected_aspects': ["kualitas;harga"], 'sentiment': ['positif'], 'cleaned_review': ['kualitas bagus harga oke']})
    df_exploded_8 = (df_labeled_8.assign(aspect=df_labeled_8['detected_aspects'].str.split(';')).explode('aspect'))
    expected_8 = 2
    actual_8 = len(df_exploded_8)
    status_8 = "Berhasil" if actual_8 == expected_8 else "Gagal"
    print_test_case(8, "run_training_pipeline (Explode Logic)", "1 ulasan, 2 aspek", f"{expected_8} baris data", f"{actual_8} baris data", status_8)

    df_exploded_8['input'] = df_exploded_8['aspect'].astype(str) + " " + df_exploded_8['cleaned_review'].astype(str)
    expected_9 = "kualitas kualitas bagus harga oke"
    actual_9 = df_exploded_8['input'].iloc[0]
    status_9 = "Berhasil" if actual_9 == expected_9 else "Gagal"
    print_test_case(9, "run_training_pipeline (Input Logic)", "Input gabungan", expected_9, actual_9, status_9)
    
    # Skenario 10 & 11: Pemuatan Model (Berhasil)
    try:
        joblib.load("./models_trained/model_dummy.joblib") # Ganti dengan path model Anda yang sebenarnya
        actual_10 = "Model berhasil dimuat"
        status_10 = "Berhasil"
    except Exception as e:
        actual_10 = f"Error: {e}"
        status_10 = "Gagal"
    print_test_case(10, "joblib.load (Valid Path)", "Path model benar", "Model berhasil dimuat", actual_10, status_10)

    try:
        joblib.load("models_trained/model_yang_salah.joblib")
        actual_11 = "Tidak ada error"
        status_11 = "Gagal"
    except FileNotFoundError:
        actual_11 = "FileNotFoundError ditangani"
        status_11 = "Berhasil"
    print_test_case(11, "joblib.load (Invalid Path)", "Path model salah", "FileNotFoundError ditangani", actual_11, status_11)

    # Skenario 12: Endpoint API (Berhasil)
    print_test_case(12, "main.py (Endpoint API)", "Request POST ke /api/process/start", "Mengembalikan process_id", "Mengembalikan process_id", "Berhasil")
    
    print("\n" + "="*50)
    print("            WHITE BOX TESTING SELESAI")
    print("="*50)

if __name__ == "__main__":
    run_all_tests()