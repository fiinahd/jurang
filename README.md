# senna: Sistem Analisis Sentimen Berbasis Aspek 🤖

<p align="center">
  <img src="https://raw.githubusercontent.com/fiinahd/jurang/main/frontend/assets/dexter.svg" width="120" />
</p>

<p align="center">
  Sebuah aplikasi web interaktif untuk melakukan analisis sentimen berbasis aspek secara <em>end-to-end</em>.
  <br />
  Mulai dari data mentah hingga visualisasi hasil yang kaya wawasan.
</p>

---

**senna** adalah sebuah *tool* atau perkakas bantu yang dirancang untuk menyederhanakan dan mengotomatiskan seluruh alur kerja penelitian analisis sentimen. Pengguna dapat mengunggah dataset ulasan, dan sistem akan memandu melalui setiap tahapan, mulai dari pembersihan data, ekstraksi aspek, pelabelan, hingga pelatihan model dan penyajian hasil dalam bentuk tabel serta grafik interaktif.

## ✨ Fitur Utama

- **Pipeline Terpandu:** Antarmuka berbasis *wizard* yang memandu pengguna melalui 7 tahapan analisis.
- **Preprocessing Otomatis:** Membersihkan data teks secara otomatis (case folding, stemming, stopword removal).
- **Ekstraksi Aspek Cerdas:** Menggunakan *Part-of-Speech (POS) Tagging* untuk merekomendasikan kandidat aspek dari data.
- **Sampling Dinamis:** Opsi untuk memilih persentase data yang akan dilabeli, dengan pembatasan cerdas untuk dataset besar.
- **Antarmuka Pelabelan Interaktif:** Halaman khusus untuk melabeli data dengan mudah, dilengkapi *pagination*.
- **Dasbor Visualisasi Hasil:**
  - Grafik peringkat aspek berdasarkan skor sentimen bersih (*Net Sentiment Score*).
  - Dasbor interaktif untuk melihat distribusi sentimen per aspek.
  - *Word cloud* dinamis untuk menampilkan kata kunci positif dan negatif yang terkait dengan setiap aspek.
- **Ekspor Laporan:** Kemampuan untuk mengunduh hasil prediksi mentah dan laporan visualisasi dalam format Excel multi-sheet yang rapi.

## 🛠️ Teknologi yang Digunakan

| Komponen | Teknologi / Library |
| :--- | :--- |
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | Vue.js (CDN), Tailwind CSS (CDN), HTML5 |
| **Analisis Data & ML** | Pandas, Scikit-learn (KNN, TF-IDF) |
| **Pemrosesan Bahasa (NLP)**| Stanza, Sastrawi |
| **Visualisasi** | Chart.js, WordCloud2.js |
| **Ekspor Laporan** | XlsxWriter |

## 🚀 Cara Menjalankan

Berikut adalah langkah-langkah untuk menjalankan proyek ini di lingkungan lokal.

### Prasyarat
- Python 3.9+
- Git

### 1. Backend

```bash
# Clone repositori
git clone [https://github.com/fiinahd/jurang.git](https://github.com/fiinahd/jurang.git)
cd jurang/backend

# Buat dan aktifkan virtual environment
python -m venv venv
source venv/bin/activate  # Untuk macOS/Linux
# venv\Scripts\activate   # Untuk Windows

# Install semua library yang dibutuhkan
pip install -r requirements.txt

# Unduh model bahasa untuk Stanza
python -c "import stanza; stanza.download('id', verbose=False)"

# Jalankan server FastAPI
uvicorn app.main:app --reload
```
Server backend akan berjalan di `http://127.0.0.1:8000`. Biarkan terminal ini tetap terbuka.

### 2. Frontend

Buka terminal baru atau File Explorer, lalu:
1. Navigasi ke folder `frontend`.
2. Buka file **`index.html`** langsung di browser pilihan Anda (Chrome, Firefox, dll).

Aplikasi web sekarang siap digunakan!
