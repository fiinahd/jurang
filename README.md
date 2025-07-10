Senna - Sistem Analisis Sentimen Berbasis Aspek
Senna adalah sebuah aplikasi web interaktif yang dirancang untuk melakukan Analisis Sentimen Berbasis Aspek (Aspect-Based Sentiment Analysis - ABSA). Sistem ini memungkinkan pengguna untuk mengunggah data ulasan produk, mengidentifikasi aspek-aspek penting secara otomatis, melatih model klasifikasi, dan memvisualisasikan hasilnya dalam dasbor yang informatif.

Tujuan utama dari sistem ini adalah untuk mengubah data ulasan mentah menjadi insight yang dapat ditindaklanjuti, membantu dalam pengambilan keputusan berbasis data dengan memahami sentimen pelanggan terhadap setiap fitur atau aspek spesifik dari suatu produk.

✨ Fitur Utama
Pipeline End-to-End: Alur kerja lengkap mulai dari unggah data hingga visualisasi hasil.

Unggah Data Fleksibel: Mendukung unggah data ulasan melalui file Excel (.xlsx).

Preprocessing Teks Otomatis: Membersihkan teks ulasan secara otomatis (case folding, stopword removal, stemming) menggunakan Sastrawi.

Ekstraksi Aspek Cerdas: Menggunakan Part-of-Speech (POS) Tagging dengan Stanza untuk mengidentifikasi kata benda yang paling sering muncul sebagai kandidat aspek.

Antarmuka Interaktif: Pengguna dapat memilih aspek yang relevan dan melakukan pelabelan data sampel melalui antarmuka web yang mudah digunakan.

Pelatihan Model: Melatih model klasifikasi K-Nearest Neighbors (KNN) dengan pembobotan TF-IDF untuk memprediksi sentimen.

Dasbor Visualisasi: Menampilkan hasil analisis dalam bentuk:

Diagram peringkat sentimen per aspek.

Distribusi sentimen (positif, negatif, netral) per aspek.

Word cloud untuk kata kunci positif dan negatif.

Ekspor Laporan: Mengunduh laporan hasil analisis lengkap dengan visualisasi dalam format Excel dan PDF.

🚀 Teknologi yang Digunakan
Arsitektur sistem ini terbagi menjadi dua bagian utama:

Backend: Dibangun menggunakan Python dengan framework FastAPI untuk menyajikan API yang cepat dan asinkron.

Frontend: Halaman web single-page application (SPA) yang dinamis, dibangun menggunakan Vue.js dan ditata dengan Tailwind CSS.

📚 Pustaka Utama
Berikut adalah daftar pustaka kunci yang menjadi fondasi dari sistem ini:

Backend (Python)
fastapi & uvicorn: Kerangka kerja API dan server.

pandas: Manipulasi dan pemrosesan data.

scikit-learn: Pelatihan model machine learning (TF-IDF, KNN).

Sastrawi: Proses stemming untuk Bahasa Indonesia.

stanza: Analisis linguistik (POS Tagging) untuk ekstraksi aspek.

matplotlib & wordcloud: Pembuatan grafik dan word cloud.

xlsxwriter & openpyxl: Membaca dan menulis file Excel.

fpdf2: Pembuatan laporan dalam format PDF.

Frontend (JavaScript)
Vue.js: Kerangka kerja JavaScript untuk membangun antarmuka pengguna.

Tailwind CSS: Utilitas CSS untuk desain yang cepat dan responsif.

Chart.js: Menampilkan diagram dan grafik interaktif.

wordcloud2.js: Membuat visualisasi word cloud di kanvas HTML.
