# Prediksi Status Kredit 

Aplikasi prediksi status kredit berbasis **Machine Learning** untuk mengklasifikasikan
kondisi kredit nasabah ke dalam tiga kategori:
**Lancar**, **Kurang Lancar**, dan **Macet** berdasarkan data keuangan.

Project ini dibuat sebagai **tugas UAS Mata Kuliah Machine Learning**
Program Studi Teknik Informatika.

---

## Deskripsi Singkat

Dalam proses pengajuan kredit, lembaga keuangan perlu menilai kemampuan nasabah
dalam membayar kewajiban kreditnya. Penilaian manual berpotensi subjektif dan
memerlukan waktu.

Aplikasi ini memanfaatkan **Decision Tree Classifier** untuk membantu memprediksi
status kredit nasabah secara **otomatis, objektif, dan berbasis data**.

---

## Tujuan
- Membantu proses evaluasi kelayakan kredit
- Mengurangi risiko kredit macet
- Menyediakan simulasi kondisi keuangan nasabah
- Mengimplementasikan model machine learning ke dalam aplikasi nyata

---

## Jenis Machine Learning
- **Supervised Learning**
- **Multiclass Classification**

Output model berupa kategori:
- Lancar
- Kurang Lancar
- Macet

---

## Dataset

Dataset yang digunakan merupakan **dataset buatan (sintetis)** dengan logika
keuangan yang realistis.

Jumlah data: **±250 data**

### Fitur yang Digunakan:
- gaji_bulanan
- pengeluaran_bulanan
- jumlah_tanggungan
- jumlah_pinjaman
- tenor_bulan
- bunga_tahunan
- rasio_pengeluaran
- rasio_pinjaman
- rasio_cicilan
- rasio_tanggungan

Label target:
- status_kredit

---

## Algoritma yang Digunakan

**Decision Tree Classifier**

Alasan pemilihan:
- Mudah diinterpretasikan
- Cocok untuk dataset kecil–menengah
- Kompleksitas rendah
- Sesuai untuk kasus penilaian kredit

---

## Evaluasi Model

- Data dibagi dengan rasio **80% training** dan **20% testing**
- Dilakukan **8x random split** untuk menguji kestabilan model
- Evaluasi menggunakan:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- Disertai visualisasi grafik akurasi

---

## Implementasi Aplikasi

Model yang telah dilatih diintegrasikan ke dalam aplikasi web menggunakan
**Streamlit**.

### Fitur Aplikasi:
- Input data keuangan dengan format mata uang
- Prediksi status kredit secara real-time
- Informasi cicilan dan rasio keuangan
- Peringatan risiko berdasarkan hasil prediksi

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone repository
```bash
git clone https://github.com/Arijonz/prediksi_kredit.git
cd prediksi_kredit
