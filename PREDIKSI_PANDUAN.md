# SI-PADI JATIM - Panduan Penggunaan Form Prediksi

## 📋 Overview

SI-PADI JATIM sekarang dilengkapi dengan **Form Prediksi Langsung** yang menggunakan model SVR-RBF yang telah dilatih dengan data 34 Kabupaten/Kota di Jawa Timur.

## 🎯 Cara Menggunakan Form Prediksi

### Tab 1: 🔮 Prediksi Cepat (Manual Input)

Form ini memungkinkan Anda melakukan prediksi produksi padi secara real-time dengan menginput:

#### **Data Dasar:**
- **Tahun** (2020-2030): Tahun prediksi
- **Bulan** (1-12): Bulan dalam tahun tersebut
- **Kabupaten/Kota**: Pilih salah satu dari 34 wilayah di Jawa Timur
- **Luas Panen (ha)**: Luas area pertanian dalam hektar

#### **Data Iklim:**
- **Curah Hujan (mm)**: Intensitas hujan per bulan (0-500 mm)
- **Kelembapan (%)**: Tingkat kelembapan udara (0-100%)
- **Suhu Rata-rata (°C)**: Suhu lingkungan (15-40°C)
- **Kecepatan Angin (m/s)**: Kecepatan angin rata-rata (0-20 m/s)

#### **Radiasi Matahari:**
- **Sinar Matahari (jam/hari)**: Durasi sinar matahari per hari (0-12 jam)

### 📊 Hasil Prediksi

Setelah klik **[🔮 Prediksi Produksi Padi]**, sistem akan menampilkan:

1. **Hasil Utama** (3 Metric Cards):
   - Produksi Prediksi (ton)
   - Produktivitas (ton/ha)
   - Luas Panen (ha)

2. **Detail Prediksi**:
   - Informasi lengkap input yang Anda masukkan
   - Validasi dari semua parameter

## 🔧 Model Details

### Model Architecture
- **Type**: Support Vector Regression (SVR)
- **Kernel**: Radial Basis Function (RBF)
- **Fitur Input**: 47 (kombinasi dari data dasar, iklim, lokasi encoding)
- **Trained Data**: 34 Kabupaten/Kota Jawa Timur (2018-2024)

### Fitur yang Digunakan
```
1. Luas Panen (numeric)
2. Curah Hujan (numeric)
3. Kelembapan (numeric)
4. Suhu (numeric)
5. Kecepatan Angin (numeric)
6. Sinar Matahari (numeric)
7. Tahun (numeric)
8-41. Kabupaten/Kota (one-hot encoded - 34 kolom)
42-43. Bulan (sin/cos encoded - 2 kolom)
```

## 📥 Tab 2: Batch Input Manual

Gunakan tab ini untuk memasukkan multiple data points dan test dengan data lokal Anda:

1. Isi form dengan data tahun, kabupaten, luas panen, produktivitas
2. Klik **[➕ Tambah Data]** untuk menambah baris
3. Setelah selesai, klik **[🚀 Gunakan untuk Prediksi]**

## 📤 Tab 3: Upload File CSV

Upload file CSV yang berisi data bulk:

### Format CSV yang Diterima:

**Format 1 - Data Lengkap (untuk prediksi dengan model)**:
```
Tahun,Bulan,Kabupaten/Kota,Luas_Panen,Curah_Hujan,Kelembapan,Suhu,Kecepatan_Angin,Sinar_Matahari
2024,6,Kabupaten Bangkalan,1000,150,75,27,3,7
```

**Format 2 - Data Sederhana (untuk batch prediksi)**:
```
Tahun,Kabupaten,Luas_Panen,Produktivitas
2024,Bangkalan,1000,50
```

## ✅ Validasi Input

Sistem otomatis akan memvalidasi input Anda:
- ✅ Tahun: 2020-2030
- ✅ Bulan: 1-12
- ✅ Luas Panen: > 0 ha
- ✅ Curah Hujan: 0-500 mm
- ✅ Kelembapan: 0-100%
- ✅ Suhu: 15-40°C
- ✅ Kecepatan Angin: 0-20 m/s
- ✅ Sinar Matahari: 0-12 jam/hari

## 📌 Tips Penggunaan

1. **Akurasi Prediksi**: Model ini dilatih dengan data real dari Jawa Timur selama 6 tahun. Akurasi terbaik saat input data sesuai dengan range historis.

2. **Data Iklim**: Pastikan data iklim yang Anda input realistis. Gunakan data dari Stasiun Meteorologi setempat untuk akurasi terbaik.

3. **Kabupaten/Kota**: Pilih lokasi yang sesuai dengan keadaan geografis wilayah Anda.

4. **Bulan Tanam**: Bulan yang dipilih akan mempengaruhi encoding trigonometri (sin/cos) yang digunakan model.

## 🔍 Interpretasi Hasil

```
Produksi Prediksi (ton) = Model Output
Produktivitas (ku/ha) = Produksi Prediksi / Luas Panen * 100
```

Contoh: Jika Luas Panen = 1000 ha dan Produksi Prediksi = 500 ton
→ Produktivitas = 50 ku/ha

## ⚠️ Keterbatasan

- Model ini prediksi untuk musim yang akan datang, bukan prediksi real-time
- Akurasi tergantung pada kualitas dan representatifitas data input
- Model dilatih dengan data historis 2018-2024, gunakan dengan hati-hati untuk prediksi jauh di masa depan
- Kondisi ekstrem (bencana alam, hama besar) tidak dapat diprediksi model

## 📞 Dukungan

Jika ada pertanyaan atau issue:
1. Periksa bahwa semua input dalam range yang valid
2. Verifikasi data iklim dari sumber yang terpercaya
3. Hubungi tim pengembang untuk masalah teknis

---

**Versi**: 1.0  
**Model**: model_svr_rbf_90_30_partikel_100_iterasi.save  
**Last Updated**: 2024
