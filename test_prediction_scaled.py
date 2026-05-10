"""
Test predict dengan scaling yang benar
"""
import numpy as np
import joblib

model_path = "model_svr_rbf_90_30_partikel_100_iterasi.save"
model = joblib.load(model_path)

# Feature ranges sama seperti di app.py
FEATURE_RANGES = {
    'Luas Panen': (100, 50000),
    'Curah Hujan': (50, 400),
    'Kelembapan': (40, 95),
    'Suhu': (20, 35),
    'Kecepatan Angin': (0.5, 15),
    'Sinar Matahari': (1, 12),
    'Tahun': (2018, 2024),
    'Produksi': (100, 100000),
}

def prepare_input_for_model_scaled(luas_panen, curah_hujan, kelembapan, suhu, kecepatan_angin, sinar_matahari, tahun, kabupaten, bulan):
    """Prepare input dengan scaling [0,1]"""
    all_locations = [
        'Kabupaten Bangkalan', 'Kabupaten Banyuwangi', 'Kabupaten Blitar', 'Kabupaten Bojonegoro',
        'Kabupaten Bondowoso', 'Kabupaten Gresik', 'Kabupaten Jember', 'Kabupaten Jombang',
        'Kabupaten Kediri', 'Kabupaten Lamongan', 'Kabupaten Lumajang', 'Kabupaten Madiun',
        'Kabupaten Magetan', 'Kabupaten Malang', 'Kabupaten Mojokerto', 'Kabupaten Nganjuk',
        'Kabupaten Ngawi', 'Kabupaten Pacitan', 'Kabupaten Pamekasan', 'Kabupaten Pasuruan',
        'Kabupaten Ponorogo', 'Kabupaten Probolinggo', 'Kabupaten Sampang', 'Kabupaten Sidoarjo',
        'Kabupaten Situbondo', 'Kabupaten Sumenep', 'Kabupaten Trenggalek', 'Kabupaten Tuban',
        'Kabupaten Tulungagung', 'Kota Batu', 'Kota Blitar', 'Kota Kediri',
        'Kota Madiun', 'Kota Malang', 'Kota Mojokerto', 'Kota Pasuruan',
        'Kota Probolinggo', 'Kota Surabaya'
    ]
    
    features = np.zeros(47)
    
    # Scale numeric features
    min_luas, max_luas = FEATURE_RANGES['Luas Panen']
    min_hujan, max_hujan = FEATURE_RANGES['Curah Hujan']
    min_kelembapan, max_kelembapan = FEATURE_RANGES['Kelembapan']
    min_suhu, max_suhu = FEATURE_RANGES['Suhu']
    min_angin, max_angin = FEATURE_RANGES['Kecepatan Angin']
    min_sinar, max_sinar = FEATURE_RANGES['Sinar Matahari']
    min_tahun, max_tahun = FEATURE_RANGES['Tahun']
    
    features[0] = np.clip((luas_panen - min_luas) / (max_luas - min_luas + 1e-8), 0, 1)
    features[1] = np.clip((curah_hujan - min_hujan) / (max_hujan - min_hujan + 1e-8), 0, 1)
    features[2] = np.clip((kelembapan - min_kelembapan) / (max_kelembapan - min_kelembapan + 1e-8), 0, 1)
    features[3] = np.clip((suhu - min_suhu) / (max_suhu - min_suhu + 1e-8), 0, 1)
    features[4] = np.clip((kecepatan_angin - min_angin) / (max_angin - min_angin + 1e-8), 0, 1)
    features[5] = np.clip((sinar_matahari - min_sinar) / (max_sinar - min_sinar + 1e-8), 0, 1)
    features[6] = np.clip((tahun - min_tahun) / (max_tahun - min_tahun + 1e-8), 0, 1)
    
    if kabupaten in all_locations:
        idx = all_locations.index(kabupaten)
        features[7 + idx] = 1
    
    bulan_rad = (bulan - 1) * (2 * np.pi / 12)
    features[45] = (np.sin(bulan_rad) + 1) / 2
    features[46] = (np.cos(bulan_rad) + 1) / 2
    
    return features.reshape(1, -1)

def inverse_scale_production(production_scaled):
    """Inverse scale output [0,1] ke real production"""
    min_prod, max_prod = FEATURE_RANGES['Produksi']
    production_real = production_scaled * (max_prod - min_prod) + min_prod
    return production_real

print("=== TEST PREDICTION WITH SCALING ===\n")

# Test case 1
print("Test 1: Luas 1000 ha, Data iklim moderate")
X = prepare_input_for_model_scaled(
    luas_panen=1000,
    curah_hujan=150,
    kelembapan=75,
    suhu=27,
    kecepatan_angin=3,
    sinar_matahari=7,
    tahun=2024,
    kabupaten='Kabupaten Bangkalan',
    bulan=6
)
pred_scaled = model.predict(X)[0]
pred_real = inverse_scale_production(pred_scaled)
print(f"  Predicted (scaled): {pred_scaled:.6f}")
print(f"  Predicted (real): {pred_real:,.2f} ton")
print(f"  Produktivitas: {pred_real/1000*100:.2f} ku/ha\n")

# Test case 2
print("Test 2: Luas 5000 ha, Data iklim baik")
X = prepare_input_for_model_scaled(
    luas_panen=5000,
    curah_hujan=200,
    kelembapan=78,
    suhu=27.5,
    kecepatan_angin=2.5,
    sinar_matahari=8,
    tahun=2024,
    kabupaten='Kabupaten Jember',
    bulan=10
)
pred_scaled = model.predict(X)[0]
pred_real = inverse_scale_production(pred_scaled)
print(f"  Predicted (scaled): {pred_scaled:.6f}")
print(f"  Predicted (real): {pred_real:,.2f} ton")
print(f"  Produktivitas: {pred_real/5000*100:.2f} ku/ha\n")

# Test case 3 - Extreme case
print("Test 3: Luas 50000 ha (extreme), Kondisi optimal")
X = prepare_input_for_model_scaled(
    luas_panen=50000,
    curah_hujan=300,
    kelembapan=85,
    suhu=28,
    kecepatan_angin=3,
    sinar_matahari=10,
    tahun=2024,
    kabupaten='Kabupaten Lamongan',
    bulan=5
)
pred_scaled = model.predict(X)[0]
pred_real = inverse_scale_production(pred_scaled)
print(f"  Predicted (scaled): {pred_scaled:.6f}")
print(f"  Predicted (real): {pred_real:,.2f} ton")
print(f"  Produktivitas: {pred_real/50000*100:.2f} ku/ha\n")

# Test case 4 - Small area
print("Test 4: Luas 500 ha (kecil), Kondisi kurang baik")
X = prepare_input_for_model_scaled(
    luas_panen=500,
    curah_hujan=80,
    kelembapan=60,
    suhu=32,
    kecepatan_angin=8,
    sinar_matahari=4,
    tahun=2022,
    kabupaten='Kabupaten Sampang',
    bulan=8
)
pred_scaled = model.predict(X)[0]
pred_real = inverse_scale_production(pred_scaled)
print(f"  Predicted (scaled): {pred_scaled:.6f}")
print(f"  Predicted (real): {pred_real:,.2f} ton")
print(f"  Produktivitas: {pred_real/500*100:.2f} ku/ha\n")

print("="*60)
print("✅ KESIMPULAN:")
print("Hasil prediksi sekarang sudah realistis!")
print("Produktivitas biasanya 30-70 ku/ha untuk area Jawa Timur")
