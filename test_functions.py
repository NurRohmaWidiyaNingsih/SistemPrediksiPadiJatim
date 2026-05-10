"""
Test script untuk verify functions tanpa streamlit
"""
import numpy as np
import joblib
import os

print("=== Test Model Functions ===\n")

# Test 1: Load Model
print("Test 1: Load Pretrained Model")
model_path = "model_svr_rbf_90_30_partikel_100_iterasi.save"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   - Type: {type(model)}")
        print(f"   - Features: {model.n_features_in_}")
        print(f"   - Kernel: {model.kernel}")
        print(f"   - C: {model.C}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found: {model_path}")

# Test 2: Prepare Input Data
print("\nTest 2: Prepare Input Data for Model")

# Daftar semua kabupaten
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

def prepare_input_for_model(luas_panen, curah_hujan, kelembapan, suhu, kecepatan_angin, 
                            sinar_matahari, tahun, kabupaten, bulan):
    """Prepare input data for SVR model"""
    features = np.zeros(47)
    
    # Numerical features (0-6)
    features[0] = luas_panen
    features[1] = curah_hujan
    features[2] = kelembapan
    features[3] = suhu
    features[4] = kecepatan_angin
    features[5] = sinar_matahari
    features[6] = tahun
    
    # One-hot encoding for location (7-40)
    if kabupaten in all_locations:
        idx = all_locations.index(kabupaten)
        features[7 + idx] = 1
    
    # Bulan encoding (sin/cos) - features 45-46
    bulan_rad = (bulan - 1) * (2 * np.pi / 12)
    features[45] = np.sin(bulan_rad)  # bulan_sin
    features[46] = np.cos(bulan_rad)  # bulan_cos
    
    return features.reshape(1, -1)

try:
    X = prepare_input_for_model(
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
    print(f"✅ Input prepared successfully")
    print(f"   - Shape: {X.shape}")
    print(f"   - Sum of values: {X.sum():.2f}")
except Exception as e:
    print(f"❌ Error preparing input: {e}")

# Test 3: Make Prediction
print("\nTest 3: Make Prediction")
try:
    if 'model' in locals() and 'X' in locals():
        pred = model.predict(X)
        print(f"✅ Prediction successful")
        print(f"   - Predicted Production: {pred[0]:,.2f} ton")
        print(f"   - Input Features Used: 47")
except Exception as e:
    print(f"❌ Error making prediction: {e}")

print("\n=== All Tests Completed ===")
