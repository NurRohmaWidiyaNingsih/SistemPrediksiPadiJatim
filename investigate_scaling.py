"""
Investigasi model untuk melihat apakah perlu scaling
"""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

model_path = "model_svr_rbf_90_30_partikel_100_iterasi.save"
model = joblib.load(model_path)

print("=== MODEL INVESTIGATION ===\n")

# Check model attributes
print("Model Attributes:")
print(f"  - n_features_in_: {model.n_features_in_}")
print(f"  - feature_names_in_: {model.feature_names_in_}")
print(f"  - Kernel: {model.kernel}")
print(f"  - C: {model.C}")
print(f"  - Gamma: {model.gamma}")
print(f"  - Coef0: {model.coef0}")

# Check if has scaling info
print("\n" + "="*50)
print("Model Parameters Range Check:")

# Cek range dari dual_coef_ untuk estimasi range data yang dilatih
if hasattr(model, 'dual_coef_'):
    print(f"  - dual_coef_ range: [{model.dual_coef_.min():.6f}, {model.dual_coef_.max():.6f}]")
    
if hasattr(model, 'intercept_'):
    print(f"  - intercept_: {model.intercept_}")

if hasattr(model, 'support_vectors_'):
    sv = model.support_vectors_
    print(f"\nSupport Vectors Analysis:")
    print(f"  - Count: {len(sv)}")
    print(f"  - Shape: {sv.shape}")
    print(f"  - Min value: {sv.min():.6f}")
    print(f"  - Max value: {sv.max():.6f}")
    print(f"  - Mean value: {sv.mean():.6f}")
    print(f"  - Std value: {sv.std():.6f}")
    
    # Per fitur
    print(f"\n  Support Vector Stats per Feature (first 10):")
    for i in range(min(10, sv.shape[1])):
        print(f"    Feature {i}: [{sv[:, i].min():.4f}, {sv[:, i].max():.4f}], mean={sv[:, i].mean():.4f}, std={sv[:, i].std():.4f}")

# Test predictions dengan scaled vs unscaled
print("\n" + "="*50)
print("Test Predictions (Scaled vs Unscaled):")

# Prepare raw input
luas_panen = 1000
curah_hujan = 150
kelembapan = 75
suhu = 27
kecepatan_angin = 3
sinar_matahari = 7
tahun = 2024

all_locations = ['Kabupaten Bangkalan'] + [f'Kabupaten X{i}' for i in range(37)]

features_raw = np.zeros(47)
features_raw[0] = luas_panen
features_raw[1] = curah_hujan
features_raw[2] = kelembapan
features_raw[3] = suhu
features_raw[4] = kecepatan_angin
features_raw[5] = sinar_matahari
features_raw[6] = tahun
features_raw[7] = 1  # Bangkalan encoding
bulan_rad = 5 * (2 * np.pi / 12)  # bulan 6
features_raw[45] = np.sin(bulan_rad)
features_raw[46] = np.cos(bulan_rad)

X_raw = features_raw.reshape(1, -1)

# Predict dengan raw
pred_raw = model.predict(X_raw)[0]
print(f"\n1. Raw Input Prediction: {pred_raw:.6f}")

# Coba dengan scaling (z-score normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
pred_scaled = model.predict(X_scaled)[0]
print(f"2. Scaled Input Prediction: {pred_scaled:.6f}")

# Coba dengan min-max scaling
X_minmax = (X_raw - np.min(X_raw)) / (np.max(X_raw) - np.min(X_raw) + 1e-8)
pred_minmax = model.predict(X_minmax)[0]
print(f"3. Min-Max Scaled Prediction: {pred_minmax:.6f}")

# Coba normalize (0-1)
X_norm = X_raw / (np.abs(X_raw).max() + 1e-8)
pred_norm = model.predict(X_norm)[0]
print(f"4. Normalized Prediction: {pred_norm:.6f}")

print("\n" + "="*50)
print("Conclusions:")
print("Jika raw prediction mendominasi (>>1000), kemungkinan besar data TIDAK di-scale saat training")
print("Jika scaled prediction lebih reasonable, maka data DI-SCALE saat training")
