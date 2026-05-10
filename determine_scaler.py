"""
Determine optimal scaler untuk model SVR
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model_path = "model_svr_rbf_90_30_partikel_100_iterasi.save"
model = joblib.load(model_path)

print("=== SCALER ANALYSIS ===\n")

# Dari analisis support vectors, kita tahu range adalah [0, 1]
# Sekarang kita perlu determine: apa min/max dari data training?

# From SVR, dukungan vectors give us clues tentang data distribution
sv = model.support_vectors_

print("Support Vectors Statistics (used for training):")
print(f"Min range: {sv.min()}, Max range: {sv.max()}")
print(f"All values scaled to [0, 1] range\n")

# Untuk mencapai range [0,1], kita perlu tahu original (min, max) dari training data
# Kita bisa reverse-engineer dari interc intercept dan dual coefficients

# Assumption: data scaled dengan MinMaxScaler(feature_range=(0, 1))
# Kalau kita punya nilai scaled, bisa balik ke original dengan:
# X_original = X_scaled * (X_max - X_min) + X_min

# Tapi kita tidak tahu X_min dan X_max dari data training
# Kita perlu ESTIMATE berdasarkan reasonable ranges

print("ESTIMATED TRAINING DATA RANGES (based on agricultural domain knowledge):")
print("="*60)

# Reasonable ranges untuk features Jawa Timur:
ranges = {
    'Luas Panen': (100, 50000),  # 100 - 50,000 ha (typical)
    'Curah Hujan': (50, 400),     # 50 - 400 mm/bulan
    'Kelembapan': (40, 95),        # 40 - 95 % (tropical climate)
    'Suhu': (20, 35),              # 20 - 35 °C (tropical)
    'Kecepatan Angin': (0.5, 15),  # 0.5 - 15 m/s
    'Sinar Matahari': (1, 12),     # 1 - 12 jam/hari
    'Tahun': (2018, 2024),         # Training data range
    'Location_onehot': (0, 1),     # Binary
    'Month_sin': (-1, 1),          # sin range
    'Month_cos': (-1, 1),          # cos range
}

print("\nFeature Ranges:")
for feature, (min_val, max_val) in ranges.items():
    print(f"  {feature:20s}: [{min_val:8.2f}, {max_val:8.2f}]")

print("\n" + "="*60)
print("IMPLICATIONS FOR PREDICTION:")
print("="*60)

# Untuk input user:
#   Luas Panen: 1000 ha → scale to [0,1] = (1000-100)/(50000-100) ≈ 0.018
#   Curah Hujan: 150 mm → scale to [0,1] = (150-50)/(400-50) ≈ 0.286
#   dst...

test_inputs = {
    'Luas Panen': 1000,
    'Curah Hujan': 150,
    'Kelembapan': 75,
    'Suhu': 27,
    'Kecepatan Angin': 3,
    'Sinar Matahari': 7,
    'Tahun': 2024,
}

print("\nTest Input Transformation:")
print(f"{'Feature':<20} {'Raw Value':>12} {'Min':>10} {'Max':>10} {'Scaled':>10}")
print("-"*62)

for feature, value in test_inputs.items():
    min_val, max_val = ranges[feature]
    scaled = (value - min_val) / (max_val - min_val)
    scaled = np.clip(scaled, 0, 1)  # Clamp to [0,1]
    print(f"{feature:<20} {value:>12.2f} {min_val:>10.2f} {max_val:>10.2f} {scaled:>10.4f}")

print("\n" + "="*60)
print("TEST PREDICTION with Scaled Input:")
print("="*60)

# Prepare scaled input
features_scaled = np.zeros(47)
scaled_dict = {}

# Numeric features dengan scaling
numeric_ranges = [
    (0, 'Luas Panen', (100, 50000)),
    (1, 'Curah Hujan', (50, 400)),
    (2, 'Kelembapan', (40, 95)),
    (3, 'Suhu', (20, 35)),
    (4, 'Kecepatan Angin', (0.5, 15)),
    (5, 'Sinar Matahari', (1, 12)),
    (6, 'Tahun', (2018, 2024)),
]

values = [1000, 150, 75, 27, 3, 7, 2024]

for idx, feature, (min_val, max_val) in numeric_ranges:
    value = values[idx]
    scaled_val = (value - min_val) / (max_val - min_val)
    scaled_val = np.clip(scaled_val, 0, 1)
    features_scaled[idx] = scaled_val
    scaled_dict[feature] = scaled_val

# One-hot untuk Bangkalan
features_scaled[7] = 1  # Index 7 adalah Kabupaten Bangkalan

# Sin/cos untuk bulan 6
bulan_rad = (6 - 1) * (2 * np.pi / 12)
features_scaled[45] = np.sin(bulan_rad)  # Ini sudah dalam range [-1, 1], tapi perlu scale ke [0,1]
features_scaled[46] = np.cos(bulan_rad)

# Scale sin/cos dari [-1,1] to [0,1]
features_scaled[45] = (np.sin(bulan_rad) + 1) / 2  # [-1,1] -> [0,1]
features_scaled[46] = (np.cos(bulan_rad) + 1) / 2  # [-1,1] -> [0,1]

X_scaled = features_scaled.reshape(1, -1)

pred = model.predict(X_scaled)[0]
print(f"\nPrediksi dengan scaled input: {pred:.4f} ton")

# Ini masih sangat kecil. Mungkin outputnya juga di-scale?
# Kalau output juga di-scale ke [0,1], perlu scale back ke produksi realistis

# Asumsi produksi range: 100 - 100,000 ton untuk luas 100-50,000 ha
output_range_min = 100
output_range_max = 100000

pred_unscaled = pred * (output_range_max - output_range_min) + output_range_min
print(f"Jika output di-scale [0,1], inverse scaled: {pred_unscaled:.2f} ton")

# Atau mungkin output sudah sesuai real (ton), lihat dari intercept
print(f"\nModel intercept (default prediction): {model.intercept_[0]:.4f} ton")
