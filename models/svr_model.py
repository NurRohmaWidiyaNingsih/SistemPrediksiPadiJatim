import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import itertools
import warnings
warnings.filterwarnings("ignore")

def svr_grid_search(X_train, y_train, X_test, y_test, C_values=None, gamma_values=None, epsilon_values=None, 
                   use_scaling=True, external_scaler_X=None):
    """
    Grid Search untuk mencari parameter SVR terbaik
    
    Parameters:
    -----------
    use_scaling : bool, default=True
        Jika True, akan melakukan scaling pada data
        Jika False, asumsikan X sudah di-scale dari luar, tapi Y masih perlu scaling internal
    external_scaler_X : StandardScaler, optional
        Scaler yang sudah di-fit untuk X (jika use_scaling=False)
    """
    # Parameter yang akan dicoba (gunakan default jika tidak disediakan)
    if C_values is None:
        C_values = [0.1, 1, 10, 100, 1000]
    if gamma_values is None:
        gamma_values = [0.001, 0.01, 0.1, 1, 'scale', 'auto']
    if epsilon_values is None:
        epsilon_values = [0.01, 0.1, 0.5, 1.0]

    best_mape = float('inf')
    best_params = None
    best_predictions = None

    # Setup scaling
    if use_scaling:
        # Normalisasi penuh internal
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_test_scaled = scaler_X.transform(X_test)
    else:
        # X sudah di-scale dari luar, tapi Y perlu scaling untuk training
        scaler_y = StandardScaler()
        X_train_scaled = X_train  # Sudah di-scale
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        X_test_scaled = X_test  # Sudah di-scale

    # Grid search
    total_combinations = len(C_values) * len(gamma_values) * len(epsilon_values)
    print(f"    Mencoba {total_combinations} kombinasi parameter...")

    counter = 0
    for C, gamma, epsilon in itertools.product(C_values, gamma_values, epsilon_values):
        counter += 1
        if counter % 20 == 0:
            print(f"    Progress: {counter}/{total_combinations} kombinasi")

        try:
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train_scaled, y_train_scaled)

            y_pred_scaled = model.predict(X_test_scaled)
            
            # SELALU inverse transform Y (baik use_scaling=True/False)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            mape = mean_absolute_percentage_error(y_test, y_pred) * 100

            if mape < best_mape:
                best_mape = mape
                best_params = {'C': C, 'gamma': gamma, 'epsilon': epsilon}
                best_predictions = y_pred
        except Exception as e:
            print(f"    Error dengan parameter C={C}, gamma={gamma}, epsilon={epsilon}: {e}")

    # Hitung RMSE dengan parameter terbaik
    rmse = np.sqrt(mean_squared_error(y_test, best_predictions))

    return best_params, best_mape, rmse, best_predictions

def run_svr(train, test, C_values=None, gamma_values=None, epsilon_values=None, use_scaling=True):
    """
    Jalankan model SVR dengan Grid Search
    Memproses per kabupaten dan kecamatan, lalu agregasi hasilnya.
    
    Parameters:
    -----------
    use_scaling : bool, default=True
        Jika True, akan melakukan scaling internal pada data
        Jika False, asumsikan data sudah di-scale dari button normalisasi di UI
    """
    # Deteksi nama kolom (case-insensitive)
    tahun_col = 'Tahun' if 'Tahun' in train.columns else 'tahun'
    produksi_col = 'Produksi' if 'Produksi' in train.columns else 'produksi'
    kabupaten_col = 'Kabupaten' if 'Kabupaten' in train.columns else 'kabupaten'
    kecamatan_col = 'Kecamatan' if 'Kecamatan' in train.columns else 'kecamatan'

    # Pastikan kolom yang diperlukan ada
    assert tahun_col in train.columns, f"Kolom '{tahun_col}' tidak ditemukan"
    assert produksi_col in train.columns, f"Kolom '{produksi_col}' tidak ditemukan"
    assert kabupaten_col in train.columns, f"Kolom '{kabupaten_col}' tidak ditemukan"
    assert kecamatan_col in train.columns, f"Kolom '{kecamatan_col}' tidak ditemukan"

    # Pastikan kolom Tahun adalah numerik - buat copy untuk menghindari SettingWithCopyWarning
    train = train.copy()
    test = test.copy()
    
    train[tahun_col] = pd.to_numeric(train[tahun_col], errors='coerce')
    test[tahun_col] = pd.to_numeric(test[tahun_col], errors='coerce')

    results = []
    all_y_test = []
    all_y_pred = []
    predictions_detail = []  # Untuk menyimpan detail prediksi per tahun
    
    print(f"\n{'='*60}")
    print(f"🔧 SVR Model - Scaling: {'ENABLED (Internal)' if use_scaling else 'DISABLED (Pre-scaled)'}")
    print(f"{'='*60}\n")
    
    # Loop per kabupaten dan kecamatan
    for kab in train[kabupaten_col].unique():
        print(f"\n=== Kabupaten: {kab} ===")
        # Ambil data per kabupaten
        train_kab = train[train[kabupaten_col] == kab]
        test_kab = test[test[kabupaten_col] == kab]

        # Loop untuk setiap kecamatan dalam kabupaten
        for kec in train_kab[kecamatan_col].unique():
            if kec not in test_kab[kecamatan_col].unique():
                print(f"  Kecamatan: {kec} - Tidak ada data testing, dilewati.")
                continue

            print(f"  Kecamatan: {kec}")
            # Filter data untuk kecamatan ini
            train_kec = train_kab[train_kab[kecamatan_col] == kec]
            test_kec = test_kab[test_kab[kecamatan_col] == kec]

            # Skip jika tidak ada data testing atau training kurang dari 2 titik
            if len(test_kec) == 0 or len(train_kec) < 2:
                print(f"    ⚠️ Data tidak cukup, dilewati.")
                continue

            # Persiapkan fitur - gunakan semua fitur yang tersedia
            feature_columns = ['Luas_Tanam', 'Luas_Panen', 'Produktivitas', 'luas_sawah']
            
            # Periksa apakah kolom ada di dataset (case-insensitive)
            available_features = []
            for col in feature_columns:
                # Cari kolom dengan case-insensitive
                matching_cols = [c for c in train_kec.columns if c.lower() == col.lower()]
                if matching_cols and matching_cols[0] in test_kec.columns:
                    available_features.append(matching_cols[0])
            
            if not available_features:
                print(f"    ⚠️ Tidak ada fitur yang cocok, dilewati.")
                continue

            # Tambahkan tahun sebagai fitur numerik
            train_kec = train_kec.copy()
            test_kec = test_kec.copy()
            train_kec['tahun_num'] = train_kec[tahun_col].astype(int)
            test_kec['tahun_num'] = test_kec[tahun_col].astype(int)
            available_features.append('tahun_num')

            try:
                X_train = train_kec[available_features].values
                y_train = train_kec[produksi_col].values
                X_test = test_kec[available_features].values
                y_test = test_kec[produksi_col].values

                # Jalankan grid search dengan parameter use_scaling
                best_params, mape_kec, rmse_kec, y_pred = svr_grid_search(
                    X_train, y_train, X_test, y_test, 
                    C_values=C_values, 
                    gamma_values=gamma_values, 
                    epsilon_values=epsilon_values,
                    use_scaling=use_scaling  # Pass parameter use_scaling
                )
                
                C = best_params['C']
                gamma = best_params['gamma']
                epsilon = best_params['epsilon']

                print(f"    Parameter terbaik: C={C}, gamma={gamma}, epsilon={epsilon}")
                print(f"    MAPE: {mape_kec:.2f}%, RMSE: {rmse_kec:.4f}")

                results.append({
                    'Kabupaten': kab,
                    'Kecamatan': kec,
                    'C': C,
                    'gamma': gamma,
                    'epsilon': epsilon,
                    'MAPE (%)': mape_kec,
                    'RMSE': rmse_kec
                })

                # Simpan detail prediksi per tahun
                for i, row in test_kec.iterrows():
                    idx = i - test_kec.index[0]  # Relative index within test_kec
                    if idx < len(y_pred):  # Ensure we don't go out of bounds
                        predictions_detail.append({
                            'Kabupaten': kab,
                            'Kecamatan': kec,
                            'Tahun': int(row[tahun_col]),
                            'Produksi_Aktual': row[produksi_col],
                            'Produksi_Prediksi': y_pred[idx],
                            'Error_Absolut': abs(row[produksi_col] - y_pred[idx]),
                            'Error_Persen': abs((row[produksi_col] - y_pred[idx]) / row[produksi_col]) * 100
                        })

                # Kumpulkan semua prediksi untuk evaluasi keseluruhan
                all_y_test.extend(y_test)
                all_y_pred.extend(y_pred)

            except Exception as e:
                print(f"    ❌ Error untuk {kab}, {kec}: {str(e)}")
                continue

    # Hitung MAPE dan RMSE keseluruhan
    # Menggunakan rata-rata MAPE per kecamatan (seperti di Colab)
    if results:
        results_df_temp = pd.DataFrame(results)
        mape_overall = results_df_temp['MAPE (%)'].mean()  # Rata-rata MAPE per kecamatan
        rmse_overall = results_df_temp['RMSE'].mean()  # Rata-rata RMSE per kecamatan
    else:
        mape_overall = 0
        rmse_overall = 0

    # Buat visualisasi ringkasan
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MAPE per kecamatan
    if results:
        results_df = pd.DataFrame(results)
        top_10 = results_df.nsmallest(10, 'MAPE (%)')
        
        ax1.barh(range(len(top_10)), top_10['MAPE (%)'].values)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels([f"{row['Kabupaten'][:10]}...\n{row['Kecamatan'][:10]}..." 
                             for _, row in top_10.iterrows()], fontsize=8)
        ax1.set_xlabel('MAPE (%)')
        ax1.set_title('Top 10 Kecamatan dengan MAPE Terendah (SVR)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Scatter plot aktual vs prediksi
        ax2.scatter(all_y_test, all_y_pred, alpha=0.5)
        ax2.plot([min(all_y_test), max(all_y_test)], 
                 [min(all_y_test), max(all_y_test)], 
                 'r--', label='Perfect Prediction')
        ax2.set_xlabel('Produksi Aktual')
        ax2.set_ylabel('Produksi Prediksi')
        ax2.set_title(f'Aktual vs Prediksi (SVR)\nMAPE: {mape_overall:.2f}%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()

    # Kembalikan prediksi dalam format yang sesuai dengan test data
    y_pred_return = pd.Series(all_y_pred, index=range(len(all_y_pred)))

    return {
        "mape": mape_overall,
        "rmse": rmse_overall,
        "y_pred": y_pred_return,
        "fig": fig,
        "summary": results,  # Ringkasan per kecamatan
        "details": predictions_detail  # Detail prediksi per tahun
    }
