import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class SVRPSOModel:
    """
    Model SVR dengan optimasi parameter menggunakan Particle Swarm Optimization (PSO)
    """
    
    def __init__(self, n_particles=250, n_iter=100, random_seed=None, use_scaling=True):
        """
        Inisialisasi model SVR-PSO
        
        Parameters:
        -----------
        n_particles : int, default=250
            Jumlah partikel dalam PSO
        n_iter : int, default=100
            Jumlah iterasi PSO
        random_seed : int, optional
            Seed untuk reproducibility
        use_scaling : bool, default=True
            Jika True, akan melakukan scaling pada data
            Jika False, asumsikan data sudah di-scale dari luar
        """
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.random_seed = random_seed
        self.use_scaling = use_scaling
        self.scaler_X = StandardScaler() if use_scaling else None
        self.scaler_y = StandardScaler() if use_scaling else None
        self.best_params = None
        self.best_model = None
        self.best_score = float('inf')
        self.history = []
        self.mape_progress = []
        
    def _evaluate_svr(self, params, X_train, y_train, X_test, y_test):
        """
        Evaluasi model SVR dengan parameter tertentu
        
        Parameters:
        -----------
        params : array-like
            Parameter [C, epsilon, gamma_val]
        X_train : array-like
            Data training (sudah di-scale jika use_scaling=False)
        y_train : array-like
            Target training (belum di-scale, akan di-scale internal)
        X_test : array-like
            Data testing (sudah di-scale jika use_scaling=False)
        y_test : array-like
            Target testing (asli, belum di-scale)
            
        Returns:
        --------
        mape : float
            Mean Absolute Percentage Error
        y_pred : array-like
            Prediksi (sudah di-inverse transform)
        params_dict : dict
            Parameter yang digunakan
        """
        C, epsilon, gamma_val = params
        
        # Konversi gamma dari indeks ke nilai aktual
        gamma_options = [0.001, 0.01, 0.1, 1, 'scale', 'auto']
        gamma_idx = int(gamma_val) % len(gamma_options)
        gamma = gamma_options[gamma_idx]
        
        try:
            # Buat model SVR dengan parameter
            model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
            
            # Y SELALU perlu scaling untuk training (baik use_scaling=True/False)
            if not hasattr(self, '_temp_scaler_y') or self._temp_scaler_y is None:
                self._temp_scaler_y = StandardScaler()
                y_train_scaled = self._temp_scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            else:
                y_train_scaled = self._temp_scaler_y.transform(y_train.reshape(-1, 1)).ravel()
            
            # Fit model
            model.fit(X_train, y_train_scaled)
            
            # Prediksi
            y_pred_scaled = model.predict(X_test)
            
            # SELALU inverse transform Y
            y_pred = self._temp_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Hitung MAPE
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            return mape, y_pred, {'C': C, 'epsilon': epsilon, 'gamma': gamma}, model
        except Exception as e:
            # Jika terjadi error, kembalikan nilai MAPE yang sangat tinggi
            return float('inf'), None, None, None
    
    def _pso_optimize(self, X_train, y_train, X_test, y_test, lb=None, ub=None, w=0.8, c1=1.7, c2=1.3, verbose=True, show_plot=False):
        """
        Algoritma PSO untuk optimasi parameter SVR
        
        Parameters:
        -----------
        X_train : array-like
            Data training (sudah di-scale jika use_scaling=False)
        y_train : array-like
            Target training (asli, akan di-scale internal untuk setiap evaluasi)
        X_test : array-like
            Data testing (sudah di-scale jika use_scaling=False)
        y_test : array-like
            Target testing (selalu belum di-scale untuk kalkulasi MAPE)
        verbose : bool, default=True
            Tampilkan progress
        show_plot : bool, default=False
            Tampilkan plot perubahan MAPE per iterasi
            
        Returns:
        --------
        best_params : dict
            Parameter terbaik
        best_score : float
            MAPE terbaik
        rmse : float
            RMSE dengan parameter terbaik
        best_pred : array-like
            Prediksi terbaik
        """
        # Set random seed jika ada
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Setup scaler hanya untuk X jika use_scaling=True
        if self.use_scaling:
            X_train_scaled = self.scaler_X.fit_transform(X_train)
            X_test_scaled = self.scaler_X.transform(X_test)
        else:
            # X sudah di-scale dari luar
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Reset temp scaler untuk Y (akan di-create di setiap evaluasi)
        self._temp_scaler_y = None
        
        # Definisi batas parameter (C, epsilon, gamma_idx)
        if lb is None:
            lb = np.array([0.1, 0.0001, 0.0001])  # lower bounds (default)
        if ub is None:
            ub = np.array([1000, 10.0, 1.0])  # upper bounds (default)
        
        # Inisialisasi posisi dan kecepatan partikel
        particles = np.random.uniform(lb, ub, (self.n_particles, 3))
        velocities = np.zeros((self.n_particles, 3))
        
        # Inisialisasi personal best dan global best
        personal_best_pos = particles.copy()
        personal_best_score = np.array([float('inf')] * self.n_particles)
        
        global_best_pos = None
        global_best_score = float('inf')
        global_best_pred = None
        global_best_params = None
        global_best_model = None
        
        # Konstanta PSO (menggunakan parameter yang diberikan)
        # w = inertia, c1 = cognitive, c2 = social
        
        # Reset mape_progress
        self.mape_progress = []
        
        # Iterasi PSO
        for i in range(self.n_iter):
            for j in range(self.n_particles):
                # Evaluasi partikel
                score, pred, params, model = self._evaluate_svr(
                    particles[j],
                    X_train_scaled,
                    y_train,  # Pass y_train asli, akan di-scale di dalam _evaluate_svr
                    X_test_scaled,
                    y_test
                )
                
                # Update personal best
                if score < personal_best_score[j]:
                    personal_best_pos[j] = particles[j].copy()
                    personal_best_score[j] = score
                    
                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_pos = particles[j].copy()
                        global_best_pred = pred
                        global_best_params = params
                        global_best_model = model
            
            # Simpan MAPE terbaik per iterasi
            self.mape_progress.append(global_best_score)
            
            # Simpan history
            self.history.append({
                'iteration': i + 1,
                'best_score': global_best_score,
                'best_params': global_best_params.copy() if global_best_params else None
            })
            
            # Update kecepatan dan posisi partikel
            r1, r2 = np.random.rand(2)
            for j in range(self.n_particles):
                velocities[j] = (w * velocities[j] +
                               c1 * r1 * (personal_best_pos[j] - particles[j]) +
                               c2 * r2 * (global_best_pos - particles[j]))
                
                particles[j] += velocities[j]
                
                # Batasi posisi partikel dalam batas yang ditentukan
                particles[j] = np.clip(particles[j], lb, ub)
            
            # Tampilkan progress
            if verbose and (i % 5 == 0 or i == self.n_iter - 1):
                print(f"    Iterasi {i+1}/{self.n_iter}, MAPE terbaik: {global_best_score:.2f}%")
                if global_best_params:
                    print(f"    Param terbaik: C={global_best_params['C']:.3f}, "
                          f"epsilon={global_best_params['epsilon']:.3f}, "
                          f"gamma={global_best_params['gamma']}")
        
        # Plot grafik MAPE jika diminta
        if show_plot and len(self.mape_progress) > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(self.mape_progress, marker='o', color='blue')
            plt.title("Perubahan MAPE selama Iterasi PSO")
            plt.xlabel("Iterasi")
            plt.ylabel("MAPE (%)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        
        # Hitung RMSE dengan parameter terbaik
        if global_best_pred is not None:
            rmse = np.sqrt(mean_squared_error(y_test, global_best_pred))
        else:
            rmse = float('inf')
        
        return global_best_params, global_best_score, rmse, global_best_pred, global_best_model
    
    def fit(self, X_train, y_train, X_test, y_test, lb=None, ub=None, w=0.8, c1=1.7, c2=1.3, verbose=True, show_plot=False):
        """
        Training model SVR dengan optimasi PSO
        
        Parameters:
        -----------
        X_train : array-like or DataFrame
            Data training
        y_train : array-like or Series
            Target training
        X_test : array-like or DataFrame
            Data testing untuk evaluasi
        y_test : array-like or Series
            Target testing untuk evaluasi
        lb : array-like, optional
            Lower bounds untuk [C, epsilon, gamma]
        ub : array-like, optional
            Upper bounds untuk [C, epsilon, gamma]
        w : float, default=0.8
            Inertia weight untuk PSO
        c1 : float, default=1.7
            Cognitive parameter untuk PSO
        c2 : float, default=1.3
            Social parameter untuk PSO
        verbose : bool, default=True
            Tampilkan progress
        show_plot : bool, default=False
            Tampilkan plot perubahan MAPE per iterasi
            
        Returns:
        --------
        self : object
            Instance model yang sudah di-train
        """
        # Convert to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Reset history
        self.history = []
        self.mape_progress = []
        
        # Jalankan optimasi PSO
        if verbose:
            print(f"Menjalankan optimasi PSO dengan {self.n_particles} partikel dan {self.n_iter} iterasi...")
            if self.random_seed is not None:
                print(f"Random seed: {self.random_seed}")
        
        self.best_params, self.best_score, self.best_rmse, self.best_pred, self.best_model = self._pso_optimize(
            X_train, y_train, X_test, y_test, lb=lb, ub=ub, w=w, c1=c1, c2=c2, verbose=verbose, show_plot=show_plot
        )
        
        if self.best_params and verbose:
            print(f"\nHasil akhir:")
            print(f"  C: {self.best_params['C']:.3f}")
            print(f"  gamma: {self.best_params['gamma']}")
            print(f"  epsilon: {self.best_params['epsilon']:.3f}")
            print(f"  MAPE: {self.best_score:.2f}%")
            print(f"  RMSE: {self.best_rmse:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Prediksi menggunakan model terbaik
        
        Parameters:
        -----------
        X : array-like or DataFrame
            Data untuk prediksi
            
        Returns:
        --------
        y_pred : array-like
            Hasil prediksi
        """
        if self.best_model is None:
            raise ValueError("Model belum di-train! Jalankan fit() terlebih dahulu.")
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Scale input jika use_scaling=True
        if self.use_scaling and self.scaler_X is not None:
            X_scaled = self.scaler_X.transform(X)
        else:
            # Jika use_scaling=False, asumsikan X sudah di-scale dari luar
            X_scaled = X
        
        # Prediksi
        y_pred_scaled = self.best_model.predict(X_scaled)
        
        # Inverse transform Y (Y SELALU perlu inverse transform karena selalu di-scale saat training)
        if hasattr(self, '_temp_scaler_y') and self._temp_scaler_y is not None:
            y_pred = self._temp_scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        elif self.use_scaling and self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled
        
        return y_pred
    
    def get_params(self):
        """
        Mendapatkan parameter terbaik
        
        Returns:
        --------
        params : dict
            Parameter terbaik hasil optimasi
        """
        return self.best_params
    
    def get_metrics(self):
        """
        Mendapatkan metrik evaluasi
        
        Returns:
        --------
        metrics : dict
            Dictionary berisi MAPE dan RMSE
        """
        return {
            'MAPE': self.best_score,
            'RMSE': self.best_rmse
        }
    
    def get_mape_progress(self):
        """
        Mendapatkan progress MAPE per iterasi
        
        Returns:
        --------
        mape_progress : list
            List MAPE terbaik per iterasi
        """
        return self.mape_progress
    
    def get_history(self):
        """
        Mendapatkan history optimasi
        
        Returns:
        --------
        history : list
            List dictionary berisi history setiap iterasi
        """
        return self.history
    
    def plot_mape_progress(self, save_path=None):
        """
        Plot grafik perubahan MAPE per iterasi
        
        Parameters:
        -----------
        save_path : str, optional
            Path untuk menyimpan gambar. Jika None, hanya ditampilkan
        """
        if len(self.mape_progress) == 0:
            print("Belum ada data MAPE progress. Jalankan fit() terlebih dahulu.")
            return
        
        plt.figure(figsize=(6, 4))
        plt.plot(self.mape_progress, marker='o', color='blue')
        plt.title("Perubahan MAPE selama Iterasi PSO")
        plt.xlabel("Iterasi")
        plt.ylabel("MAPE (%)")
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot disimpan ke: {save_path}")
        
        plt.show()


def train_svr_pso_per_region(train_data, test_data, 
                              region_col='kabupaten', 
                              subregion_col='Kecamatan',
                              target_col='Produksi',
                              feature_cols=None,
                              n_particles=250,
                              n_iter=100,
                              random_seed=None,
                              lb=None,
                              ub=None,
                              w=0.8,
                              c1=1.7,
                              c2=1.3,
                              show_plot=False,
                              verbose=True,
                              use_scaling=True):
    """
    Training model SVR-PSO per kabupaten dan kecamatan
    
    Parameters:
    -----------
    train_data : DataFrame
        Data training
    test_data : DataFrame
        Data testing
    region_col : str, default='kabupaten'
        Nama kolom kabupaten
    subregion_col : str, default='Kecamatan'
        Nama kolom kecamatan
    target_col : str, default='Produksi'
        Nama kolom target
    feature_cols : list, optional
        List nama kolom fitur. Jika None, akan menggunakan default features
    n_particles : int, default=250
        Jumlah partikel PSO
    n_iter : int, default=100
        Jumlah iterasi PSO
    random_seed : int, optional
        Seed untuk reproducibility
    show_plot : bool, default=False
        Tampilkan plot MAPE per iterasi untuk setiap kecamatan
    verbose : bool, default=True
        Tampilkan progress
    use_scaling : bool, default=True
        Jika True, akan melakukan scaling pada data
        Jika False, asumsikan data sudah di-scale dari luar
        
    Returns:
    --------
    results : DataFrame
        Hasil evaluasi per kecamatan
    predictions : DataFrame
        Prediksi per kecamatan
    models : dict
        Dictionary model per kecamatan
    """
    # Default feature columns
    if feature_cols is None:
        feature_cols = ['Luas_Tanam', 'Luas_Panen', 'Produktivitas', 'luas_sawah']
    
    results = []
    predictions_all = []
    models = {}
    
    # Pastikan format tahun - buat copy untuk menghindari SettingWithCopyWarning
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    # Pastikan kolom Tahun adalah numerik
    if 'Tahun' in train_data.columns:
        train_data['Tahun'] = pd.to_numeric(train_data['Tahun'], errors='coerce')
    if 'Tahun' in test_data.columns:
        test_data['Tahun'] = pd.to_numeric(test_data['Tahun'], errors='coerce')
    
    # Iterasi per kabupaten
    for kab in train_data[region_col].unique():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Kabupaten: {kab}")
            print(f"{'='*50}")
        
        train_kab = train_data[train_data[region_col] == kab]
        test_kab = test_data[test_data[region_col] == kab]
        
        # Iterasi per kecamatan
        for kec in train_kab[subregion_col].unique():
            if kec not in test_kab[subregion_col].unique():
                if verbose:
                    print(f"  Kecamatan: {kec} - Tidak ada data testing, dilewati.")
                continue
            
            if verbose:
                print(f"\n  Kecamatan: {kec}")
            
            train_kec = train_kab[train_kab[subregion_col] == kec].copy()
            test_kec = test_kab[test_kab[subregion_col] == kec].copy()
            
            # Pastikan data cukup
            if len(test_kec) == 0 or len(train_kec) < 2:
                if verbose:
                    print(f"    ⚠️ Data tidak cukup, dilewati.")
                continue
            
            # Periksa fitur yang tersedia
            available_features = [col for col in feature_cols
                                 if col in train_kec.columns and col in test_kec.columns]
            
            if not available_features:
                if verbose:
                    print(f"    ⚠️ Tidak ada fitur yang cocok, dilewati.")
                continue
            
            # Tambahkan tahun sebagai fitur numerik jika ada
            if 'Tahun' in train_kec.columns:
                train_kec['tahun_num'] = train_kec['Tahun'].astype(int)
                test_kec['tahun_num'] = test_kec['Tahun'].astype(int)
                available_features.append('tahun_num')
            
            if verbose:
                print(f"    Menggunakan fitur: {', '.join(available_features)}")
            
            # Persiapkan data
            X_train = train_kec[available_features].values
            y_train = train_kec[target_col].values
            X_test = test_kec[available_features].values
            y_test = test_kec[target_col].values
            
            # Training model
            try:
                model = SVRPSOModel(n_particles=n_particles, n_iter=n_iter, random_seed=random_seed, use_scaling=use_scaling)
                model.fit(X_train, y_train, X_test, y_test, lb=lb, ub=ub, w=w, c1=c1, c2=c2, verbose=verbose, show_plot=show_plot)
                
                if model.best_params:
                    # Simpan model
                    models[f"{kab}_{kec}"] = model
                    
                    # Simpan hasil evaluasi
                    results.append({
                        'Kabupaten': kab,
                        'Kecamatan': kec,
                        'C': model.best_params['C'],
                        'gamma': model.best_params['gamma'],
                        'epsilon': model.best_params['epsilon'],
                        'MAPE (%)': model.best_score,
                        'RMSE': model.best_rmse
                    })
                    
                    # Simpan prediksi
                    y_pred = model.predict(X_test)
                    for i, row in test_kec.iterrows():
                        idx = i - test_kec.index[0]  # Relative index within test_kec
                        if idx < len(y_pred):  # Ensure we don't go out of bounds
                            pred_data = {
                                'kabupaten': kab,
                                'Kecamatan': kec,
                                'Produksi_Prediksi': y_pred[idx],
                                'Produksi_Aktual': row[target_col]
                            }
                            
                            # Tambahkan tahun jika ada
                            if 'Tahun' in test_kec.columns:
                                pred_data['Tahun'] = int(row['Tahun'])
                            
                            predictions_all.append(pred_data)
                else:
                    if verbose:
                        print(f"    ⚠️ Optimasi gagal untuk {kec}, dilewati.")
                        
            except Exception as e:
                if verbose:
                    print(f"    ⚠️ Error saat training {kec}: {str(e)}")
                continue
    
    # Convert ke DataFrame
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    predictions_df = pd.DataFrame(predictions_all) if predictions_all else pd.DataFrame()
    
    # Tampilkan ringkasan
    if not results_df.empty and verbose:
        print(f"\n{'='*50}")
        print("📊 RINGKASAN HASIL EVALUASI SVR-PSO")
        print(f"{'='*50}")
        
        results_df = results_df.sort_values(['Kabupaten', 'MAPE (%)'])
        
        for kab in results_df['Kabupaten'].unique():
            hasil_kab = results_df[results_df['Kabupaten'] == kab]
            print(f"\nKabupaten {kab}:")
            print(hasil_kab[['Kecamatan', 'C', 'gamma', 'epsilon', 'MAPE (%)', 'RMSE']].to_string(index=False))
            print(f"Rata-rata MAPE: {hasil_kab['MAPE (%)'].mean():.2f}%")
        
        print(f"\n{'='*50}")
        print(f"Rata-rata MAPE keseluruhan: {results_df['MAPE (%)'].mean():.2f}%")
        
        best_idx = results_df['MAPE (%)'].idxmin()
        worst_idx = results_df['MAPE (%)'].idxmax()
        
        print(f"MAPE terendah: {results_df.loc[best_idx, 'MAPE (%)']:.2f}% "
              f"({results_df.loc[best_idx, 'Kabupaten']}, {results_df.loc[best_idx, 'Kecamatan']})")
        print(f"MAPE tertinggi: {results_df.loc[worst_idx, 'MAPE (%)']:.2f}% "
              f"({results_df.loc[worst_idx, 'Kabupaten']}, {results_df.loc[worst_idx, 'Kecamatan']})")
        print(f"{'='*50}")
    
    return results_df, predictions_df, models


def run_svr_pso(train, test, n_particles=50, n_iter=100, random_seed=5, lb=None, ub=None, w=0.8, c1=1.7, c2=1.3, use_scaling=True):
    """
    Fungsi wrapper untuk menjalankan SVR-PSO dari Streamlit
    Kompatibel dengan format app.py
    
    Parameters:
    -----------
    train : DataFrame
        Data training
    test : DataFrame
        Data testing
    n_particles : int, default=50
        Jumlah partikel PSO
    n_iter : int, default=100
        Jumlah iterasi PSO
    random_seed : int, default=2
        Random seed untuk reproducibility
    use_scaling : bool, default=True
        Jika True, akan melakukan scaling internal pada data
        Jika False, asumsikan data sudah di-scale dari button normalisasi di UI
        
    Returns:
    --------
    dict dengan keys:
        - mape: float (rata-rata MAPE keseluruhan)
        - rmse: float (rata-rata RMSE keseluruhan)
        - y_pred: Series (prediksi)
        - fig: matplotlib figure
        - summary: list (ringkasan per kecamatan)
        - details: list (detail prediksi per tahun)
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*60}")
    print(f"🔧 SVR-PSO Model - Scaling: {'ENABLED (Internal)' if use_scaling else 'DISABLED (Pre-scaled)'}")
    print(f"{'='*60}\n")
    
    # Deteksi nama kolom (case-insensitive)
    tahun_col = 'Tahun' if 'Tahun' in train.columns else 'tahun'
    produksi_col = 'Produksi' if 'Produksi' in train.columns else 'produksi'
    kabupaten_col = 'Kabupaten' if 'Kabupaten' in train.columns else 'kabupaten'
    kecamatan_col = 'Kecamatan' if 'Kecamatan' in train.columns else 'kecamatan'
    
    # Training untuk semua region
    results_df, predictions_df, models = train_svr_pso_per_region(
        train, 
        test,
        region_col=kabupaten_col,
        subregion_col=kecamatan_col,
        target_col=produksi_col,
        feature_cols=['Luas_Tanam', 'Luas_Panen', 'Produktivitas', 'luas_sawah'],
        n_particles=n_particles,
        n_iter=n_iter,
        random_seed=random_seed,
        lb=lb,
        ub=ub,
        w=w,
        c1=c1,
        c2=c2,
        show_plot=False,
        verbose=True,
        use_scaling=use_scaling  # Pass parameter use_scaling
    )
    
    # Hitung MAPE dan RMSE keseluruhan
    if not results_df.empty:
        mape_overall = results_df['MAPE (%)'].mean()
        rmse_overall = results_df['RMSE'].mean()
    else:
        mape_overall = 0
        rmse_overall = 0
    
    # Siapkan data untuk visualisasi
    all_y_test = []
    all_y_pred = []
    
    if not predictions_df.empty:
        all_y_test = predictions_df['Produksi_Aktual'].values
        all_y_pred = predictions_df['Produksi_Prediksi'].values
    
    # Buat visualisasi ringkasan
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: MAPE per kecamatan
    if not results_df.empty:
        top_10 = results_df.nsmallest(10, 'MAPE (%)')
        
        ax1.barh(range(len(top_10)), top_10['MAPE (%)'].values)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels([f"{row['Kabupaten'][:10]}...\n{row['Kecamatan'][:10]}..." 
                             for _, row in top_10.iterrows()], fontsize=8)
        ax1.set_xlabel('MAPE (%)')
        ax1.set_title('Top 10 Kecamatan dengan MAPE Terendah (SVR-PSO)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Scatter plot aktual vs prediksi
        if len(all_y_test) > 0:
            ax2.scatter(all_y_test, all_y_pred, alpha=0.5)
            ax2.plot([min(all_y_test), max(all_y_test)], 
                     [min(all_y_test), max(all_y_test)], 
                     'r--', label='Perfect Prediction')
            ax2.set_xlabel('Produksi Aktual')
            ax2.set_ylabel('Produksi Prediksi')
            ax2.set_title(f'Aktual vs Prediksi (SVR-PSO)\nMAPE: {mape_overall:.2f}%')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # Siapkan detail prediksi dalam format yang sesuai
    predictions_detail = []
    if not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            detail = {
                'Kabupaten': row['kabupaten'],
                'Kecamatan': row['Kecamatan'],
                'Produksi_Aktual': row['Produksi_Aktual'],
                'Produksi_Prediksi': row['Produksi_Prediksi'],
                'Error_Absolut': abs(row['Produksi_Aktual'] - row['Produksi_Prediksi']),
                'Error_Persen': abs((row['Produksi_Aktual'] - row['Produksi_Prediksi']) / row['Produksi_Aktual']) * 100
            }
            if 'Tahun' in row:
                detail['Tahun'] = row['Tahun']
            predictions_detail.append(detail)
    
    # Siapkan summary dalam format yang sesuai
    summary = []
    if not results_df.empty:
        for _, row in results_df.iterrows():
            summary.append({
                'Kabupaten': row['Kabupaten'],
                'Kecamatan': row['Kecamatan'],
                'C': row['C'],
                'gamma': row['gamma'],
                'epsilon': row['epsilon'],
                'MAPE': row['MAPE (%)'],
                'RMSE': row['RMSE']
            })
    
    # Kembalikan prediksi dalam format Series
    y_pred_return = pd.Series(all_y_pred, index=range(len(all_y_pred)))
    
    return {
        "mape": mape_overall,
        "rmse": rmse_overall,
        "y_pred": y_pred_return,
        "fig": fig,
        "summary": summary,
        "details": predictions_detail
    }
