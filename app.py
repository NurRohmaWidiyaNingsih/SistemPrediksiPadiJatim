"""
SI-PADI JATIM - Sistem Prediksi Produksi Padi Berbasis Machine Learning
Menggunakan SVR dengan PSO Optimization dan Kernel ANOVA RBf
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import io
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== SCALER CONFIGURATION ====================
# Model dilatih dengan Min-Max scaling ke range [0, 1]
# Ini adalah range yang diperkirakan dari data training (2018-2024)
FEATURE_RANGES = {
    'Luas Panen': (100, 50000),           # ha
    'Curah Hujan': (50, 400),             # mm/bulan
    'Kelembapan': (40, 95),               # %
    'Suhu': (20, 35),                     # °C
    'Kecepatan Angin': (0.5, 15),         # m/s
    'Sinar Matahari': (1, 12),            # jam/hari
    'Tahun': (2018, 2026),                # years
    'Produksi': (100, 100000),            # ton (untuk inverse scaling output)
}

# ==================== GLOBAL CONSTANTS ====================
ALL_LOCATIONS = [
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

BULAN_NAMES = [
    'Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
    'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'
]

BULAN_MAP = {
    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
    'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
    'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
}

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="SI-PADI JATIM",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
def load_custom_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback inline CSS jika file tidak ada
        st.markdown("""
        <style>
        :root {
            --primary: #2e7d32;
            --primary-light: #66bb6a;
            --primary-dark: #1b5e20;
        }
        </style>
        """, unsafe_allow_html=True)

load_custom_css()

# ==================== UTILITY FUNCTIONS ====================
def format_dataframe_display(df):
    """Format dataframe untuk tampilan yang konsisten"""
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if col not in ['Tahun', 'tahun']:
            display_df[col] = display_df[col].apply(
                lambda x: f'{id_format(x, 2)}' if pd.notna(x) and x % 1 != 0 else f'{int(x)}'
            )
    return display_df

def validate_input_data(luas_panen, produktivitas):
    """Validasi input data"""
    if luas_panen <= 0 or produktivitas <= 0:
        return False, "Luas panen dan produktivitas harus lebih dari 0"
    return True, "✅ Valid"

def custom_header(title, subtitle=""):
    """Custom header yang aesthetic dengan text visibility yang jelas"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); 
                padding: 30px; border-radius: 12px; margin-bottom: 30px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
        <h1 style="color: rgb(255, 255, 255) !important; margin: 0 !important; font-size: 2.5rem !important; font-weight: 900 !important; letter-spacing: -0.5px !important;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important; display: block !important; padding: 0 !important;">
            {title}
        </h1>
        {f'<h3 style="color: rgb(255, 255, 255) !important; margin: 8px 0 0 0 !important; font-size: 1.05rem !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.2) !important; display: block !important; font-weight: 400 !important; padding: 0 !important;">{subtitle}</h3>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def section_header(title, level=2):
    """Section header dengan visibility jelas di light & dark mode"""
    if level == 2:
        st.markdown(f"""
        <div style="background: #ffffff; padding: 12px 20px; border-radius: 6px;
                    border-left: 4px solid #2e7d32; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                    margin: 16px 0; color: #000000; font-size: 1.2rem; font-weight: 700;">
            {title}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="border-bottom: 2px solid #2e7d32; padding: 6px 0; margin: 14px 0 8px 0;
                    color: #000000; font-size: 2rem; font-weight: 600;">
            {title}
        </div>
        """, unsafe_allow_html=True)

def page_info_box(content: str):
    """Kotak informasi ringkas di awal halaman menu (Markdown native)"""
    with st.container(border=True):
        st.markdown(content)

def build_model_bundle(results: dict) -> dict:
    """Susun bundle model untuk serialisasi joblib"""
    bundle = {
        "model": results.get("model"),
        "scaler_X": results.get("scaler_X"),
        "scaler_y": results.get("scaler_y"),
        "feature_columns": results.get("feature_columns"),
        "best_params": results.get("best_params"),
        "kernel": results.get("kernel"),
        "scenario": results.get("scenario"),
        "metrics": {
            "rmse": results.get("rmse"),
            "r2": results.get("r2"),
        },
    }
    if results.get("is_cv"):
        bundle["metrics"]["cv_avg_rmse_5fold"] = results.get("cv_avg_rmse_5fold")
        bundle["metrics"]["cv_avg_rmse_10fold"] = results.get("cv_avg_rmse_10fold")
    return bundle

def id_format(value, decimals=0):
    """Format angka ke format Indonesia: titik=ribuan, koma=desimal"""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "-"
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")

def metric_card(label, value, icon="", change=None):
    """Custom metric card"""
    change_html = ""
    if change is not None:
        color = "#2e7d32" if change > 0 else "#d32f2f"
        arrow = "↑" if change > 0 else "↓"
        change_html = f'<span style="color: {color}; font-size: 0.9rem; font-weight: 600;">{arrow} {id_format(abs(change), 1)}%</span>'
    
    return f"""
    <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                padding: 24px; border-radius: 12px; border-left: 5px solid #2e7d32;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; margin: 8px;">
        <p style="margin: 0; color: #558b2f; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
            {label}
        </p>
        <h2 style="margin: 8px 0; color: #1b5e20; font-size: 2.2rem; font-weight: 900;">
            {icon} {value}
        </h2>
        {change_html}
    </div>
    """

# ==================== MODEL FUNCTIONS ====================
def prepare_input_for_model(luas_panen, curah_hujan, kelembapan, suhu, kecepatan_angin, sinar_matahari, tahun, kabupaten, bulan):
    """Siapkan input data sesuai format model SVR (47 fitur) dengan SCALING ke [0,1]"""
    
    # Inisialisasi array dengan 47 fitur
    features = np.zeros(47)
    
    # **STEP 1: Scale numeric features ke [0, 1] sesuai dengan training range**
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
    
    # **STEP 2: One-hot encoding untuk kabupaten (7-40)**
    if kabupaten in ALL_LOCATIONS:
        idx = ALL_LOCATIONS.index(kabupaten)
        features[7 + idx] = 1
    
    # **STEP 3: Bulan encoding (sin/cos) - fitur 45-46**
    # Scale sin/cos dari [-1, 1] ke [0, 1]
    bulan_rad = (bulan - 1) * (2 * np.pi / 12)
    features[45] = (np.sin(bulan_rad) + 1) / 2  # [-1,1] -> [0,1]
    features[46] = (np.cos(bulan_rad) + 1) / 2  # [-1,1] -> [0,1]
    
    return features.reshape(1, -1)

def inverse_scale_production(production_scaled):
    """Inverse scale output produksi dari [0,1] ke real values (ton)"""
    min_prod, max_prod = FEATURE_RANGES['Produksi']
    production_real = production_scaled * (max_prod - min_prod) + min_prod
    return production_real

def load_pretrained_model(model_filename="model_final_padi.save"):
    """Load model SVR yang sudah dilatih
    
    Args:
        model_filename (str): Nama file model yang akan di-load.
        Default: 'model_final_padi.save'
    
    Returns:
        Model SVR atau None jika file tidak ditemukan/error
    """
    model_path = model_filename
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    else:
        return None

# ==================== HALAMAN ====================

def home_page():
    """Halaman beranda dengan gambaran umum sistem"""
    custom_header("🌾 SI-PADI JATIM", "Sistem Prediksi Produksi Padi Berbasis Machine Learning")
    
    # Hero Section - Simple Text
    st.markdown("""
    <div style="margin: 30px 0; line-height: 1.8;">
        <h2 style="color: #1b5e20; margin-bottom: 20px; font-size: 2rem; font-weight: 800; letter-spacing: 0.5px;">
            🌾 Mendukung Ketahanan Pangan Melalui Prediksi yang Akurat
        </h2>
        <p style="color: #2c3e50; font-size: 1.05rem; line-height: 1.9; font-weight: 500; margin: 0;">
            <span style="color: #1b5e20; font-weight: 700;">SI-PADI JATIM</span> membantu memperkirakan produksi padi di Jawa Timur dengan memanfaatkan 
            <span style="color: #2e7d32; font-weight: 600;">data agroklimat</span> dan 
            <span style="color: #2e7d32; font-weight: 600;">teknologi machine learning</span>. Platform ini dirancang untuk mendukung 
            <span style="color: #1b5e20; font-weight: 700;">petani, pengambil kebijakan, dan stakeholder</span> dalam membuat 
            <span style="color: #388e3c; font-weight: 700;">keputusan berbasis data yang akurat dan tepat waktu</span>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Section: Benefits
    section_header("💡 Manfaat Sistem", level=3)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                    padding: 28px; border-radius: 14px; border-left: 6px solid #2e7d32; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%;">
            <h4 style="color: #1b5e20; margin-top: 0; font-size: 1.1rem;">👨‍🌾 Untuk Petani</h4>
            <ul style="color: #2e7d32; font-size: 0.95rem; line-height: 1.8;">
                <li><strong>Prediksi panen akurat</strong> untuk perencanaan lebih baik</li>
                <li><strong>Strategi tanam optimal</strong> berdasarkan data</li>
                <li><strong>Manajemen risiko</strong> yang lebih efektif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%); 
                    padding: 28px; border-radius: 14px; border-left: 6px solid #fbc02d; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%;">
            <h4 style="color: #f57f17; margin-top: 0; font-size: 1.1rem;">🏛️ Untuk Policy Maker</h4>
            <ul style="color: #e65100; font-size: 0.95rem; line-height: 1.8;">
                <li><strong>Perencanaan berbasis data</strong> untuk kebijakan tepat</li>
                <li><strong>Alokasi anggaran efektif</strong> sesuai kebutuhan</li>
                <li><strong>Monitoring prediksi produksi padi di setiap wilayah</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%); 
                    padding: 28px; border-radius: 14px; border-left: 6px solid #1b5e20; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08); height: 100%;">
            <h4 style="color: #1b5e20; margin-top: 0; font-size: 1.1rem;">📊 Untuk Stakeholder</h4>
            <ul style="color: #2e7d32; font-size: 0.95rem; line-height: 1.8;">
                <li><strong>Prediksi kebutuhan pasar</strong> yang akurat</li>
                <li><strong>Supply chain planning</strong> yang optimal</li>
                <li><strong>Analisis tren mendalam</strong> untuk strategis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Section: Quick Start
    section_header("🚀 Mulai Menggunakan dalam 4 Langkah", level=3)
    
    steps = [
        ("  Data Masukan", " Input manual data iklim & lahan produksi padi"),
        ("⚙️ Proses", "Sistem mengoptimasi model SVR-ANOVA-RBF menggunakan Particle Swarm Optimization secara otomatis"),
        ("📊 Hasil", "Dapatkan prediksi produksi padi akurat dan hasil prediksi dapar didownload dalam format csv"),
        ("🔍 Analisis", "Buat keputusan strategis berdasarkan analisis tren dan insight data wawasan"),
    ]
    
    cols = st.columns(4)
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; 
                        background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
                        border-radius: 12px; border-top: 5px solid #2e7d32;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                        transition: all 0.3s ease;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;">
                    {title.split()[0]}
                </div>
                <h4 style="color: #1b5e20; margin: 12px 0 8px 0; font-size: 1rem; font-weight: bold;">
                    {title.split(' ', 1)[1] if ' ' in title else ''}
                </h4>
                <p style="color: #555; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                    {desc}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Section: Key Features
    section_header("✨ Fitur Unggulan", level=3)
    
    features = [
        ("Visualisasi Dashor", "Grafik interaktif, dashboard analytics, dan analisis mendalam untuk setiap kabupaten/kota"),
        ("Prediksi Akurat", "Model machine learning SVR-ANOVA-RBF dengan optimasi PSO untuk prediksi presisi tinggi"),
        ("Unggah CSV", "Fitur upload file CSV dan validasi data otomatis dengan deteksi kolom fleksibel"),
        ("Analisis Tren", "Analisis tren temporal per tahun, analisis wilayah regional, dan statistik komprehensif"),
    ]
    
    col1, col2 = st.columns(2)
    for idx, (feature, desc) in enumerate(features):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(f"""
            <div style="background: white; border: 2px solid #e0e0e0; padding: 20px; 
                        border-radius: 10px; margin: 10px 0;
                        transition: all 0.3s ease;
                        border-left: 5px solid #2e7d32;">
                <div style="color: #1b5e20; font-size: 1.3rem; font-weight: bold; margin-bottom: 8px;">
                    {feature}
                </div>
                <div style="color: #666; font-size: 0.95rem;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

def input_data_page():
    """Halaman input data manual dan prediksi dengan model SVR"""
    custom_header("🎯 Prediksi Cepat", "Prediksi produksi padi menggunakan model SVR yang sudah dilatih")
    
    page_info_box("""
**Tentang Menu Ini**

Sistem menyediakan 3 varian. SVR - ANOVA RBF (Model Terbaik) menghitung kemiripan pada tiap dimensi fitur 
secara terpisah sehingga lebih sensitif terhadap pola kompleks. SVR - Standar RBF menghitung kemiripan fitur secara global, 
disediakan sebagai model pembanding. 10-Fold CV ANOVA RBF adalah model hasil validasi terbaik dari 10-Fold Cross-Validation.

Menu ini memprediksi **produksi padi (ton)** dan **produktivitas (ton/ha)** tanpa melatih ulang model —
langsung memakai model tersimpan.

**Fitur:**
- **Prediksi Cepat** — input satu baris, hasil instan.
- **Batch Input** — kumpulkan banyak baris, prediksi sekaligus + ringkasan statistik.

**Alur data:**
1. Anda mengisi variabel iklim & lahan per kabupaten/bulan.
2. Data dinormalisasi MinMax [0,1], kabupaten di-*one-hot* (38 wilayah), bulan di-*sin-cos* encoding → 47 fitur.
3. Model SVR menghasilkan produksi (skala [0,1]) lalu dikonversi ke ton.

**Parameter input** (bukan hyperparameter training):
Kabupaten/Kota, Tahun, Bulan, Luas Panen (ha), Curah Hujan (mm), Kelembapan (%),
Suhu (°C), Kecepatan Angin (m/s), Sinar Matahari (jam/hari).
Parameter C, γ, ε *tidak* diatur di sini — sudah tertanam dalam model terlatih.
""")
    
    tab1, tab2 = st.tabs(["🎯 Prediksi Cepat", "📋 Batch Input"])
    
    # Tab 1: Prediksi Manual Cepat dengan Model
    with tab1:
        st.markdown("### 🔮 Prediksi Produksi Padi dengan Model SVR")
        
        # --- TAMBAHAN KODE: Pilihan Model ---
        st.markdown("#### 🤖 Pilih Arsitektur Model")
        pilihan_model_tab1 = st.radio(
            "Pilih model SVR yang akan digunakan:",
            options=[
                "SVR - ANOVA RBF (Model Terbaik)",
                "SVR - Standar RBF",
                "SVR - 10-Fold CV ANOVA RBF"
            ],
            horizontal=True,
            key="radio_model_tab1"
        )

        # Tentukan file model berdasarkan pilihan user dan tampilkan keterangan
        if pilihan_model_tab1 == "SVR - ANOVA RBF (Split Data)":
            file_model_aktif = "model_final_padi.save"
            st.info("💡 **Keterangan Model:** Model ini dilatih menggunakan kernel ANOVA RBF dengan pembagian dataset Rasio 90:10. Parameter optimal PSO yang didapat: Partikel = 30, Iterasi = 100. Parameter SVR: C = 1,000, Epsilon = 0,003367, Gamma = 282,487.")
        elif pilihan_model_tab1 == "SVR - Standar RBF (Split Data)":
            file_model_aktif = "model_svr_rbf.save"
            st.info("💡 **Keterangan Model:** Model ini dilatih menggunakan kernel standar RBF dengan pembagian dataset Rasio 90:10. Parameter optimal PSO yang didapat: Partikel = 100, Iterasi = 100. Parameter SVR: C = 242,449, Epsilon = 0,001518, Gamma = 0,0232.")
        else:  # "SVR - 10-Fold CV ANOVA RBF (Model Terbaik)"
            file_model_aktif = "model_svr_cv.save"
            st.info("💡 **Keterangan Model:** Model ini divalidasi menggunakan 10-Fold Cross-Validation dengan kernel ANOVA RBF pada rasio data dasar 90:10. Model ini merupakan hasil evaluasi terbaik yang berada di **Fold 4**. Parameter optimal PSO: Partikel = 30, Iterasi = 50. Parameter SVR: C = 50,157, Epsilon = 0,000001, Gamma = 166,850.")

        # Load model
        model = load_pretrained_model(file_model_aktif)
        if model is None:
            st.error(f"❌ Model tidak ditemukan: {file_model_aktif}")
            st.stop()
        
        with st.form("prediction_form", border=True):
            # Row 1: Data Dasar
            st.markdown("#### 📊 Data Dasar")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                tahun = st.number_input("📅 Tahun", min_value=2020, max_value=2030, value=2024)
            with col2:
                bulan = st.selectbox("📅 Bulan", 
                    list(range(1, 13)), format_func=lambda x: ["Januari", "Februari", "Maret", "April", "Mei", "Juni", 
                                                                "Juli", "Agustus", "September", "Oktober", "November", "Desember"][x-1])
            with col3:
                kabupaten = st.selectbox("🏛️ Kabupaten/Kota", 
                    ['Kabupaten Bangkalan', 'Kabupaten Banyuwangi', 'Kabupaten Blitar', 'Kabupaten Bojonegoro',
                    'Kabupaten Bondowoso', 'Kabupaten Gresik', 'Kabupaten Jember', 'Kabupaten Jombang',
                    'Kabupaten Kediri', 'Kabupaten Lamongan', 'Kabupaten Lumajang', 'Kabupaten Madiun',
                    'Kabupaten Magetan', 'Kabupaten Malang', 'Kabupaten Mojokerto', 'Kabupaten Nganjuk',
                    'Kabupaten Ngawi', 'Kabupaten Pacitan', 'Kabupaten Pamekasan', 'Kabupaten Pasuruan',
                    'Kabupaten Ponorogo', 'Kabupaten Probolinggo', 'Kabupaten Sampang', 'Kabupaten Sidoarjo',
                    'Kabupaten Situbondo', 'Kabupaten Sumenep', 'Kabupaten Trenggalek', 'Kabupaten Tuban',
                    'Kabupaten Tulungagung', 'Kota Batu', 'Kota Blitar', 'Kota Kediri',
                    'Kota Madiun', 'Kota Malang', 'Kota Mojokerto', 'Kota Pasuruan',
                    'Kota Probolinggo', 'Kota Surabaya'],
                    index=0)
            with col4:
                luas_panen = st.number_input("🌾 Luas Panen (ha)", min_value=0.1, max_value=100000.0, value=1000.0, step=10.0)
            
            # Row 2: Data Iklim
            st.markdown("#### 🌦️ Data Iklim")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                curah_hujan = st.number_input("🌧️ Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=150.0, step=5.0)
            with col2:
                kelembapan = st.number_input("💧 Kelembapan (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
            with col3:
                suhu = st.number_input("🌡️ Suhu Rata-rata (°C)", min_value=15.0, max_value=40.0, value=27.0, step=0.5)
            with col4:
                kecepatan_angin = st.number_input("💨 Kecepatan Angin (m/s)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
            
            # Row 3: Sinar Matahari
            st.markdown("#### ☀️ Radiasi Matahari")
            sinar_matahari = st.slider("☀️ Sinar Matahari (jam/hari)", min_value=0.0, max_value=12.0, value=7.0, step=0.5)
            
            # Submit button - centered
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                predicted = st.form_submit_button("🔮 Prediksi Produksi Padi", use_container_width=True)
            
            if predicted:
                try:
                    # Siapkan input sesuai format model (dengan scaling)
                    X_input = prepare_input_for_model(
                        luas_panen, curah_hujan, kelembapan, suhu, 
                        kecepatan_angin, sinar_matahari, tahun, kabupaten, bulan
                    )
                    
                    # Prediksi (output dalam skala [0,1])
                    produksi_scaled = model.predict(X_input)[0]
                    
                    # Inverse scale output ke real production values (ton)
                    produksi_prediksi = inverse_scale_production(produksi_scaled)
                    
                    # Tampilkan hasil
                    st.markdown("---")
                    st.markdown("### 📊 Hasil Prediksi")
                    
                    # Hasil dalam bentuk cards
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #2e7d32, #388e3c); 
                                    border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        <div style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #ffffff !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">📦 Produksi Prediksi</div>
                        <p style="margin: 15px 0 0 0; font-size: 2.8rem; font-weight: 900; color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{id_format(produksi_prediksi, 0)}</p>
                        <p style="margin: 5px 0 0 0; font-size: 1rem; font-weight: 600; color: #ffffff !important;">ton</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        produktivitas = produksi_prediksi / luas_panen if luas_panen > 0 else 0
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #fbc02d, #f9a825); 
                                    border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        <div style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #000000 !important; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">📊 Produktivitas</div>
                        <p style="margin: 15px 0 0 0; font-size: 2.8rem; font-weight: 900; color: #000000 !important; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">{id_format(produktivitas, 2)}</p>
                        <p style="margin: 5px 0 0 0; font-size: 1rem; font-weight: 600; color: #000000 !important;">ton/ha</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #1976d2, #1565c0); 
                                    border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        <div style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #ffffff !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">🌾 Luas Panen</div>
                        <p style="margin: 15px 0 0 0; font-size: 2.8rem; font-weight: 900; color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{id_format(luas_panen, 0)}</p>
                        <p style="margin: 5px 0 0 0; font-size: 1rem; font-weight: 600; color: #ffffff !important;">ha</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detail informasi
                    st.markdown("---")
                    st.markdown("#### 📋 Detail Prediksi")
                    
                    detail_cols = st.columns(3)
                    with detail_cols[0]:
                        st.write(f"**Kabupaten/Kota**: {kabupaten}")
                        st.write(f"**Tahun**: {tahun}")
                        st.write(f"**Bulan**: {['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'][bulan-1]}")
                    
                    with detail_cols[1]:
                        st.write(f"**Curah Hujan**: {id_format(curah_hujan, 1)} mm")
                        st.write(f"**Kelembapan**: {id_format(kelembapan, 1)} %")
                        st.write(f"**Suhu**: {id_format(suhu, 1)} °C")
                    
                    with detail_cols[2]:
                        st.write(f"**Kecepatan Angin**: {id_format(kecepatan_angin, 1)} m/s")
                        st.write(f"**Sinar Matahari**: {id_format(sinar_matahari, 1)} jam/hari")
                    
                    # Save to session state untuk referensi
                    st.session_state.last_prediction = {
                        'kabupaten': kabupaten,
                        'tahun': tahun,
                        'bulan': bulan,
                        'luas_panen': luas_panen,
                        'curah_hujan': curah_hujan,
                        'kelembapan': kelembapan,
                        'suhu': suhu,
                        'kecepatan_angin': kecepatan_angin,
                        'sinar_matahari': sinar_matahari,
                        'produksi_prediksi': produksi_prediksi
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error prediksi: {str(e)}")
    
    # Tab 2: Batch Prediksi dengan Model SVR
    with tab2:
        st.markdown("### 🔮 Prediksi Batch Produksi Padi")
        
        st.info("""
        📌 **Fitur Batch Prediksi:**  
        1️⃣ Masukkan data produksi padi satu per satu dengan klik "➕ Tambah Data"  
        2️⃣ Lihat tabel data yang sudah dikumpulkan  
        3️⃣ Klik "🔮 Prediksi Semua" untuk prediksi menggunakan model SVR  
        4️⃣ Hasil prediksi akan ditampilkan di tabel baru dengan statistik lengkap
        """)
        
        # --- TAMBAHAN KODE: Pilihan Model ---
        st.markdown("#### 🤖 Pilih Arsitektur Model")
        pilihan_model_tab2 = st.radio(
            "Pilih model SVR yang akan digunakan:",
            options=[
                "SVR - ANOVA RBF (Split Data)",
                "SVR - Standar RBF (Split Data)",
                "SVR - 10-Fold CV ANOVA RBF (Model Terbaik)"
            ],
            horizontal=True,
            key="radio_model_tab2"
        )

        # Tentukan file model berdasarkan pilihan user dan tampilkan keterangan
        if pilihan_model_tab2 == "SVR - ANOVA RBF (Split Data)":
            file_model_batch = "model_final_padi.save"
            st.info("💡 **Keterangan Model:** Model ini dilatih menggunakan kernel ANOVA RBF dengan pembagian dataset Rasio 90:10. Parameter optimal PSO yang didapat: Partikel = 30, Iterasi = 100. Parameter SVR: C = 10.000, Epsilon = 0,003367, Gamma = 282,487.")
        elif pilihan_model_tab2 == "SVR - Standar RBF (Split Data)":
            file_model_batch = "model_svr_rbf.save"
            st.info("💡 **Keterangan Model:** Model ini dilatih menggunakan kernel standar RBF dengan pembagian dataset Rasio 90:10. Parameter optimal PSO yang didapat: Partikel = 100, Iterasi = 100. Parameter SVR: C = 242,449, Epsilon = 0,001518, Gamma = 0,0232.")
        else:  # "SVR - 10-Fold CV ANOVA RBF (Model Terbaik)"
            file_model_batch = "model_svr_cv.save"
            st.info("💡 **Keterangan Model:** Model ini divalidasi menggunakan 10-Fold Cross-Validation dengan kernel ANOVA RBF pada rasio data dasar 90:10. Model ini merupakan hasil evaluasi terbaik yang berada di **Fold 4**. Parameter optimal PSO: Partikel = 30, Iterasi = 50. Parameter SVR: C = 50,157, Epsilon = 0,000001, Gamma = 166,850.")

        # Load model SVR sesuai pilihan user
        model_svr = load_pretrained_model(file_model_batch)
        if model_svr is None:
            st.error(f"❌ Model SVR tidak ditemukan: {file_model_batch}")
            st.info("📌 Silakan pastikan file model ada di direktori utama.")
        else:
            # ===== STEP 1: FORM INPUT DATA =====
            st.markdown("#### Step 1: Input Data Satu Baris")
            with st.form("batch_add_form", border=True):
                st.markdown("**📊 Data Dasar**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    tahun_batch = st.number_input("📅 Tahun", min_value=2020, max_value=2030, value=2024, key="batch_tahun_add")
                with col2:
                    bulan_batch = st.selectbox("📅 Bulan", 
                        list(range(1, 13)), format_func=lambda x: ["Januari", "Februari", "Maret", "April", "Mei", "Juni", 
                                                                    "Juli", "Agustus", "September", "Oktober", "November", "Desember"][x-1],
                        key="batch_bulan_add")
                with col3:
                    kabupaten_batch = st.selectbox("🏛️ Kabupaten/Kota", 
                        ['Kabupaten Bangkalan', 'Kabupaten Banyuwangi', 'Kabupaten Blitar', 'Kabupaten Bojonegoro',
                        'Kabupaten Bondowoso', 'Kabupaten Gresik', 'Kabupaten Jember', 'Kabupaten Jombang',
                        'Kabupaten Kediri', 'Kabupaten Lamongan', 'Kabupaten Lumajang', 'Kabupaten Madiun',
                        'Kabupaten Magetan', 'Kabupaten Malang', 'Kabupaten Mojokerto', 'Kabupaten Nganjuk',
                        'Kabupaten Ngawi', 'Kabupaten Pacitan', 'Kabupaten Pamekasan', 'Kabupaten Pasuruan',
                        'Kabupaten Ponorogo', 'Kabupaten Probolinggo', 'Kabupaten Sampang', 'Kabupaten Sidoarjo',
                        'Kabupaten Situbondo', 'Kabupaten Sumenep', 'Kabupaten Trenggalek', 'Kabupaten Tuban',
                        'Kabupaten Tulungagung', 'Kota Batu', 'Kota Blitar', 'Kota Kediri',
                        'Kota Madiun', 'Kota Malang', 'Kota Mojokerto', 'Kota Pasuruan',
                        'Kota Probolinggo', 'Kota Surabaya'],
                        index=0, key="batch_kab_add")
                with col4:
                    luas_panen_batch = st.number_input("🌾 Luas Panen (ha)", min_value=0.1, max_value=100000.0, value=1000.0, step=10.0, key="batch_luas_add")
                
                st.markdown("**🌦️ Data Iklim**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    curah_hujan_batch = st.number_input("🌧️ Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=150.0, step=5.0, key="batch_hujan_add")
                with col2:
                    kelembapan_batch = st.number_input("💧 Kelembapan (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0, key="batch_kelembapan_add")
                with col3:
                    suhu_batch = st.number_input("🌡️ Suhu Rata-rata (°C)", min_value=15.0, max_value=40.0, value=27.0, step=0.5, key="batch_suhu_add")
                with col4:
                    kecepatan_angin_batch = st.number_input("💨 Kecepatan Angin (m/s)", min_value=0.0, max_value=20.0, value=3.0, step=0.1, key="batch_angin_add")
                
                st.markdown("**☀️ Radiasi Matahari**")
                sinar_matahari_batch = st.slider("☀️ Sinar Matahari (jam/hari)", min_value=0.0, max_value=12.0, value=7.0, step=0.5, key="batch_sinar_add")
                
                col_left, col_center, col_right = st.columns([1, 2, 1])
                with col_center:
                    add_data_btn = st.form_submit_button("➕ Tambah Data", use_container_width=True)
                
                if add_data_btn:
                    if "batch_data" not in st.session_state:
                        st.session_state.batch_data = []
                    
                    data_row = {
                        "Tahun": tahun_batch,
                        "Bulan": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"][bulan_batch-1],
                        "Kabupaten": kabupaten_batch,
                        "Luas_Panen": luas_panen_batch,
                        "Curah_Hujan": curah_hujan_batch,
                        "Kelembapan": kelembapan_batch,
                        "Suhu": suhu_batch,
                        "Kecepatan_Angin": kecepatan_angin_batch,
                        "Sinar_Matahari": sinar_matahari_batch
                    }
                    
                    st.session_state.batch_data.append(data_row)
                    st.success(f"✅ Data ditambahkan!")
                    st.rerun()
            
            # ===== STEP 2: TAMPILKAN TABEL DATA YANG DIKUMPULKAN =====
            if "batch_data" in st.session_state and len(st.session_state.batch_data) > 0:
                st.markdown("---")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                            padding: 16px; border-radius: 12px; margin: 16px 0;
                            border-left: 4px solid #2e7d32;">
                    <p style="margin: 0; color: #1b5e20; font-weight: 600;">
                        Step 2: Data yang Sudah Dikumpulkan ({len(st.session_state.batch_data)} baris)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                df_data = pd.DataFrame(st.session_state.batch_data)
                
                # Format untuk display
                df_display = df_data.copy()
                df_display['Luas_Panen'] = df_display['Luas_Panen'].apply(lambda x: f"{id_format(x, 0)} ha")
                df_display['Curah_Hujan'] = df_display['Curah_Hujan'].apply(lambda x: f"{id_format(x, 1)} mm")
                df_display['Kelembapan'] = df_display['Kelembapan'].apply(lambda x: f"{id_format(x, 1)} %")
                df_display['Suhu'] = df_display['Suhu'].apply(lambda x: f"{id_format(x, 1)} °C")
                df_display['Kecepatan_Angin'] = df_display['Kecepatan_Angin'].apply(lambda x: f"{id_format(x, 1)} m/s")
                df_display['Sinar_Matahari'] = df_display['Sinar_Matahari'].apply(lambda x: f"{id_format(x, 1)} jam")
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
                
                # Buttons untuk prediksi dan clear
                col_predict, col_clear = st.columns(2)
                with col_predict:
                    if st.button("🔮 Prediksi Semua Data", use_container_width=True, key="predict_all_btn"):
                        # Prediksi semua data yang dikumpulkan
                        st.session_state.batch_results = []
                        
                        with st.spinner("🔄 Sedang melakukan prediksi untuk semua data..."):
                            try:
                                for idx, row in df_data.iterrows():
                                    X_input = prepare_input_for_model(
                                        row['Luas_Panen'], row['Curah_Hujan'], row['Kelembapan'], row['Suhu'],
                                        row['Kecepatan_Angin'], row['Sinar_Matahari'], row['Tahun'], 
                                        row['Kabupaten'], ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                                                        'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'].index(row['Bulan']) + 1
                                    )
                                    
                                    produksi_scaled = model_svr.predict(X_input)[0]
                                    produksi_prediksi = inverse_scale_production(produksi_scaled)
                                    
                                    result_row = row.copy()
                                    result_row['Produksi_Prediksi_Ton'] = produksi_prediksi
                                    st.session_state.batch_results.append(result_row)
                                
                                st.success(f"✅ Prediksi berhasil untuk {len(st.session_state.batch_results)} data!")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error prediksi: {str(e)}")
                
                with col_clear:
                    if st.button("🗑️ Hapus Semua Data", use_container_width=True, key="clear_data_btn"):
                        st.session_state.batch_data = []
                        if "batch_results" in st.session_state:
                            st.session_state.batch_results = []
                        st.rerun()
            
            # ===== STEP 3: TAMPILKAN HASIL PREDIKSI =====
            if "batch_results" in st.session_state and len(st.session_state.batch_results) > 0:
                st.markdown("---")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%); 
                            padding: 16px; border-radius: 12px; margin: 16px 0;
                            border-left: 4px solid #fbc02d;">
                    <p style="margin: 0; color: #f57f17; font-weight: 600;">
                        Step 3: Hasil Prediksi ({len(st.session_state.batch_results)} baris diprediksi)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                df_results = pd.DataFrame(st.session_state.batch_results)
                
                # Format untuk display hasil
                df_result_display = df_results.copy()
                df_result_display['Luas_Panen'] = df_result_display['Luas_Panen'].apply(lambda x: f"{id_format(x, 0)} ha")
                df_result_display['Curah_Hujan'] = df_result_display['Curah_Hujan'].apply(lambda x: f"{id_format(x, 1)} mm")
                df_result_display['Kelembapan'] = df_result_display['Kelembapan'].apply(lambda x: f"{id_format(x, 1)} %")
                df_result_display['Suhu'] = df_result_display['Suhu'].apply(lambda x: f"{id_format(x, 1)} °C")
                df_result_display['Kecepatan_Angin'] = df_result_display['Kecepatan_Angin'].apply(lambda x: f"{id_format(x, 1)} m/s")
                df_result_display['Sinar_Matahari'] = df_result_display['Sinar_Matahari'].apply(lambda x: f"{id_format(x, 1)} jam")
                df_result_display['Produksi_Prediksi_Ton'] = df_result_display['Produksi_Prediksi_Ton'].apply(lambda x: f"{id_format(x, 0)} ton")
                
                st.dataframe(df_result_display, use_container_width=True, hide_index=True)
                
                # Statistics
                st.markdown("#### 📈 Statistik Hasil Prediksi")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    total_produksi = df_results['Produksi_Prediksi_Ton'].sum()
                    st.metric("📦 Total Produksi", f"{id_format(total_produksi, 0)} ton")
                
                with col_stat2:
                    rata_rata_produksi = df_results['Produksi_Prediksi_Ton'].mean()
                    st.metric("📊 Rata-rata", f"{id_format(rata_rata_produksi, 0)} ton")
                
                with col_stat3:
                    max_produksi = df_results['Produksi_Prediksi_Ton'].max()
                    st.metric("📈 Maksimal", f"{id_format(max_produksi, 0)} ton")
                
                with col_stat4:
                    min_produksi = df_results['Produksi_Prediksi_Ton'].min()
                    st.metric("📉 Minimal", f"{id_format(min_produksi, 0)} ton")
                
                # Action buttons
                st.markdown("---")
                col_export, col_clear_result = st.columns(2)
                
                with col_export:
                    csv_data = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV Hasil",
                        data=csv_data,
                        file_name=f"batch_prediksi_hasil_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_clear_result:
                    if st.button("🔄 Prediksi Ulang", use_container_width=True, key="repredict_btn"):
                        st.session_state.batch_results = []
                        st.rerun()

def visualisasi_page():
    """Halaman visualisasi data dengan analytics profesional"""
    custom_header("📊 Visualisasi Data & Analytics", "Eksplorasi pola dan tren dalam data pertanian Anda")
    
    # Initialize session state untuk data visualisasi
    if "viz_df" not in st.session_state:
        st.session_state.viz_df = None
    
    # File Upload Section
    st.markdown("#### 📄 Unggah File Data (CSV atau Excel)")
    st.markdown("Unggah file CSV atau Excel (.xlsx, .xls) Anda untuk melakukan analisis dan visualisasi data padi")
    
    # Tampilkan informasi tentang format file yang diterima
    with st.expander("📋 Format File yang Diterima", expanded=False):
        st.markdown("""
        **File CSV atau Excel harus memiliki kolom-kolom berikut:**
        
        | Kolom | Tipe | Deskripsi | Contoh |
        |-------|------|-----------|--------|
        | **Tahun** | Integer | Tahun pengumpulan data | 2023 |
        | **Blok/Kabupaten** | Text | Nama kabupaten atau wilayah | Kabupaten Malang |
        | **Luas Panen** | Numeric | Luas area panen dalam hektar | 15000 |
        | **Curah Hujan** | Numeric | Curah hujan dalam mm/bulan | 250 |
        | **Kelembapan** | Numeric | Kelembapan udara dalam % | 75 |
        | **Suhu** | Numeric | Suhu rata-rata dalam °C | 28.5 |
        | **Kecepatan Angin** | Numeric | Kecepatan angin dalam m/s | 3.2 |
        | **Sinar Matahari** | Numeric | Durasi sinar matahari jam/hari | 8.5 |
        | **Produksi** | Numeric | Produksi padi dalam ton | 45000 |
        
        **Contoh File CSV:**
        ```
        Tahun,Blok/Kabupaten,Luas Panen,Curah Hujan,Kelembapan,Suhu,Kecepatan Angin,Sinar Matahari,Produksi
        2023,Kabupaten Malang,15000,250,75,28.5,3.2,8.5,45000
        2023,Kabupaten Surabaya,12000,280,78,29,2.8,8,42000
        2023,Kabupaten Sidoarjo,18000,200,72,27,4,9,50000
        2022,Kabupaten Malang,14000,230,73,28,3.5,8.2,43000
        2022,Kabupaten Surabaya,11000,260,76,28.5,3,8.5,40000
        ```
        
        **Format Excel:**
        - File Excel (.xlsx atau .xls) harus memiliki struktur yang sama seperti CSV
        - Gunakan sheet pertama sebagai sumber data
        - Baris pertama harus berisi nama kolom (header)
        - Gunakan format angka standar (desimal dengan titik)
        
        ⚠️ **Catatan Penting:**
        - Harus dalam format CSV (comma-separated values) atau Excel (.xlsx, .xls)
        - Baris pertama harus berisi nama kolom
        - Gunakan titik (.) untuk desimal, bukan koma
        - Kolom dapat menggunakan nama alternatif: Tahun/periode, Kabupaten/Wilayah/Kota, Produksi/production
        """)
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )
    
    # Proses file yang di-upload
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_extension == 'xls':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error("❌ Format file tidak didukung. Gunakan: CSV, XLSX, atau XLS")
                return
            
            st.session_state.viz_df = df
            st.success(f"✅ File berhasil diunggah! ({file_extension.upper()})")
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")
            st.info("💡 Pastikan file dalam format yang benar dan openpyxl sudah terinstall")
            return
    elif st.session_state.viz_df is not None:
        df = st.session_state.viz_df
    elif "df_for_prediction" in st.session_state:
        df = st.session_state.df_for_prediction
    else:
        st.info("📌 Silakan unggah file CSV atau Excel untuk memulai analisis.")
        return
    
    if df is None or len(df) == 0:
        st.error("Data tidak valid atau kosong")
        return
    
    # File Statistics Card
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("📊 Baris", len(df))
    with col_stat2:
        st.metric("📋 Kolom", len(df.columns))
    with col_stat3:
        if 'Produksi' in df.columns:
            st.metric("🌾 Produksi", f"{id_format(df['Produksi'].sum(), 0)} ton")
        else:
            st.metric("⚠️ Produksi", "-")
    with col_stat4:
        if 'Luas Panen' in df.columns:
            st.metric("🌾 Luas", f"{id_format(df['Luas Panen'].sum(), 0)} ha")
        else:
            st.metric("⚠️ Luas", "-")
    
    st.markdown("---")
    
    st.markdown("---")
    
    # Tabs untuk berbagai jenis visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Tren Temporal", "🏛️ Analisis Wilayah", "📊 Statistik", "🔍 Distribusi", "💡 Insight Analytics"])
    
    with tab1:
        st.markdown("#### 📈 Analisis Tren Temporal Per Tahun & Kabupaten")
        
        # Cek kolom yang diperlukan
        tahun_col = None
        kabupaten_col = None
        produksi_col = None
        
        for col_name in ['Tahun', 'tahun', 'Periode', 'periode', 'Tahun Tanam', 'tahun_tanam']:
            if col_name in df.columns:
                tahun_col = col_name
                break
        
        for col_name in ['Kabupaten/Kota', 'Kabupaten', 'kabupaten', 'Wilayah', 'wilayah', 'Kota', 'kota']:
            if col_name in df.columns:
                kabupaten_col = col_name
                break
        
        for col_name in ['Produksi', 'produksi', 'Production']:
            if col_name in df.columns:
                produksi_col = col_name
                break
        
        if tahun_col and kabupaten_col and produksi_col:
            try:
                df_copy = df.copy()
                
                # Mapping nama bulan Indonesia ke nomor
                bulan_map = {
                    'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4, 'Mei': 5, 'Juni': 6,
                    'Juli': 7, 'Agustus': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
                }
                
                # Parse Periode format "Januari 2018" atau "2018 Januari" atau "201801"
                def parse_periode(periode_str):
                    try:
                        periode_str = str(periode_str).strip()
                        
                        # Try format "Januari 2018" atau "Januari2018"
                        for bulan_name, bulan_num in bulan_map.items():
                            if bulan_name in periode_str:
                                tahun_str = periode_str.replace(bulan_name, '').strip()
                                tahun = int(tahun_str)
                                return tahun, bulan_num
                        
                        # Try format "201801"
                        if len(periode_str) == 6 and periode_str.isdigit():
                            tahun = int(periode_str[:4])
                            bulan = int(periode_str[4:6])
                            return tahun, bulan
                        
                        # Try format "2018-01"
                        if '-' in periode_str:
                            parts = periode_str.split('-')
                            tahun = int(parts[0])
                            bulan = int(parts[1])
                            return tahun, bulan
                    except:
                        return None, None
                    
                    return None, None
                
                # Apply parsing
                df_copy[['tahun_temp', 'bulan_temp']] = df_copy[tahun_col].apply(
                    lambda x: pd.Series(parse_periode(x))
                )
                
                df_valid = df_copy[df_copy['tahun_temp'].notna()].copy()
                
                if len(df_valid) > 0:
                    # Get unique years dan kabupaten
                    tahun_list = sorted(df_valid['tahun_temp'].unique())
                    kabupaten_list = sorted(df_valid[kabupaten_col].unique())
                    
                    # Filters dalam 2 kolom
                    col_filter1, col_filter2 = st.columns(2)
                    
                    with col_filter1:
                        selected_kabupaten = st.selectbox(
                            "🏛️ Pilih Kabupaten/Kota:",
                            kabupaten_list,
                            index=0
                        )
                    
                    with col_filter2:
                        selected_tahun = st.selectbox(
                            "📅 Pilih Tahun:",
                            tahun_list,
                            index=len(tahun_list)-1
                        )
                    
                    # Filter data berdasarkan pilihan
                    df_filtered = df_valid[
                        (df_valid[kabupaten_col] == selected_kabupaten) & 
                        (df_valid['tahun_temp'] == int(selected_tahun))
                    ].copy()
                    
                    if len(df_filtered) > 0:
                        # Group by bulan
                        monthly = df_filtered.groupby('bulan_temp')[produksi_col].sum().sort_index()
                        
                        st.markdown("---")
                        
                        # Informasi ringkas - gunakan columns dan info cards
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center;">
                                <div style="color: #666; font-size: 12px; font-weight: bold;">🏛️ Kabupaten</div>
                                <div style="color: #1b5e20; font-size: 24px; font-weight: bold;">{selected_kabupaten.replace('Kabupaten ', '').replace('Kabupaten/Kota ', '').replace('Kota ', '')}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_info2:
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center;">
                                <div style="color: #666; font-size: 12px; font-weight: bold;">📅 Tahun</div>
                                <div style="color: #1b5e20; font-size: 24px; font-weight: bold;">{int(selected_tahun)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_info3:
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; text-align: center;">
                                <div style="color: #666; font-size: 12px; font-weight: bold;">📦 Total Produksi</div>
                                <div style="color: #1b5e20; font-size: 24px; font-weight: bold;">{id_format(monthly.sum(), 0)} ton</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        # Grafik utama
                        plt.close('all')
                        fig, ax = plt.subplots(figsize=(14, 6))
                        
                        bulan_names = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 
                                    'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
                        
                        # Warna berdasarkan nilai tertinggi
                        colors = ['#2e7d32' if val == monthly.max() else '#388e3c' for val in monthly.values]
                        
                        bars = ax.bar(range(len(monthly)), monthly.values, 
                                    color=colors, alpha=0.85, edgecolor='#1b5e20', linewidth=2)
                        
                        # Value labels pada setiap bar - pastikan update dengan data terbaru
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{id_format(int(height), 0)}', ha='center', va='bottom', 
                                fontweight='bold', fontsize=11, color='#1b5e20')
                        
                        # Styling
                        ax.set_xticks(range(len(monthly)))
                        ax.set_xticklabels([bulan_names[int(m)-1] for m in monthly.index], fontsize=11, fontweight='bold')
                        ax.set_ylabel('Produksi (ton)', fontweight='bold', fontsize=12)
                        ax.set_title(f'Tren Produksi Padi - {selected_kabupaten.replace("Kabupaten/Kota ", "")} Tahun {int(selected_tahun)}', 
                                    fontweight='bold', fontsize=13, pad=15)
                        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
                        ax.set_ylim(bottom=0)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Statistik tambahan
                        st.markdown("---")
                        st.markdown("#### 📊 Analisis Detail")
                        
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            st.metric("📈 Produksi Tertinggi", f"{id_format(monthly.max(), 0)} ton", 
                                    delta=f"Bulan {bulan_names[int(monthly.idxmax())-1]}")
                        with col_stat2:
                            st.metric("📉 Produksi Terendah", f"{id_format(monthly.min(), 0)} ton",
                                    delta=f"Bulan {bulan_names[int(monthly.idxmin())-1]}")
                        with col_stat3:
                            st.metric("📊 Rata-rata", f"{id_format(monthly.mean(), 0)} ton")
                        with col_stat4:
                            selisih = monthly.max() - monthly.min()
                            st.metric("📌 Selisih", f"{id_format(selisih, 0)} ton")
                    else:
                        st.warning(f"⚠️ Tidak ada data untuk {selected_kabupaten} tahun {int(selected_tahun)}")
                else:
                    st.warning("⚠️ Data tidak valid untuk diproses")
                    st.info(f"📌 Contoh format data yang diharapkan: 'Januari 2018' atau '201801'")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
        else:
            st.warning(f"⚠️ Kolom yang diperlukan tidak ditemukan.")
            st.info(f"Kolom tersedia: {', '.join(df.columns.tolist())}")
        
        # Tren bulanan jika ada
        bulan_col = None
        if 'Bulan' in df.columns:
            bulan_col = 'Bulan'
        elif 'bulan' in df.columns:
            bulan_col = 'bulan'
        
        if bulan_col and produksi_col:
            try:
                st.markdown("---")
                st.markdown("#### 🌙 Rataan Produksi Bulanan")
                monthly = df.groupby(bulan_col)[produksi_col].mean()
                
                if len(monthly) > 0:
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        st.metric("🏆 Bulan Tertinggi", f"{monthly.idxmax()}")
                        st.metric("📈 Nilai Tertinggi", f"{id_format(monthly.max(), 0)} ton")
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(12, 5))
                        ax.plot(monthly.index, monthly.values, marker='o', linewidth=2.5, 
                            markersize=10, color='#fbc02d', markerfacecolor='#f9a825', 
                            markeredgewidth=2, markeredgecolor='#f9a825')
                        ax.fill_between(range(len(monthly)), monthly.values, alpha=0.2, color='#fbc02d')
                        ax.set_xlabel('Bulan', fontweight='bold', fontsize=11)
                        ax.set_ylabel('Produksi Rata-rata (ton)', fontweight='bold', fontsize=11)
                        ax.set_title('Tren Musiman Produksi Padi', fontweight='bold', fontsize=12, pad=15)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Analisis bulanan tidak tersedia: {str(e)}")
    
    with tab2:
        st.markdown("#### 🏛️ Analisis Produksi per Wilayah")
        
        # Cek kolom dengan berbagai kemungkinan nama
        kabupaten_col = None
        for col_name in ['Kabupaten/Kota', 'Kabupaten', 'kabupaten', 'Wilayah', 'wilayah', 'Kota', 'kota']:
            if col_name in df.columns:
                kabupaten_col = col_name
                break
        
        produksi_col_tab2 = None
        for col_name in ['Produksi', 'produksi', 'Production']:
            if col_name in df.columns:
                produksi_col_tab2 = col_name
                break
        
        if kabupaten_col and produksi_col_tab2:
            try:
                kabupaten = df.groupby(kabupaten_col)[produksi_col_tab2].sum().sort_values(ascending=False)
                
                if len(kabupaten) > 0:
                    # Metric cards untuk top kabupaten
                    st.markdown("**Top 3 Wilayah/Kabupaten:**")
                    top_3 = kabupaten.head(3)
                    cols_top = st.columns(min(3, len(top_3)))
                    for idx, (kab_name, val) in enumerate(top_3.items()):
                        with cols_top[idx]:
                            display_name = kab_name.replace('Kabupaten ', '').replace('Kota ', '').replace('Kabupaten/Kota ', '')
                            st.metric(f"#{idx+1}", display_name, f"{id_format(val, 0)} ton")
                    
                    st.markdown("---")
                    
                    # Grafik horizontal bar yang lebih baik
                    fig, ax = plt.subplots(figsize=(12, max(6, len(kabupaten)*0.3)))
                    num_items = len(kabupaten)
                    colors = plt.cm.Greens(np.linspace(0.4, 0.9, num_items))
                    
                    bars = ax.barh(range(len(kabupaten)), kabupaten.values, color=colors, 
                                edgecolor='#1b5e20', linewidth=1, alpha=0.85)
                    
                    # Value labels
                    for i, (bar, val) in enumerate(zip(bars, kabupaten.values)):
                        ax.text(val, bar.get_y() + bar.get_height()/2.,
                            f' {id_format(int(val), 0)} ton', ha='left', va='center', 
                            fontweight='bold', fontsize=9)
                    
                    ax.set_yticks(range(len(kabupaten)))
                    cleaned_labels = [name.replace('Kabupaten ', '').replace('Kota ', '').replace('Kabupaten/Kota ', '') 
                                    for name in kabupaten.index]
                    ax.set_yticklabels(cleaned_labels, fontsize=10)
                    ax.set_xlabel('Produksi Total (ton)', fontweight='bold', fontsize=11)
                    ax.set_title('Distribusi Produksi Padi per Wilayah', fontweight='bold', fontsize=12, pad=15)
                    ax.grid(axis='x', alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Data wilayah kosong")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        else:
            st.warning(f"⚠️ Kolom tidak ditemukan. Kolom tersedia: {', '.join(df.columns.tolist())}")
    
    with tab3:
        st.markdown("#### 📊 Statistik Deskriptif Lengkap")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 0:
            try:
                stats_df = df[numeric_cols].describe().T
                
                # Multiselect untuk semua kolom numerik
                col_sel = st.multiselect(
                    "📈 Pilih kolom untuk dianalisis:", 
                    numeric_cols, 
                    default=[numeric_cols[0]] if numeric_cols else []
                )
                
                if col_sel:
                    # Tampilkan tabel statistik
                    stats_display = stats_df.loc[col_sel]
                    st.dataframe(stats_display, use_container_width=True)
                    
                    # Summary cards untuk kolom yang dipilih
                    st.markdown("---")
                    st.markdown("**📋 Ringkasan Kolom yang Dipilih:**")
                    
                    num_cols_selected = len(col_sel)
                    cols_summary = st.columns(min(3, num_cols_selected))
                    
                    for idx, col_name in enumerate(col_sel):
                        col_idx = idx % 3
                        with cols_summary[col_idx]:
                            try:
                                mean_val = df[col_name].mean()
                                std_val = df[col_name].std()
                                min_val = df[col_name].min()
                                max_val = df[col_name].max()
                                
                                st.markdown(f"""
                                <div style="padding: 12px; background: linear-gradient(135deg, #2e7d32, #388e3c); border-radius: 8px; color: white; margin-bottom: 10px;">
                                <h4 style="margin: 0; font-size: 12px;">{col_name}</h4>
                                <p style="margin: 5px 0 2px 0; font-size: 11px;"><b>Rata:</b> {id_format(mean_val, 2)}</p>
                                <p style="margin: 2px 0 2px 0; font-size: 11px;"><b>StdDev:</b> {id_format(std_val, 2)}</p>
                                <p style="margin: 2px 0 2px 0; font-size: 11px;"><b>Min:</b> {id_format(min_val, 2)}</p>
                                <p style="margin: 2px 0 0 0; font-size: 11px;"><b>Max:</b> {id_format(max_val, 2)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            except:
                                st.info(f"ℹ️ {col_name}: Tidak bisa dianalisis")
            except Exception as e:
                st.error(f"❌ Error statistik: {str(e)}")
        else:
            st.info("ℹ️ Tidak ada kolom numerik untuk dianalisis")
    
    with tab4:
        st.markdown("#### 🔍 Distribusi Data")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            try:
                col_dist = st.selectbox("Pilih kolom untuk distribusi:", numeric_cols)
                
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                # Histogram
                axes[0].hist(df[col_dist].dropna(), bins=20, color='#2e7d32', alpha=0.7, edgecolor='#1b5e20', linewidth=1.5)
                axes[0].axvline(df[col_dist].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {id_format(df[col_dist].mean(), 2)}')
                axes[0].axvline(df[col_dist].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {id_format(df[col_dist].median(), 2)}')
                axes[0].set_xlabel(col_dist, fontweight='bold', fontsize=11)
                axes[0].set_ylabel('Frekuensi', fontweight='bold', fontsize=11)
                axes[0].set_title(f'Histogram Distribusi {col_dist}', fontweight='bold', fontsize=12)
                axes[0].legend()
                axes[0].grid(axis='y', alpha=0.3)
                
                # Box plot
                axes[1].boxplot(df[col_dist].dropna(), vert=False, patch_artist=True,
                            boxprops=dict(facecolor='#2e7d32', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))
                axes[1].set_xlabel(col_dist, fontweight='bold', fontsize=11)
                axes[1].set_title(f'Box Plot & Pencilan {col_dist}', fontweight='bold', fontsize=12)
                axes[1].grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"❌ Error distribusi: {str(e)}")
        else:
            st.info("ℹ️ Tidak ada kolom numerik untuk dianalisis")
    
    with tab5:
        st.markdown("### 💡 Insight Analytics Mendalam")
        
        # Correlation Analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            st.markdown("#### 🔗 Analisis Korelasi Antar Variabel")
            
            correlation_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
            
            # Set ticks dan labels
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.columns)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.columns)
            
            # Add values to heatmap
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    text = ax.text(j, i, f'{id_format(correlation_matrix.iloc[i, j], 2)}',
                                ha='center', va='center', color='black', fontweight='bold', fontsize=9)
            
            ax.set_title('Heatmap Korelasi Variabel', fontweight='bold', fontsize=12, pad=15)
            plt.colorbar(im, ax=ax, label='Korelasi')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Year-over-Year Comparison
        st.markdown("---")
        st.markdown("#### 📊 Perbandingan Periode")
        
        # Cari kolom periode/tahun
        periode_col = None
        produksi_col_tab5 = None
        for col_name in ['Tahun', 'tahun', 'Periode', 'periode']:
            if col_name in df.columns:
                periode_col = col_name
                break
        for col_name in ['Produksi', 'produksi']:
            if col_name in df.columns:
                produksi_col_tab5 = col_name
                break
        
        if periode_col and produksi_col_tab5:
            try:
                yearly_stats = df.groupby(periode_col).agg({
                    produksi_col_tab5: ['sum', 'mean', 'count']
                }).round(2)
                
                yearly_stats.columns = ['Total Produksi', 'Rata-rata', 'Jumlah Data']
                
                col_yr1, col_yr2, col_yr3 = st.columns(3)
                with col_yr1:
                    st.markdown(f"""
                    <div style="padding: 15px; background: linear-gradient(135deg, #2e7d32, #388e3c); border-radius: 8px; color: white;">
                    <p style="margin: 0; font-size: 12px; opacity: 0.9;"><b>Total Keseluruhan</b></p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: 900;">{id_format(yearly_stats['Total Produksi'].sum(), 0)}</p>
                    <p style="margin: 2px 0 0 0; font-size: 11px; opacity: 0.8;">ton</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_yr2:
                    best_periode = yearly_stats['Total Produksi'].idxmax()
                    # Handle both numeric dan text periode
                    if isinstance(best_periode, str):
                        best_periode_display = best_periode
                    else:
                        best_periode_display = str(int(best_periode))
                    
                    st.markdown(f"""
                    <div style="padding: 15px; background: linear-gradient(135deg, #1976d2, #1565c0); border-radius: 8px; color: white;">
                    <p style="margin: 0; font-size: 12px; opacity: 0.9;"><b>Periode Terbaik</b></p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: 900;">{best_periode_display}</p>
                    <p style="margin: 2px 0 0 0; font-size: 11px; opacity: 0.8;">ton</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_yr3:
                    growth_rate = ((yearly_stats['Total Produksi'].iloc[-1] - yearly_stats['Total Produksi'].iloc[0]) / yearly_stats['Total Produksi'].iloc[0] * 100) if len(yearly_stats) > 1 else 0
                    growth_color = "#2e7d32" if growth_rate > 0 else "#d32f2f"
                    growth_arrow = "📈" if growth_rate > 0 else "📉"
                    st.markdown(f"""
                    <div style="padding: 15px; background: linear-gradient(135deg, {growth_color}, {growth_color}); border-radius: 8px; color: white;">
                    <p style="margin: 0; font-size: 12px; opacity: 0.9;"><b>Pertumbuhan</b></p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: 900;">{growth_arrow} {id_format(abs(growth_rate), 1)}%</p>
                    <p style="margin: 2px 0 0 0; font-size: 11px; opacity: 0.8;">YoY</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.dataframe(yearly_stats, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Analisis periode tidak tersedia: {str(e)}")
        
        # Top Performers
        st.markdown("---")
        st.markdown("#### 🏆 Top Performers Analytics")
        
        # Cari kolom wilayah dan produksi
        wilayah_col = None
        produksi_col_top = None
        for col_name in ['Kabupaten/Kota', 'Kabupaten', 'kabupaten', 'Wilayah', 'wilayah', 'Kota', 'kota']:
            if col_name in df.columns:
                wilayah_col = col_name
                break
        for col_name in ['Produksi', 'produksi']:
            if col_name in df.columns:
                produksi_col_top = col_name
                break
        
        if wilayah_col and produksi_col_top:
            try:
                top_kabupaten = df.groupby(wilayah_col)[produksi_col_top].sum().nlargest(5)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                colors_gradient = ['#1b5e20', '#2e7d32', '#388e3c', '#43a047', '#66bb6a']
                bars = ax.bar(range(len(top_kabupaten)), top_kabupaten.values, color=colors_gradient, 
                            edgecolor='#1b5e20', linewidth=2, alpha=0.9)
                
                # Value labels on bars
                for i, (bar, val) in enumerate(zip(bars, top_kabupaten.values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(val)} ton', ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                ax.set_xticks(range(len(top_kabupaten)))
                clean_names = [name.replace('Kabupaten ', '').replace('Kota ', '').replace('Kabupaten/Kota ', '') 
                            for name in top_kabupaten.index]
                ax.set_xticklabels(clean_names, rotation=45, ha='right')
                ax.set_ylabel('Total Produksi (ton)', fontweight='bold', fontsize=11)
                ax.set_title(f'Top 5 {wilayah_col} Produksi Tertinggi', fontweight='bold', fontsize=12, pad=15)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Top performers tidak tersedia: {str(e)}")
        else:
            st.info("ℹ️ Kolom wilayah atau produksi tidak ditemukan")
        
        # Data Quality Summary
        st.markdown("---")
        st.markdown("#### 📋 Ringkasan Kualitas Data")
        
        col_dq1, col_dq2, col_dq3, col_dq4 = st.columns(4)
        
        with col_dq1:
            null_count = df.isnull().sum().sum()
            st.metric("❌ Total Missing", null_count)
        
        with col_dq2:
            completion_rate = (1 - null_count / (df.shape[0] * df.shape[1])) * 100 if df.shape[0] * df.shape[1] > 0 else 0
            st.metric("✅ Komplit", f"{id_format(completion_rate, 1)}%")
        
        with col_dq3:
            st.metric("📊 Unique Values", df.nunique().sum())
        
        with col_dq4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("📈 Memory", f"{id_format(memory_mb, 2)} MB")

def tentang_model_page():
    """Halaman penjelasan tentang model dengan desain profesional"""
    custom_header("Model Prediksi", "Arsitektur teknologi machine learning untuk prediksi akurat")
    
    # Section: Model Overview
    st.markdown("""
    **SI-PADI JATIM menggunakan kombinasi tiga komponen teknologi machine learning:**
    
    - **Support Vector Regression (SVR)** - Algoritma regresi yang robust dan akurat
    - **ANOVA Radial Basis Function (ANOVA RBF)** - Kernel non-linear untuk pola kompleks
    - **Particle Swarm Optimization (PSO)** - Optimasi metaheuristik untuk parameter terbaik
    """)
    
    st.markdown("---")
    
    # Tech Stack Cards with columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Support Vector Regression**\n\nAlgoritma regresi robust untuk prediksi nilai kontinyu dengan pola kompleks")
    
    with col2:
        st.info("**Particle Swarm Optimization**\n\nOptimasi metaheuristik untuk mencari parameter SVR terbaik otomatis")
    
    with col3:
        st.info("**ANOVA RBF Kernel**\n\nKernel non-linear untuk menangani pola data kompleks dan tidak linear")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Alur Kerja", "🔧 SVR", "🌐 PSO", "⚪ ANOVA RBF"])
    
    with tab1:
        st.markdown("#### Alur Kerja Prediksi")
        
        steps_workflow = [
            ("1. Input Data", "Data historis produksi padi dengan variabel iklim dan lahan"),
            ("2. Preprocessing", "Validasi data, penanganan missing values, dan normalisasi ke range [0,1]"),
            ("3. Split Dataset", "Pembagian data menjadi training set (90%) dan test set (10%)"),
            ("4. Optimasi Parameter", "PSO mencari kombinasi parameter C, γ, ε yang optimal"),
            ("5. Training Model", "SVR dilatih dengan parameter optimal dari hasil PSO"),
            ("6. Prediksi", "Model menghasilkan prediksi produksi padi"),
            ("7. Evaluasi", "Perhitungan metrik akurasi (MAPE, RMSE, R²)"),
        ]
        
        for step, desc in steps_workflow:
            st.markdown(f"**{step}**  \n{desc}\n")
    
    with tab2:
        st.markdown("#### Support Vector Regression (SVR)")
        st.markdown("""
        SVR adalah algoritma machine learning untuk prediksi regresi yang dikembangkan dari Support Vector Machine (SVM).
        Algoritma ini mencari hyperplane optimal yang meminimalkan error prediksi sambil menjaga margin maksimum.
        """)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("""
**Prinsip Kerja:**
- Margin Maksimum: SVR mencari hyperplane dengan margin terlebar untuk generalisasi lebih baik
- Toleransi Error: Parameter epsilon (ε) mendefinisikan batas toleransi error
- Support Vectors: Hanya data points kritis yang mempengaruhi model
- Kernel Trick: Transformasi data ke dimensi lebih tinggi untuk menangani pola non-linear

**Keunggulan SVR:**
- ✓ Robust terhadap outlier dan noise
- ✓ Bekerja dengan berbagai kernel
- ✓ Memory efficient untuk dataset besar
- ✓ Generalisasi baik pada data baru
            """)
        
        with col2:
            st.markdown("**Parameter SVR Utama**")
            param_data = {
                "Parameter": ["C (Regularisasi)", "Gamma (γ)", "Epsilon (ε)"],
                "Range": ["1 - 1000", "0.00001 - 100", "0.000001 - 0.1"],
                "Penjelasan": [
                    "C tinggi = model kompleks",
                    "γ tinggi = pengaruh lokal",
                    "Margin toleransi error"
                ]
            }
            st.dataframe(param_data, use_container_width=True, hide_index=True)
        
        st.info("💡 PSO akan mengoptimasi ketiga parameter (C, γ, ε) secara otomatis untuk hasil terbaik")
    
    with tab3:
        st.markdown("#### Particle Swarm Optimization (PSO)")
        st.markdown("""
        PSO adalah algoritma optimasi metaheuristik yang terinspirasi dari perilaku kawanan burung dan gerombolan ikan.
        Algoritma ini digunakan untuk mencari kombinasi parameter SVR yang menghasilkan akurasi prediksi tertinggi.
        """)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("""
**Konsep PSO:**
- Partikel: Setiap solusi kandidat adalah satu "partikel" dalam ruang pencarian
- Posisi & Kecepatan: Partikel bergerak dalam ruang parameter (C, γ, ε)
- Memori Pribadi: Setiap partikel mengingat posisi terbaik yang pernah ditemukannya
- Informasi Global: Partikel saling berbagi informasi tentang solusi terbaik global
- Iterasi: Proses berulang hingga konvergensi atau iterasi maksimum tercapai

**Mengapa PSO untuk SVR?**
- ✓ Tidak memerlukan gradien fungsi
- ✓ Dapat menghindari local minima
- ✓ Konvergensi cepat ke solusi optimal
- ✓ Fleksibel untuk berbagai jenis problem
- ✓ Robust pada kondisi berbeda
            """)
        
        with col2:
            st.markdown("**Konfigurasi PSO**")
            pso_data = {
                "Komponen": ["Jumlah Partikel", "Iterasi", "Parameter Kontrol"],
                "Nilai": ["30", "100", "Balanced"],
                "Keterangan": [
                    "Solusi yang dieksplorasi",
                    "Generasi optimasi",
                    "Eksplorasi & eksploitasi"
                ]
            }
            st.dataframe(pso_data, use_container_width=True, hide_index=True)
        
        st.success("🎯 Hasil PSO: Kombinasi parameter SVR (C, γ, ε) yang optimal untuk dataset Anda")
    
    with tab4:
        st.markdown("#### ANOVA RBF (ANOVA Radial Basis Function) Kernel")
        st.markdown("""
        ANOVA RBF adalah fungsi kernel non-linear yang mengubah data input ke dimensi lebih tinggi untuk menangani 
        pola data yang kompleks dan tidak dapat dipisahkan secara linear. ANOVA kernel menambahkan kapabilitas seleksi fitur otomatis.
        """)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.markdown("""
**Formula ANOVA RBF:**

$$K(x_i, x_j) = \\exp\\left(-\\gamma \\cdot ||x_i - x_j||^2\\right)$$

ANOVA RBF mengukur similarity (kemiripan) antara dua data points berdasarkan jarak Euclidean dengan seleksi fitur otomatis.

**Karakteristik ANOVA RBF:**
- ✓ Powerful: Dapat menangani pola sangat kompleks
- ✓ Flexible: Banyak hyperparameter untuk tuning
- ✓ Smooth: Menghasilkan decision boundary yang smooth
- ✓ Universal: Dapat approximate fungsi apapun
- ✓ Popular: Pilihan default kernel di berbagai aplikasi ML
- ✓ Smart: Seleksi fitur otomatis untuk pertidaksamaan data
            """)
        
        with col2:
            st.markdown("**Kontrol Gamma (γ)**")
            gamma_data = {
                "Kondisi": ["γ Tinggi", "γ Rendah"],
                "Karakteristik": [
                    "Bump sempit & tinggi",
                    "Bump lebar & datar"
                ],
                "Dampak": [
                    "Pengaruh lokal > overfitting",
                    "Pengaruh global > underfitting"
                ]
            }
            st.dataframe(gamma_data, use_container_width=True, hide_index=True)
        
        st.warning("⚠️ ANOVA RBF kernel berbasis distance, sehingga data scaling sangat penting. SI-PADI JATIM melakukan standardisasi otomatis untuk semua input data.")

def about_page():
    """Halaman tentang sistem dengan desain profesional"""
    custom_header("SI-PADI JATIM", "Sistem Informasi Prediksi Produksi Padi Jawa Timur")
    
    # Section 1: Tentang Sistem
    st.markdown("""
### Tentang Sistem

SI-PADI JATIM adalah sistem informasi terintegrasi yang dirancang untuk mendukung prediksi produksi padi 
di Jawa Timur. Dengan memanfaatkan teknologi machine learning dan data agroklimat historis, sistem ini 
membantu petani, pengambil kebijakan, dan stakeholder dalam mengambil keputusan yang lebih baik dan strategis.
    """)
    
    st.divider()
    
    # Section 2: Visi & Misi
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
**Visi**

Menjadi sistem prediksi produksi padi terdepan yang mendukung ketahanan pangan Jawa Timur 
melalui inovasi teknologi dan data-driven decision making.
        """)
    
    with col2:
        st.info("""
**Misi**

Menyediakan alat prediksi akurat dan mudah digunakan untuk meningkatkan produktivitas pertanian 
dan mengurangi risiko produksi di tingkat petani dan wilayah.
        """)
    
    st.divider()
    
    # Section 3: Fitur Utama
    st.markdown("### Fitur Utama Sistem")
    
    features_list = [
        ("Input Data Fleksibel", "Unggah file CSV atau input data secara manual dengan interface yang user-friendly."),
        ("Prediksi Akurat", "Menggunakan model SVR-PSO-ANOVA-RBF untuk hasil prediksi presisi tinggi."),
        ("Analisis Tren Mendalam", "Analisis tren per tahun, wilayah, dan statistik lengkap."),
        ("Visualisasi Interaktif", "Dashboard dengan grafik interaktif dan insight analytics."),
    ]
    
    for idx, (title, desc) in enumerate(features_list):
        st.info(f"**{idx + 1}. {title}**\n\n{desc}")
    
    st.divider()
    
    # Section 4: Teknologi
    st.markdown("### Teknologi & Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Support Vector Regression (SVR)**
- Algoritma machine learning untuk prediksi regresi
- Menangani hubungan non-linear dengan efektif
- Robust terhadap outlier dan noise

**ANOVA Radial Basis Function (ANOVA RBF)**
- Kernel untuk menangkap pola kompleks
- Lebih fleksibel dari kernel linear
- Cocok untuk data non-linear dengan seleksi fitur otomatis
        """)
    
    with col2:
        st.markdown("""
**Particle Swarm Optimization (PSO)**
- Optimasi hyperparameter SVR
- Mencari parameter optimal secara otomatis
- Meningkatkan akurasi prediksi

**Data & Features**
- Periode: 2018-2024
- Wilayah: 34 Kabupaten/Kota Jawa Timur
- Variabel: Iklim & Lahan
        """)
    
    st.divider()
    
    # Section 5: Panduan Singkat
    st.markdown("### Panduan Penggunaan Singkat")
    
    steps = [
        ("Input Data", "Masukkan data secara manual di halaman Input Data."),
        ("Proses", "Sistem melakukan preprocessing, scaling, dan optimasi model menggunakan PSO."),
        ("Hasil Prediksi", "Dapatkan hasil prediksi akurat dengan visualisasi grafik dan statistik."),
        ("Analisis", "Gunakan dashboard untuk menganalisis tren dan membuat keputusan strategis."),
    ]
    
    for idx, (step_title, step_desc) in enumerate(steps):
        st.markdown(f"**{idx + 1}. {step_title}**  \n{step_desc}\n")
    
    st.divider()
    
    # Section 6: Informasi Teknis
    st.markdown("### Informasi Teknis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Stack Teknologi**
- Framework: Streamlit
- Language: Python 3.8+
- ML Library: Scikit-learn
- Data Processing: Pandas, NumPy
- Visualization: Matplotlib
        """)
    
    with col2:
        st.markdown("""
**Informasi Sistem**
- Versi: 1.0.0
- Status: Production
- Dataset: 2018-2024
- Wilayah: 38 Kabupaten/Kota 
        """)
    
    st.divider()
    
    # Section 7: Footer
    st.markdown("""
    ---
    **SI-PADI JATIM v1.0** — Sistem Informasi Prediksi Produksi Padi Jawa Timur
    
    Dibuat oleh: **Nur Rohma Widiya Ningsih**
    
    Mendukung ketahanan pangan dan produktivitas pertanian padi di Jawa Timur
    ---
    """)


# ==================== PROSES MODEL SVR-PSO PAGE ====================

def proses_model_svr_pso_page():
    """Halaman untuk proses model SVR-PSO dengan preprocessing pipeline"""
    custom_header("⚙️ Proses Model SVR-PSO", "Pipeline Data Processing & Machine Learning Training")
    
    page_info_box("""
**Tentang Menu Ini**

Menu ini **melatih model SVR baru** dari data historis Anda.
Particle Swarm Optimization (PSO) mencari kombinasi hyperparameter SVR terbaik (C, γ, ε)
sebelum model dipakai untuk prediksi dan dievaluasi.

**Fitur (10 langkah):**
- Upload CSV/Excel atau input manual baris demi baris.
- Preprocessing: ekstrak Bulan/Tahun, imputasi missing value, pilih skenario pemodelan.
- Split kronologis (90:10) atau **10-Fold Cross Validation** .
- Encoding: one-hot kabupaten, sin-cos bulan, MinMaxScaler terpisah untuk X dan y.
- Training PSO + SVR, metrik RMSE & R², grafik konvergensi, unduh CSV & model joblib.

**Alur data:**
1. Data masuk (wajib kolom iklim, lahan, wilayah, periode, dan **Produksi** sebagai target).
2. Diproses sesuai skenario (global dengan/tanpa wilayah, per daerah, atau fitur kustom).
3. Dilatih dengan PSO; hasil prediksi dinormalisasi kembali ke satuan ton untuk evaluasi.

**Parameter yang dapat diatur:**
**Kernel:** RBF atau ANOVA RBF | **PSO:** jumlah partikel, iterasi (w=0.7, c1=c2=1.5 tetap) |
**SVR (range pencarian):** C, ε (epsilon), γ (gamma).
""")
    
    # ANOVA RBF kernel function
    def hitung_anova_rbf(X, Y, gamma, degree=2):
        """ANOVA RBF kernel computation"""
        X, Y = np.asarray(X), np.asarray(Y)
        K = np.zeros((X.shape[0], Y.shape[0]))
        for k in range(X.shape[1]):
            diff_sq = (X[:, k].reshape(-1, 1) - Y[:, k].reshape(1, -1)) ** 2
            K += np.exp(-gamma * diff_sq)
        return (K / X.shape[1]) ** degree
    
    # Initialize session state untuk menyimpan progress
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'scenario_data' not in st.session_state:
        st.session_state.scenario_data = None
    if 'scenario_choice' not in st.session_state:
        st.session_state.scenario_choice = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None
    if 'split_data' not in st.session_state:
        st.session_state.split_data = None
    if 'missing_value_checked' not in st.session_state:
        st.session_state.missing_value_checked = False
    if 'imputed_data' not in st.session_state:
        st.session_state.imputed_data = None
    if 'encoded_data' not in st.session_state:
        st.session_state.encoded_data = None
    if 'normalized_data' not in st.session_state:
        st.session_state.normalized_data = None
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    if 'scaler_X' not in st.session_state:
        st.session_state.scaler_X = None
    if 'scaler_y' not in st.session_state:
        st.session_state.scaler_y = None
    if 'is_cv_mode' not in st.session_state:
        st.session_state.is_cv_mode = False
    if 'cv_results' not in st.session_state:
        st.session_state.cv_results = None
    
    # Helper function untuk extract bulan dari Periode
    def extract_bulan_dari_periode(periode_str):
        """Extract bulan dari string 'Januari 2018' -> 1, 'Februari 2018' -> 2, etc"""
        bulan_map = {
            'Januari': 1, 'Februari': 2, 'Maret': 3, 'April': 4,
            'Mei': 5, 'Juni': 6, 'Juli': 7, 'Agustus': 8,
            'September': 9, 'Oktober': 10, 'November': 11, 'Desember': 12
        }
        for bulan_name, bulan_num in bulan_map.items():
            if bulan_name in periode_str:
                return bulan_num
        return 1  # Default ke Januari jika tidak ketemu
    
    # Helper function untuk extract tahun dari Periode
    def extract_tahun_dari_periode(periode_str):
        """Extract tahun dari string 'Januari 2018' -> 2018"""
        parts = periode_str.split()
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
        return 2018  # Default
    
    # ================= STEP 1: UPLOAD / INPUT MANUAL =================
    section_header("📂 Step 1: Input Data")
    
    if 'manual_rows' not in st.session_state:
        st.session_state.manual_rows = []
    
    input_method = st.radio(
        "Pilih Metode Input Data:",
        ["📁 Upload File (CSV/Excel)", "✏️ Input Manual"],
        horizontal=True,
        key="step1_input_method"
    )
    
    if input_method == "📁 Upload File (CSV/Excel)":
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Pilih file CSV atau Excel",
                type=["csv", "xlsx", "xls"],
                key="file_uploader"
            )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.uploaded_data = df
                st.success(f"✅ File berhasil diupload: {uploaded_file.name} ({len(df)} baris)")
                
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.info(f"📊 Shape data: {df.shape[0]} baris, {df.shape[1]} kolom")
                    st.write("Kolom:", list(df.columns))
            
            except Exception as e:
                st.error(f"❌ Error membaca file: {str(e)}")
    else:
        st.markdown("#### ✏️ Input Data Manual")
        st.caption("Masukkan data training satu per satu. Kolom 'Periode' akan digabung otomatis dari Bulan + Tahun.")
        
        with st.form("manual_train_form", border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                kab_manual = st.selectbox("🏛️ Kabupaten/Kota", ALL_LOCATIONS, key="manual_kab")
            with col2:
                bulan_manual = st.selectbox("📅 Bulan", range(1, 13),
                    format_func=lambda x: BULAN_NAMES[x-1], key="manual_bulan")
            with col3:
                tahun_manual = st.number_input("📅 Tahun", min_value=2018, max_value=2026, value=2024, key="manual_tahun")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                luas_manual = st.number_input("🌾 Luas Panen (ha)", min_value=0.0, max_value=100000.0, value=1000.0, step=10.0, key="manual_luas")
            with col2:
                ch_manual = st.number_input("🌧️ Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=150.0, step=5.0, key="manual_ch")
            with col3:
                kelembapan_manual = st.number_input("💧 Kelembapan (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0, key="manual_kelembapan")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                suhu_manual = st.number_input("🌡️ Suhu (°C)", min_value=15.0, max_value=40.0, value=27.0, step=0.5, key="manual_suhu")
            with col2:
                angin_manual = st.number_input("💨 Kec. Angin (m/s)", min_value=0.0, max_value=35.0, value=3.0, step=0.1, key="manual_angin")
            with col3:
                sinar_manual = st.number_input("☀️ Sinar Matahari (jam/hari)", min_value=0.0, max_value=12.0, value=7.0, step=0.5, key="manual_sinar")
            
            produksi_manual = st.number_input("📦 Produksi (ton) — TARGET", min_value=0.0, max_value=200000.0, value=10000.0, step=100.0, key="manual_produksi")
            
            submitted = st.form_submit_button("➕ Tambah Baris", use_container_width=True)
            if submitted:
                periode_baru = f"{BULAN_NAMES[bulan_manual-1]} {tahun_manual}"
                
                # Validasi duplikat
                is_duplicate = any(
                    row['Kabupaten/Kota'] == kab_manual and
                    row['Periode'] == periode_baru and
                    row['Luas Panen'] == luas_manual and
                    row['Curah Hujan'] == ch_manual and
                    row['Kelembapan'] == kelembapan_manual and
                    row['Suhu'] == suhu_manual and
                    row['Kecepatan Angin'] == angin_manual and
                    row['Sinar Matahari'] == sinar_manual and
                    row['Produksi'] == produksi_manual
                    for row in st.session_state.manual_rows
                )
                
                if is_duplicate:
                    st.warning("⚠️ Data ini sudah ada, tidak ditambahkan (duplikat)")
                else:
                    row = {
                        'Kabupaten/Kota': kab_manual,
                        'Periode': periode_baru,
                        'Luas Panen': luas_manual,
                        'Curah Hujan': ch_manual,
                        'Kelembapan': kelembapan_manual,
                        'Suhu': suhu_manual,
                        'Kecepatan Angin': angin_manual,
                        'Sinar Matahari': sinar_manual,
                        'Produksi': produksi_manual
                    }
                    st.session_state.manual_rows.append(row)
                    st.success(f"✅ Baris {len(st.session_state.manual_rows)} berhasil ditambahkan!")
                    
                    if produksi_manual == 0:
                        st.info("💡 Produksi diisi 0 — pastikan ini benar untuk data training")
                
                st.rerun()
        
        if st.session_state.manual_rows:
            st.markdown("---")
            st.markdown(f"#### 📋 Data Terkumpul ({len(st.session_state.manual_rows)} baris)")
            
            df_manual = pd.DataFrame(st.session_state.manual_rows)
            st.dataframe(df_manual, use_container_width=True, hide_index=True)
            
            col_left, col_mid, col_right = st.columns(3)
            with col_left:
                if st.button("✅ Gunakan Data Ini", use_container_width=True, key="use_manual_data"):
                    st.session_state.uploaded_data = df_manual
                    st.success(f"✅ {len(df_manual)} baris data manual siap diproses!")
                    st.rerun()
            with col_mid:
                if st.button("🔄 Reset Semua", use_container_width=True, key="reset_manual"):
                    st.session_state.manual_rows = []
                    st.rerun()
            with col_right:
                csv_manual = df_manual.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    csv_manual,
                    f"data_training_manual_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Silakan upload file atau input data manual terlebih dahulu untuk melanjutkan")
        return
    
    df = st.session_state.uploaded_data.copy()
    
    # ================= STEP 0B: DATA PREPROCESSING (Extract Bulan & Tahun) =================
    if st.session_state.processed_data is None:
        section_header("🔧 Preprocessing: Extract Bulan & Tahun dari Periode")
        
        if st.button("📋 Auto-Process Data (Extract Bulan & Tahun)", use_container_width=True):
            try:
                df_processed = df.copy()
                
                # Detect kolom Periode
                periode_col = None
                if 'Periode' in df.columns:
                    periode_col = 'Periode'
                elif 'periode' in df.columns:
                    periode_col = 'periode'
                
                # Detect kolom Kabupaten
                kab_col = None
                if 'Kabupaten/Kota' in df.columns:
                    kab_col = 'Kabupaten/Kota'
                elif 'Kabupaten' in df.columns:
                    kab_col = 'Kabupaten'
                elif 'kabupaten' in df.columns:
                    kab_col = 'kabupaten'
                
                # Extract Bulan dan Tahun
                if periode_col:
                    df_processed['Bulan'] = df_processed[periode_col].apply(extract_bulan_dari_periode)
                    df_processed['Tahun'] = df_processed[periode_col].apply(extract_tahun_dari_periode)
                    st.success(f"✅ Berhasil extract Bulan dan Tahun dari kolom '{periode_col}'")
                else:
                    st.warning("⚠️ Kolom 'Periode' tidak ditemukan")
                
                # Rename if needed
                if kab_col and kab_col != 'Kabupaten':
                    df_processed.rename(columns={kab_col: 'Kabupaten'}, inplace=True)
                
                st.session_state.processed_data = df_processed
                st.info(f"Kolom baru ditambahkan: 'Bulan', 'Tahun'")
                
                with st.expander("Preview Data Setelah Preprocessing"):
                    st.dataframe(df_processed.head(10), use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error preprocessing: {str(e)}")
        
        if st.session_state.processed_data is None:
            st.warning("⚠️ Silakan klik tombol untuk preprocessing data")
            return
    
    df = st.session_state.processed_data
    
    # ================= STEP 2: MISSING VALUE CHECK (BEFORE SPLIT) =================
    section_header("🔍 Step 2: Cek Missing Value")
    
    df_check = df.copy()
    missing_count = df_check.isnull().sum()
    has_missing = (missing_count > 0).any()
    
    if has_missing:
        st.warning("⚠️ Ditemukan missing value:")
        st.dataframe(missing_count[missing_count > 0])
        
        if st.button("🔧 Imputasi dengan Mean", use_container_width=True, key="impute_btn_step2"):
            from sklearn.impute import SimpleImputer
            
            numeric_cols = df_check.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='mean')
                df_imputed = df_check.copy()
                df_imputed[numeric_cols] = imputer.fit_transform(df_check[numeric_cols])
                
                st.session_state.processed_data = df_imputed
                df = df_imputed
                st.success("✅ Missing value berhasil di-imputasi")
                
                with st.expander("Preview Data Setelah Imputasi"):
                    st.dataframe(df_imputed.head(), use_container_width=True)
            else:
                st.info("ℹ️ Tidak ada kolom numeric untuk di-imputasi")
    else:
        st.success("✅ Tidak ada missing value")
    
    st.divider()
    
    # ================= STEP 3: SKENARIO PEMODELAN =================
    section_header("🎯 Step 3: Skenario Pemodelan")
    st.markdown("**Pilih pendekatan untuk membandingkan pengaruh data wilayah terhadap akurasi model**")
    
    scenario_choice = st.radio(
        "Pilih Skenario Pemodelan:",
        options=[
            "1. Model Gabungan DENGAN Penanda Daerah (Global - One Hot Encoding)",
            "2. Model Gabungan TANPA Penanda Daerah (Global Blind Model)",
            "3. Model Khusus Per Daerah (Local Model)",
            "4. Custom Feature Selection"
        ],
        key="scenario_selection",
        help="Pilih satu opsi untuk menentukan cara pemrosesan data"
    )
    
    # Inisialisasi selectbox untuk skenario 3
    selected_kabupaten = None
    if "3. Model Khusus Per Daerah" in scenario_choice:
        # Detect kolom Kabupaten
        kab_col = None
        if 'Kabupaten/Kota' in df.columns:
            kab_col = 'Kabupaten/Kota'
        elif 'Kabupaten' in df.columns:
            kab_col = 'Kabupaten'
        elif 'kabupaten' in df.columns:
            kab_col = 'kabupaten'
        
        if kab_col:
            unique_kab = sorted(df[kab_col].unique().tolist())
            selected_kabupaten = st.selectbox(
                f"Pilih Kabupaten/Kota dari kolom '{kab_col}':",
                options=unique_kab,
                key="kabupaten_selection"
            )
            st.info(f"📍 Anda memilih: **{selected_kabupaten}** ({len(df[df[kab_col] == selected_kabupaten])} data points)")
        else:
            st.warning("⚠️ Kolom 'Kabupaten' tidak ditemukan dalam data")
    
    # Inisialisasi multiselect untuk skenario 4
    selected_features = None
    if "4. Custom Feature Selection" in scenario_choice:
        feature_options = [c for c in df.columns if c not in ['Produksi', 'Periode', 'produksi', 'periode']]
        selected_features = st.multiselect(
            "Pilih fitur yang akan digunakan dalam pemodelan:",
            options=feature_options,
            default=[],
            key="feature_selection_multiselect"
        )
        if selected_features:
            st.info(f"✅ {len(selected_features)} dari {len(feature_options)} fitur dipilih")
        else:
            st.warning("⚠️ Belum ada fitur yang dipilih")
    
    if st.button("✅ Terapkan Skenario & Lanjut", use_container_width=True, key="apply_scenario_btn"):
        try:
            df_scenario = df.copy()
            
            # Detect kolom Kabupaten
            kab_col = None
            if 'Kabupaten/Kota' in df_scenario.columns:
                kab_col = 'Kabupaten/Kota'
            elif 'Kabupaten' in df_scenario.columns:
                kab_col = 'Kabupaten'
            elif 'kabupaten' in df_scenario.columns:
                kab_col = 'kabupaten'
            
            if "1. Model Gabungan DENGAN Penanda Daerah" in scenario_choice:
                st.success("✅ Skenario 1: Menggunakan model gabungan DENGAN penanda daerah (One Hot Encoding akan diterapkan di Step 5)")
                st.info(f"Kolom Kabupaten: **{kab_col}** (akan di-encode)")
                st.info("✅ Kolom 'Bulan' dan 'Tahun' dipertahankan untuk Step 6 (Sin-Cos Encoding)")
            elif "2. Model Gabungan TANPA Penanda Daerah" in scenario_choice:
                if kab_col:
                    df_scenario.drop(columns=[kab_col], inplace=True)
                    st.success(f"✅ Skenario 2: Kolom '{kab_col}' berhasil dihapus. Model gabungan TANPA penanda daerah")
                else:
                    st.warning("⚠️ Kolom Kabupaten tidak ditemukan, tidak ada yang dihapus")
                st.info("✅ Kolom 'Bulan' dan 'Tahun' dipertahankan untuk Step 6 (Sin-Cos Encoding)")
            elif "3. Model Khusus Per Daerah" in scenario_choice:
                if kab_col and selected_kabupaten:
                    # Filter data untuk kabupaten yang dipilih
                    df_scenario = df_scenario[df_scenario[kab_col] == selected_kabupaten].copy()
                    df_scenario.drop(columns=[kab_col], inplace=True)
                    st.success(f"✅ Skenario 3: Data difilter untuk **{selected_kabupaten}** ({len(df_scenario)} data points). Kolom '{kab_col}' dihapus.")
                else:
                    st.error("❌ Kabupaten tidak dipilih dengan benar")
                    st.stop()
                st.info("✅ Kolom 'Bulan' dan 'Tahun' dipertahankan untuk Step 6 (Sin-Cos Encoding)")
            elif "4. Custom Feature Selection" in scenario_choice:
                if selected_features:
                    cols_to_drop = [c for c in df_scenario.columns 
                                    if c not in selected_features and c not in ['Produksi', 'produksi']]
                    if cols_to_drop:
                        df_scenario.drop(columns=cols_to_drop, inplace=True)
                        st.success(f"✅ Skenario 4: {len(cols_to_drop)} fitur dihapus ({', '.join(cols_to_drop)})")
                    else:
                        st.success("✅ Skenario 4: Semua fitur dipertahankan")
                    kept_features = [c for c in df_scenario.columns if c not in ['Produksi', 'produksi']]
                    st.info(f"📊 Fitur yang digunakan: {kept_features}")
                else:
                    st.error("❌ Tidak ada fitur yang dipilih. Pilih minimal 1 fitur.")
                    st.stop()
            
            # Simpan ke session state
            st.session_state.scenario_data = df_scenario
            st.session_state.scenario_choice = scenario_choice
            
            # Verify that Bulan column is preserved
            if 'Bulan' in df_scenario.columns or 'bulan' in df_scenario.columns:
                st.success("✅ VERIFIED: Kolom 'Bulan' tersimpan dalam scenario_data")
            else:
                st.warning("⚠️ WARNING: Kolom 'Bulan' tidak ditemukan dalam scenario_data")
            
            with st.expander("👁️ Preview Data Setelah Skenario Diterapkan"):
                st.dataframe(df_scenario.head(10), use_container_width=True)
                st.info(f"📊 Shape data setelah skenario: {df_scenario.shape[0]} baris, {df_scenario.shape[1]} kolom")
                st.write("**Kolom yang tersedia:**", df_scenario.columns.tolist())
            
        except Exception as e:
            st.error(f"❌ Error menerapkan skenario: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    if st.session_state.scenario_data is None:
        st.warning("⚠️ Silakan terapkan skenario pemodelan untuk melanjutkan")
        return
    
    st.divider()
    
    # ================= STEP 4: CHRONOLOGICAL DATA SPLIT (Colab Standard) =================
    section_header("📊 Step 4: Chronological Split Data (NO SHUFFLE)")
    
    split_ratio = st.selectbox(
        "Metode Split",
        options=["90:10", "80:20", "70:30", "60:40", "50:50", "90:10 dan 10-Fold Cross Validation"],
        key="split_ratio",
        help="Pilih rasio split kronologis atau 10-Fold CV "
    )
    
    if st.button("🔄 Chronological Split Data", use_container_width=True, key="split_btn"):
        try:
            # Deteksi apakah mode CV
            is_cv = "10-Fold Cross Validation" in split_ratio
            st.session_state.is_cv_mode = is_cv
            
            # Untuk CV, ratio tetap 90:10 (90% training, 10% testing holdout)
            if is_cv:
                ratio = 0.9
            else:
                ratio = int(split_ratio.split(":")[0]) / 100
            
            # Gunakan scenario_data dari Step 3
            df_split = st.session_state.scenario_data.copy()
            
            # Detect produksi column
            prod_col = 'Produksi' if 'Produksi' in df_split.columns else 'produksi' if 'produksi' in df_split.columns else None
            
            if prod_col is None:
                st.error("❌ Kolom 'Produksi' tidak ditemukan")
                st.stop()
            
            # Sort chronologically by Tahun and Bulan
            if 'Tahun' in df_split.columns and 'Bulan' in df_split.columns:
                df_split = df_split.sort_values(by=['Tahun', 'Bulan']).reset_index(drop=True)
                st.info("✅ Data diurutkan secara kronologis (Tahun -> Bulan)")
            else:
                st.warning("⚠️ Kolom Tahun/Bulan tidak ditemukan, menggunakan urutan original")
            
            # Manual split (chronological, NO RANDOM) using iloc
            split_point = int(len(df_split) * ratio)
            
            # Drop ONLY target (Produksi) and redundant (Periode) columns from X
            # KEEP 'Bulan' dan 'Tahun' untuk Step 5 & 6 encoding
            cols_to_drop = [c for c in ['Produksi', 'Periode'] if c in df_split.columns]
            X = df_split.drop(columns=cols_to_drop, errors='ignore')
            y = df_split[[prod_col]]
            
            # Chronological split (NO SHUFFLE)
            X_train = X.iloc[:split_point]
            X_test = X.iloc[split_point:]
            y_train = y.iloc[:split_point].values.ravel()
            y_test = y.iloc[split_point:].values.ravel()
            
            st.session_state.split_data = {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test,
                'feature_names': X.columns.tolist()
            }
            
            # === INFORMASI SPLIT DETAIL ===
            # Tabel gabungan dengan label split
            df_split_view = df_split.copy()
            split_labels = ['✅ Training'] * split_point + ['🧪 Testing'] * (len(df_split) - split_point)
            df_split_view.insert(0, 'No', range(1, len(df_split) + 1))
            df_split_view['Split'] = split_labels
            
            # Rentang waktu per split
            if 'Tahun' in df_split.columns:
                train_range = f"{int(df_split.iloc[:split_point]['Tahun'].min())} - {int(df_split.iloc[:split_point]['Tahun'].max())}"
                test_range = f"{int(df_split.iloc[split_point:]['Tahun'].min())} - {int(df_split.iloc[split_point:]['Tahun'].max())}"
            else:
                train_range = test_range = "-"
            
            # Kabupaten per split
            kab_col_split = next((c for c in ['Kabupaten/Kota', 'Kabupaten', 'kabupaten'] if c in df_split.columns), None)
            train_kab = df_split.iloc[:split_point][kab_col_split].nunique() if kab_col_split else 0
            test_kab = df_split.iloc[split_point:][kab_col_split].nunique() if kab_col_split else 0
            
            # Deteksi duplikat antar Training & Testing
            cols_for_dup = [c for c in df_split.columns if c not in ['Produksi', 'produksi', 'Periode', 'periode']]
            train_dup = df_split.iloc[:split_point][cols_for_dup].reset_index(drop=True)
            test_dup = df_split.iloc[split_point:][cols_for_dup].reset_index(drop=True)
            merged_dup = pd.merge(train_dup, test_dup, on=cols_for_dup, how='inner')
            duplicate_count = len(merged_dup)
            
            # Simpan metadata ke session_state
            st.session_state.split_data['train_indices'] = list(range(split_point))
            st.session_state.split_data['test_indices'] = list(range(split_point, len(df_split)))
            st.session_state.split_data['duplicate_count'] = duplicate_count
            st.session_state.split_data['kab_count_train'] = train_kab
            st.session_state.split_data['kab_count_test'] = test_kab
            
            st.success(f"✅ Data berhasil di-split KRONOLOGIS dengan ratio {split_ratio}")
            st.info(f"📈 Training: {len(X_train)} data | Testing: {len(X_test)} data")
            st.warning(f"⚠️ Menggunakan chronological split (NO RANDOM SHUFFLE)")
            
            # Verify Bulan column is present
            if 'Bulan' in X_train.columns or 'bulan' in X_train.columns:
                st.success("✅ Kolom 'Bulan' dipertahankan untuk Step 6 (Sin-Cos Encoding)")
            else:
                st.warning("⚠️ Kolom 'Bulan' tidak ditemukan - Sin-Cos Encoding akan di-skip")
            
            with st.expander("📋 Detail Split Data", expanded=True):
                st.markdown("#### 📊 Ringkasan Split")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("📈 Training", f"{len(X_train)} baris")
                with c2:
                    st.metric("🧪 Testing", f"{len(X_test)} baris")
                with c3:
                    st.metric("📅 Rentang Training", train_range)
                with c4:
                    st.metric("📅 Rentang Testing", test_range)
                
                if kab_col_split:
                    ck1, ck2 = st.columns(2)
                    with ck1:
                        st.metric("🏛️ Kabupaten di Training", train_kab)
                    with ck2:
                        st.metric("🏛️ Kabupaten di Testing", test_kab)
                
                st.markdown("---")
                st.markdown("#### 📋 Data Lengkap dengan Label Split")
                st.caption("Kolom **Split** menandakan baris Training ✅ atau Testing 🧪")
                st.dataframe(df_split_view, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown("#### 🔍 Cek Duplikat Training vs Testing")
                if duplicate_count > 0:
                    st.warning(f"⚠️ Ditemukan **{duplicate_count}** baris duplikat antara Training dan Testing! Data identik muncul di kedua set.")
                    with st.expander("Lihat Detail Duplikat"):
                        st.dataframe(merged_dup, use_container_width=True)
                else:
                    st.success("✅ Tidak ada duplikat — semua baris Training unik vs Testing")
                
                st.markdown("---")
                st.markdown("**Kolom dalam X_train:**")
                st.write(X_train.columns.tolist())
        
        except Exception as e:
            st.error(f"❌ Error split data: {str(e)}")
    
    if st.session_state.split_data is None:
        st.warning("⚠️ Silakan split data terlebih dahulu")
        return
    
    st.divider()
    
    X_train = st.session_state.split_data['X_train'].copy()
    X_test = st.session_state.split_data['X_test'].copy()
    y_train = st.session_state.split_data['y_train'].copy()
    y_test = st.session_state.split_data['y_test'].copy()
    
    # ================= STEP 5: ONE-HOT ENCODING KABUPATEN =================
    section_header("🗺️ Step 5: One-Hot Encoding Kabupaten")
    
    # Check if Kabupaten column exists (might be dropped in Step 3 for scenarios 2 & 3)
    kab_exists = 'Kabupaten' in X_train.columns or 'kabupaten' in X_train.columns
    
    if kab_exists:
        kab_col = 'Kabupaten' if 'Kabupaten' in X_train.columns else 'kabupaten'
        
        unique_kab = X_train[kab_col].nunique()
        st.info(f"Jumlah kategori {kab_col}: {unique_kab}")
        with st.expander("Daftar Kabupaten/Kota"):
            st.write(X_train[kab_col].unique().tolist())
        
        if st.button("🔄 One-Hot Encode Kabupaten", use_container_width=True, key="onehot_btn"):
            try:
                X_train_encoded = pd.get_dummies(X_train, columns=[kab_col], prefix='Kab')
                X_test_encoded = pd.get_dummies(X_test, columns=[kab_col], prefix='Kab')
                
                # Ensure same columns in train and test
                train_cols = set(X_train_encoded.columns)
                test_cols = set(X_test_encoded.columns)
                
                # Add missing columns with 0
                for col in train_cols - test_cols:
                    X_test_encoded[col] = 0
                for col in test_cols - train_cols:
                    X_train_encoded[col] = 0
                
                # Reorder columns to match train
                X_test_encoded = X_test_encoded[X_train_encoded.columns]
                
                st.session_state.encoded_data = {
                    'X_train': X_train_encoded,
                    'X_test': X_test_encoded
                }
                st.success("✅ One-hot encoding berhasil")
                st.info(f"Features setelah encoding: {X_train_encoded.shape[1]}")
                
                with st.expander("📋 Preview One-Hot Encoded Data (5 rows)"):
                    st.dataframe(X_train_encoded.head(5), height=200)
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.info(f"ℹ️ Kolom 'Kabupaten' tidak ada (sudah di-drop pada Step 3 untuk skenario 2 atau 3)")
        st.session_state.encoded_data = {
            'X_train': X_train,
            'X_test': X_test
        }
    
    if st.session_state.encoded_data is None:
        st.warning("⚠️ One-hot encoding perlu diselesaikan")
        return
    
    X_train = st.session_state.encoded_data['X_train'].copy()
    X_test = st.session_state.encoded_data['X_test'].copy()
    
    st.divider()
    
    # ================= STEP 6: SIN-COS ENCODING BULAN =================
    section_header("🔄 Step 6: Sin-Cos Encoding Bulan")
    
    # Debug: Show available columns
    with st.expander("🔍 Debug: Kolom yang tersedia di X_train", expanded=False):
        st.write("**Kolom saat ini:**", X_train.columns.tolist())
        st.write("**Tipe data X_train:**")
        st.write(X_train.dtypes)
    
    # Detect Bulan column (case-sensitive check with both variants)
    bulan_col = None
    if 'Bulan' in X_train.columns:
        bulan_col = 'Bulan'
    elif 'bulan' in X_train.columns:
        bulan_col = 'bulan'
    
    if bulan_col is not None:
        st.info(f"✅ Kolom '{bulan_col}' DITEMUKAN! Akan di-encode dengan Sin-Cos (Cyclical Encoding)")
        with st.expander("Info Sin-Cos Encoding", expanded=False):
            st.write(f"Unique values di kolom {bulan_col}:", sorted(X_train[bulan_col].unique().tolist()))
            st.write("Metode: Mengonversi bulan (1-12) ke koordinat sin-cos untuk menangani sifat cyclical (Desember->Januari bersebelahan)")
            st.write("Rumus: bulan_radians = bulan * (2π/12), sin = sin(radians), cos = cos(radians)")
        
        if st.button("🔄 Sin-Cos Encode Bulan", use_container_width=True, key="sincos_btn"):
            try:
                def encode_bulan_sincos(df_temp, bulan_col_name):
                    """Sin-Cos encoding dengan formula Colab: (bulan) * (2*pi/12)"""
                    df_temp = df_temp.copy()
                    bulan_radians = (df_temp[bulan_col_name]) * (2 * np.pi / 12)
                    df_temp[f'{bulan_col_name}_sin'] = np.sin(bulan_radians)
                    df_temp[f'{bulan_col_name}_cos'] = np.cos(bulan_radians)
                    return df_temp.drop(columns=[bulan_col_name])
                
                X_train_encoded = encode_bulan_sincos(X_train, bulan_col)
                X_test_encoded = encode_bulan_sincos(X_test, bulan_col)
                
                st.session_state.encoded_data['X_train'] = X_train_encoded
                st.session_state.encoded_data['X_test'] = X_test_encoded
                
                st.success("✅ Sin-Cos encoding bulan berhasil")
                st.info(f"✨ Kolom '{bulan_col}' sudah dikonversi ke '{bulan_col}_sin' dan '{bulan_col}_cos'")
                st.info(f"📊 Output columns: {X_train_encoded.columns.tolist()}")
                
                with st.expander("📋 Preview Sin-Cos Encoded Data (5 rows)"):
                    st.dataframe(X_train_encoded.head(5), height=200)
            
            except Exception as e:
                st.error(f"❌ Error Sin-Cos Encoding: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    else:
        sincos_already = 'Bulan_sin' in X_train.columns or 'bulan_sin' in X_train.columns
        if sincos_already:
            st.info("✅ Sin-Cos Encoding Bulan sudah selesai dilakukan sebelumnya. Melanjutkan ke Step 7.")
        else:
            st.warning(f"⚠️ Kolom 'Bulan' TIDAK DITEMUKAN. Kolom tersedia: {X_train.columns.tolist()}")
            st.error("❌ ALERT: Sin-Cos Encoding akan di-SKIP. Periksa apakah:")
            st.error("   1. Step 0B berhasil extract kolom 'Bulan' dari 'Periode'")
            st.error("   2. Step 4 Split tidak menghapus kolom 'Bulan'")
            st.info("💡 Meskipun demikian, Anda dapat melanjutkan ke Step 7 tanpa Sin-Cos Encoding")
    
    X_train = st.session_state.encoded_data['X_train'].copy()
    X_test = st.session_state.encoded_data['X_test'].copy()
    
    # Verify flow: Display what columns are being passed to Step 7
    with st.expander("✅ Step 6 Completion & Flow Verification", expanded=False):
        st.write(f"**Sin-Cos Status:** {'✅ ENCODED' if 'Bulan_sin' in X_train.columns or 'bulan_sin' in X_train.columns else '⚠️ SKIPPED'}")
        st.write(f"**Kolom setelah Step 6:** {X_train.columns.tolist()}")
        st.write(f"**Shape X_train:** {X_train.shape}")
        st.write(f"**Bulan masih ada?** {'✅ YES (raw)' if 'Bulan' in X_train.columns or 'bulan' in X_train.columns else '✅ ENCODED ke sin/cos'}")
    
    st.divider()
    
    # ================= STEP 7: DOUBLE SCALING (X & y TERPISAH - Colab Standard) =================
    section_header("📏 Step 7: Double Scaling MinMaxScaler (X dan y Terpisah)")
    
    st.info("🔬 Menggunakan MinMaxScaler untuk X dan y TERPISAH")
    
    if st.button("📊 Terapkan Double Scaling", use_container_width=True, key="norm_btn"):
        try:
            from sklearn.preprocessing import MinMaxScaler
            
            # Get numeric columns from X_train
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                st.error("❌ Tidak ada kolom numeric untuk dinormalisasi")
                return
            
            # Scaler untuk X
            scaler_X = MinMaxScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numeric_cols] = scaler_X.fit_transform(X_train[numeric_cols])
            X_test_scaled[numeric_cols] = scaler_X.transform(X_test[numeric_cols])
            
            # Scaler untuk y (TERPISAH)
            scaler_y = MinMaxScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
            
            # Store both scalers
            st.session_state.scaler_X = scaler_X
            st.session_state.scaler_y = scaler_y
            
            st.session_state.normalized_data = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train_scaled': y_train_scaled,
                'y_test_scaled': y_test_scaled,
                'y_train_original': y_train,
                'y_test_original': y_test,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'numeric_cols': numeric_cols
            }
            
            st.success("✅ Double Scaling berhasil (MinMaxScaler untuk X dan y TERPISAH)")
            st.info(f"📊 Features scaled: {len(numeric_cols)} kolom | Target (y) scaled terpisah")
            
            with st.expander("📋 Preview X Scaled (5 rows)"):
                st.dataframe(X_train_scaled[numeric_cols].head(5), height=200)
            
            with st.expander("📊 Preview y Scaled (5 values)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**y_train Original:**", y_train[:5])
                with col2:
                    st.write("**y_train Scaled:**", y_train_scaled[:5])
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    if st.session_state.normalized_data is None:
        st.warning("⚠️ Double Scaling perlu diselesaikan")
        return
    
    X_train = st.session_state.normalized_data['X_train'].copy()
    X_test = st.session_state.normalized_data['X_test'].copy()
    y_train_scaled = st.session_state.normalized_data['y_train_scaled'].copy()
    y_test_scaled = st.session_state.normalized_data['y_test_scaled'].copy()
    
    # ================= STEP 8: KONFIGURASI PARAMETER PSO =================
    section_header("⚙️ Step 8: Konfigurasi Parameter PSO & SVR")
    
    section_header("🎯 Pilih Kernel SVR", level=3)
    kernel_choice = st.radio(
        "Kernel Type",
        options=["RBF", "ANOVA RBF"],
        horizontal=True,
        help="RBF: Standard Gaussian kernel. ANOVA RBF: Custom ANOVA kernel untuk hasil yang lebih optimal."
    )
    
    if kernel_choice == "RBF":
        st.info("✅ Menggunakan kernel **RBF (Gaussian)**")
    else:
        st.info("✅ Menggunakan kernel **ANOVA RBF** (Custom - Recommended)")
    
    st.divider()
    
    section_header("📌 Parameter PSO (Default - Fixed)", level=3)
    st.markdown("Inertia Weight (w): **0.7** | Cognitive (c1): **1.5** | Social (c2): **1.5**")
    
    st.divider()
    
    section_header("🎯 Parameter SVR (Adjustable)", level=3)
    st.markdown("**Range C (Regularization)**")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        c_min = st.number_input("C Min", value=1.0, min_value=0.001, step=1.0, help="Minimum C value from 1 to 1000")
    with col_c2:
        c_max = st.number_input("C Max", value=1000.0, min_value=1.0, step=10.0, help="Maximum C value from 1 to 1000")
    
    st.markdown("**Range Epsilon (Tolerance)**")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        epsilon_min = st.number_input("Epsilon Min", value=0.000001, min_value=0.000001, format="%.6f", help="Minimum epsilon from 0.000001 to 0.1")
    with col_e2:
        epsilon_max = st.number_input("Epsilon Max", value=0.1, min_value=0.000001, format="%.6f", help="Maximum epsilon from 0.000001 to 0.1")
    
    st.markdown("**Range Gamma (Kernel Coefficient)**")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gamma_min = st.number_input("Gamma Min", value=0.00001, min_value=0.00001, format="%.5f", help="Minimum gamma from 0.00001 to 100")
    with col_g2:
        gamma_max = st.number_input("Gamma Max", value=100.0, min_value=0.00001, step=1.0, help="Maximum gamma from 0.00001 to 100")
    
    st.markdown("**PSO Configuration**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        n_particles = st.number_input("Jumlah Partikel", value=20, min_value=5, max_value=100, step=5, help="Banyak partikel dalam swarm")
    with col_p2:
        n_iter = st.number_input("Jumlah Iterasi", value=10, min_value=2, max_value=250, step=1, help="Banyak iterasi PSO (max 250)")
    
    # Default PSO parameters (fixed)
    w, c1, c2 = 0.7, 1.5, 1.5
    
    st.divider()
    section_header("⏱️ Perkiraan Waktu Training", level=3)
    eta_secs = n_particles * n_iter * 2
    eta_mins = eta_secs // 60
    eta_secs = eta_secs % 60
    st.info(f"⏳ Estimasi waktu training: ~{eta_mins} menit {eta_secs} detik")
    
    # Warning if training will take long time
    if n_iter > 100:
        st.warning(f"⚠️ Dengan {n_iter} iterasi, training diprediksi akan menghabiskan waktu ~{eta_mins} menit. Pastikan Anda siap menunggu!")
    elif n_iter > 50:
        st.info(f"💡 Training akan membutuhkan waktu ~{eta_mins} menit dengan {n_iter} iterasi")
    
    # ================= STEP 9: TRAINING MODEL PSO =================
    section_header("🚀 Step 9: Training Model SVR dengan PSO")

    is_cv_mode = st.session_state.get('is_cv_mode', False)

    if is_cv_mode:
        st.info("📌 **Mode: 90:10 + 10-Fold Cross Validation**")
    else:
        st.info(f"📌 **Mode: Hold Out ({split_ratio})** — PSO langsung optimasi di data training")
    
    if st.button("🎯 Mulai Training", key="train_button", use_container_width=True):
        # Initialize placeholders early so exception handling can reference them
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        metric_placeholder = st.empty()
        params_placeholder = st.empty()
        chart_placeholder = st.empty()
        anim_placeholder = st.empty()

        # Spinner characters for simple animation
        spinner_chars = ['|', '/', '-', '\\']

        def update_ui(iter_info, stage_label="Progress"):
            try:
                prog = float(iter_info.get('progress', 0.0))
            except Exception:
                prog = 0.0
            # Clamp
            prog = max(0.0, min(1.0, prog))
            # Update progress bar
            progress_placeholder.progress(prog)

            # Iteration / metric
            iteration = iter_info.get('iteration')
            total = iter_info.get('total')
            best_rmse = iter_info.get('best_rmse')
            status_text = f"⏳ {stage_label}"
            if iteration is not None and total is not None:
                spin = spinner_chars[int(iteration) % len(spinner_chars)] if iteration is not None else ''
                status_text += f" — Iter {iteration}/{total} {spin}"
            status_placeholder.info(status_text)

            # Metric
            if best_rmse is not None:
                metric_placeholder.metric("Best RMSE", id_format(best_rmse, 2))

            # Best params
            best_params = iter_info.get('best_params') or {}
            if best_params:
                params_placeholder.markdown(f"**Best params:** C={best_params.get('C')}  •  ε={best_params.get('epsilon')}  •  γ={best_params.get('gamma')}")

            # RMSE history chart
            rmse_hist = iter_info.get('rmse_history')
            if rmse_hist:
                try:
                    import pandas as _pd
                    chart_placeholder.line_chart(_pd.DataFrame({'rmse': rmse_hist}))
                except Exception:
                    # fallback: simple text
                    chart_placeholder.text(f"RMSE history: {rmse_hist[-5:]}")
        try:
            start_all = time.time()
            faktor_cv = 5 if is_cv_mode else 1
            est_detik = n_particles * n_iter * faktor_cv * 3
            if est_detik > 60:
                est_str = f"{est_detik // 60} menit {est_detik % 60} detik"
            else:
                est_str = f"{est_detik} detik"
            
            st.info(f"⏳ **Estimasi: ~{est_str}**")
            st.warning("⚠️ **Jangan refresh halaman selama training berjalan!**")
            
            if is_cv_mode and kernel_choice == "RBF":
                # ===== CV MODE: 3 Tahap dengan RBF =====
                from models.svr_fast import pso_training_cv, validate_cv_fast, train_final_model_fast
                
                # Stage 1: PSO Optimization 
                status_placeholder.info("⏳ **Tahap 1/3:** PSO Optimization...")
                results_pso = None
                for iter_info in pso_training_cv(
                    X_train.values, y_train_scaled, st.session_state.scaler_y,
                    n_particles, n_iter, c_min, c_max, epsilon_min, epsilon_max, gamma_min, gamma_max,
                    w=w, c1=c1, c2=c2, n_folds=5
                ):
                    # Update interactive UI (progress bar, metric, params, chart, spinner)
                    update_ui(iter_info, stage_label="Tahap 1/3: PSO Optimization ")
                    results_pso = iter_info
                
                best_params = results_pso.get('best_params', {'C': 1, 'epsilon': 0.1, 'gamma': 0.01})
                rmse_history = results_pso.get('rmse_history', [])
                
                status_placeholder.success(f"✅ Tahap 1 selesai! Best Avg RMSE : {id_format(results_pso['best_rmse'], 2)}")
                
                # Stage 2: Validate with 10-Fold CV
                status_placeholder.info("⏳ **Tahap 2/3:** Validasi dengan 10-Fold CV...")
                fold_results, avg_val_rmse = validate_cv_fast(
                    best_params, X_train.values, y_train_scaled,
                    st.session_state.scaler_y, n_folds=10
                )
                progress_placeholder.progress(0.85)
                
                st.session_state.cv_results = {
                    'fold_results': fold_results,
                    'avg_val_rmse': avg_val_rmse
                }
                
                status_placeholder.success(f"✅ Tahap 2 selesai! Avg RMSE (10-Fold): {id_format(avg_val_rmse, 2)}")
                
                # Stage 3: Final Training & Testing
                status_placeholder.info("⏳ **Tahap 3/3:** Final Training di 100% data + Testing di data holdout 10%...")
                model, y_pred_asli, y_test_asli, test_rmse, test_r2 = train_final_model_fast(
                    best_params, X_train.values, y_train_scaled,
                    st.session_state.scaler_y, X_test.values, y_test_scaled
                )
                progress_placeholder.progress(0.95)
                if y_pred_asli is not None:
                    y_pred_asli = np.clip(y_pred_asli, 0, None)
                # Get training predictions for CV RBF
                y_pred_train_scaled = model.predict(X_train.values)
                y_pred_train_asli = st.session_state.scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
                y_pred_train_asli = np.clip(y_pred_train_asli, 0, None)
                train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train_asli)))
                ss_res_train = np.sum((y_train - y_pred_train_asli) ** 2)
                ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
                train_r2 = float(1 - ss_res_train / ss_tot_train) if ss_tot_train != 0 else 0.0
                st.session_state.training_results = {
                    'best_params': best_params,
                    'rmse': test_rmse,
                    'r2': test_r2,
                    'train_rmse': train_rmse,
                    'train_r2': train_r2,
                    'predictions': y_pred_asli,
                    'y_test': y_test_asli,
                    'predictions_train': y_pred_train_asli,
                    'y_train': y_train,
                    'rmse_history': rmse_history,
                    'model': model,
                    'particles': n_particles,
                    'iterasi': n_iter,
                    'kernel': kernel_choice,
                    'scaler_y': st.session_state.scaler_y,
                    'scaler_X': st.session_state.scaler_X,
                    'feature_columns': list(X_train.columns),
                    'scenario': st.session_state.scenario_choice,
                    'split_ratio': split_ratio,
                    'is_cv': True,
                    'cv_avg_rmse_5fold': results_pso['best_rmse'],
                    'cv_avg_rmse_10fold': avg_val_rmse,
                }
                
                progress_placeholder.progress(1.0)
                metric_placeholder.empty()
                status_placeholder.success(f"✅ Training Selesai! Train RMSE: {id_format(train_rmse, 2)} | Test RMSE: {id_format(test_rmse, 2)}")
                
            elif is_cv_mode and kernel_choice == "ANOVA RBF":
                # ===== CV MODE: 3 Tahap dengan ANOVA RBF =====
                from models.svr_anova import pso_training_cv_anova, validate_cv_anova, train_final_model_anova
                
                # Stage 1: PSO Optimization 
                status_placeholder.info("⏳ **Tahap 1/3:** PSO Optimization dengan CV (ANOVA RBF)...")
                results_pso = None
                for iter_info in pso_training_cv_anova(
                    X_train.values, y_train_scaled, st.session_state.scaler_y,
                    n_particles, n_iter, c_min, c_max, epsilon_min, epsilon_max, gamma_min, gamma_max,
                    w=w, c1=c1, c2=c2, n_folds=5
                ):
                    # Update interactive UI (progress bar, metric, params, chart, spinner)
                    update_ui(iter_info, stage_label="Tahap 1/3: PSO Optimization (ANOVA)")
                    results_pso = iter_info
                
                best_params = results_pso.get('best_params', {'C': 1, 'epsilon': 0.1, 'gamma': 0.01})
                rmse_history = results_pso.get('rmse_history', [])
                
                status_placeholder.success(f"✅ Tahap 1 selesai! Best Avg RMSE: {id_format(results_pso['best_rmse'], 2)}")
                
                # Stage 2: Validate with 10-Fold CV
                status_placeholder.info("⏳ **Tahap 2/3:** Validasi dengan 10-Fold CV (ANOVA RBF)...")
                fold_results, avg_val_rmse = validate_cv_anova(
                    best_params, X_train.values, y_train_scaled,
                    st.session_state.scaler_y, n_folds=10
                )
                progress_placeholder.progress(0.85)
                
                st.session_state.cv_results = {
                    'fold_results': fold_results,
                    'avg_val_rmse': avg_val_rmse
                }
                
                status_placeholder.success(f"✅ Tahap 2 selesai! Avg RMSE (10-Fold): {id_format(avg_val_rmse, 2)}")
                
                # Stage 3: Final Training & Testing
                status_placeholder.info("⏳ **Tahap 3/3:** Final Training + Testing...")
                model, y_pred_asli, y_test_asli, test_rmse, test_r2 = train_final_model_anova(
                    best_params, X_train.values, y_train_scaled,
                    st.session_state.scaler_y, X_test.values, y_test_scaled
                )
                progress_placeholder.progress(0.95)
                if y_pred_asli is not None:
                    y_pred_asli = np.clip(y_pred_asli, 0, None)
                # Get training predictions for CV ANOVA
                y_pred_train_scaled = model.predict(X_train.values)
                y_pred_train_asli = st.session_state.scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
                y_pred_train_asli = np.clip(y_pred_train_asli, 0, None)
                train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train_asli)))
                ss_res_train = np.sum((y_train - y_pred_train_asli) ** 2)
                ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
                train_r2 = float(1 - ss_res_train / ss_tot_train) if ss_tot_train != 0 else 0.0
                st.session_state.training_results = {
                    'best_params': best_params,
                    'rmse': test_rmse,
                    'r2': test_r2,
                    'train_rmse': train_rmse,
                    'train_r2': train_r2,
                    'predictions': y_pred_asli,
                    'y_test': y_test_asli,
                    'predictions_train': y_pred_train_asli,
                    'y_train': y_train,
                    'rmse_history': rmse_history,
                    'model': model,
                    'particles': n_particles,
                    'iterasi': n_iter,
                    'kernel': kernel_choice,
                    'scaler_y': st.session_state.scaler_y,
                    'scaler_X': st.session_state.scaler_X,
                    'feature_columns': list(X_train.columns),
                    'scenario': st.session_state.scenario_choice,
                    'split_ratio': split_ratio,
                    'is_cv': True,
                    'cv_avg_rmse_5fold': results_pso['best_rmse'],
                    'cv_avg_rmse_10fold': avg_val_rmse,
                }
                
                progress_placeholder.progress(1.0)
                metric_placeholder.empty()
                status_placeholder.success(f"✅ Training Selesai! Train RMSE: {id_format(train_rmse, 2)} | Test RMSE: {id_format(test_rmse, 2)}")
                
            elif kernel_choice == "RBF":
                # ===== HOLD OUT MODE: RBF (direct, no KFold) =====
                from models.svr_fast import pso_training_direct
                
                status_placeholder.info(f"⏳ PSO Training dengan kernel RBF (Hold Out)...")
                
                results_pso = None
                for iter_info in pso_training_direct(
                    X_train.values, y_train_scaled, X_test.values, y_test_scaled,
                    st.session_state.scaler_y,
                    n_particles, n_iter, c_min, c_max, epsilon_min, epsilon_max, gamma_min, gamma_max,
                    w=w, c1=c1, c2=c2
                ):
                    # Update interactive UI (progress bar, metric, params, chart, spinner)
                    update_ui(iter_info, stage_label="PSO Training (Hold Out - RBF)")
                    results_pso = iter_info

                best_params = results_pso.get('best_params', {'C': 1, 'epsilon': 0.1, 'gamma': 0.01})
                rmse_history = results_pso.get('rmse_history', [])
                model = results_pso.get('model')
                y_pred_scaled = results_pso.get('predictions')

                if y_pred_scaled is not None:
                    y_pred_asli = st.session_state.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    y_pred_asli = np.clip(y_pred_asli, 0, None)
                    y_test_asli = st.session_state.scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
                    test_rmse = float(np.sqrt(mean_squared_error(y_test_asli, y_pred_asli)))
                    ss_res = np.sum((y_test_asli - y_pred_asli) ** 2)
                    ss_tot = np.sum((y_test_asli - np.mean(y_test_asli)) ** 2)
                    test_r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
                    # Get training predictions for Holdout RBF
                    y_pred_train_scaled = model.predict(X_train.values)
                    y_pred_train_asli = st.session_state.scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
                    y_pred_train_asli = np.clip(y_pred_train_asli, 0, None)
                    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train_asli)))
                    ss_res_train = np.sum((y_train - y_pred_train_asli) ** 2)
                    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
                    train_r2 = float(1 - ss_res_train / ss_tot_train) if ss_tot_train != 0 else 0.0
                else:
                    y_pred_asli = None
                    y_test_asli = None
                    test_rmse = float('inf')
                    test_r2 = 0.0
                    y_pred_train_asli = None
                    train_rmse = float('inf')
                    train_r2 = 0.0

                st.session_state.training_results = {
                    'best_params': best_params,
                    'rmse': test_rmse,
                    'r2': test_r2,
                    'train_rmse': train_rmse,
                    'train_r2': train_r2,
                    'predictions': y_pred_asli,
                    'y_test': y_test_asli,
                    'predictions_train': y_pred_train_asli,
                    'y_train': y_train,
                    'rmse_history': rmse_history,
                    'model': model,
                    'particles': n_particles,
                    'iterasi': n_iter,
                    'kernel': kernel_choice,
                    'scaler_y': st.session_state.scaler_y,
                    'scaler_X': st.session_state.scaler_X,
                    'feature_columns': list(X_train.columns),
                    'scenario': st.session_state.scenario_choice,
                    'split_ratio': split_ratio,
                    'is_cv': False,
                }

                progress_placeholder.progress(1.0)
                metric_placeholder.empty()
                status_placeholder.success(f"✅ Training Selesai! Train RMSE: {id_format(train_rmse, 2)} | Test RMSE: {id_format(test_rmse, 2)}")

            else:
                # ===== HOLD OUT MODE: ANOVA RBF (direct, no KFold) =====
                from models.svr_anova import pso_training_direct_anova
                
                status_placeholder.info(f"⏳ PSO Training dengan kernel ANOVA RBF (Hold Out)...")
                
                results_pso = None
                for iter_info in pso_training_direct_anova(
                    X_train.values, y_train_scaled, X_test.values, y_test_scaled,
                    st.session_state.scaler_y,
                    n_particles, n_iter, c_min, c_max, epsilon_min, epsilon_max, gamma_min, gamma_max,
                    w=w, c1=c1, c2=c2
                ):
                    # Update interactive UI (progress bar, metric, params, chart, spinner)
                    update_ui(iter_info, stage_label="PSO Training (Hold Out - ANOVA)")
                    results_pso = iter_info

                best_params = results_pso.get('best_params', {'C': 1, 'epsilon': 0.1, 'gamma': 0.01})
                rmse_history = results_pso.get('rmse_history', [])
                model = results_pso.get('model')
                y_pred_scaled = results_pso.get('predictions')

                if y_pred_scaled is not None:
                    y_pred_asli = st.session_state.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                    y_pred_asli = np.clip(y_pred_asli, 0, None)
                    y_test_asli = st.session_state.scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
                    test_rmse = float(np.sqrt(mean_squared_error(y_test_asli, y_pred_asli)))
                    ss_res = np.sum((y_test_asli - y_pred_asli) ** 2)
                    ss_tot = np.sum((y_test_asli - np.mean(y_test_asli)) ** 2)
                    test_r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
                    # Get training predictions for Holdout ANOVA
                    y_pred_train_scaled = model.predict(X_train.values)
                    y_pred_train_asli = st.session_state.scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
                    y_pred_train_asli = np.clip(y_pred_train_asli, 0, None)
                    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train_asli)))
                    ss_res_train = np.sum((y_train - y_pred_train_asli) ** 2)
                    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
                    train_r2 = float(1 - ss_res_train / ss_tot_train) if ss_tot_train != 0 else 0.0
                else:
                    y_pred_asli = None
                    y_test_asli = None
                    test_rmse = float('inf')
                    test_r2 = 0.0
                    y_pred_train_asli = None
                    train_rmse = float('inf')
                    train_r2 = 0.0

                st.session_state.training_results = {
                    'best_params': best_params,
                    'rmse': test_rmse,
                    'r2': test_r2,
                    'train_rmse': train_rmse,
                    'train_r2': train_r2,
                    'predictions': y_pred_asli,
                    'y_test': y_test_asli,
                    'predictions_train': y_pred_train_asli,
                    'y_train': y_train,
                    'rmse_history': rmse_history,
                    'model': model,
                    'particles': n_particles,
                    'iterasi': n_iter,
                    'kernel': kernel_choice,
                    'scaler_y': st.session_state.scaler_y,
                    'scaler_X': st.session_state.scaler_X,
                    'feature_columns': list(X_train.columns),
                    'scenario': st.session_state.scenario_choice,
                    'split_ratio': split_ratio,
                    'is_cv': False,
                }

                progress_placeholder.progress(1.0)
                metric_placeholder.empty()
                status_placeholder.success(f"✅ Training Selesai! Train RMSE: {id_format(train_rmse, 2)} | Test RMSE: {id_format(test_rmse, 2)}")
        
        except Exception as e:
            status_placeholder.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.error(traceback.format_exc())
    
    if st.session_state.training_results is None:
        st.warning("⚠️ Silakan lakukan training terlebih dahulu")
        return
    
    # ================= STEP 10: HASIL DAN VISUALISASI =================
    section_header("📊 Step 10: Hasil Prediksi")
    
    results = st.session_state.training_results
    
    # Validate results
    if results.get('predictions') is None or results.get('y_test') is None:
        st.error("❌ Training gagal - predictions atau y_test tidak valid")
        if results.get('kernel') == 'ANOVA RBF':
            st.warning("⚠️ Kernel ANOVA RBF mungkin tidak compatible dengan data ini. Coba gunakan kernel RBF.")
        return
    
    if np.isinf(results.get('rmse', 0)):
        st.error("❌ Training tidak berhasil - RMSE tidak valid")
        if results.get('kernel') == 'ANOVA RBF':
            st.warning("⚠️ Kernel ANOVA RBF mungkin tidak compatible dengan data ini. Coba gunakan kernel RBF.")
        return
    
    # Pastikan semua prediksi non-negatif
    if results.get('predictions') is not None:
        results['predictions'] = np.clip(results['predictions'], 0, None)

    st.info("✅ Hasil prediksi telah di-**DENORMALISASI** ke satuan TON asli menggunakan scaler_y")
    
    # Training Configuration Info
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🎯 Kernel", results.get('kernel', 'RBF'))
    with col2:
        st.metric("⚙️ Particles", results.get('particles', 'N/A'))
    with col3:
        st.metric("🔹 Iterasi", results.get('iterasi', 'N/A'))
    with col4:
        st.metric("📊 Train Size", len(X_train))
    with col5:
        st.metric("📊 Test Size", len(X_test))
    
    st.divider()
    
    # Display selected scenario
    if st.session_state.scenario_choice:
        section_header("🎯 Skenario Pemodelan yang Digunakan", level=3)
        st.info(f"**Skenario:** {st.session_state.scenario_choice}")
    
    # CV Results Summary (if CV mode)
    if results.get('is_cv', False):
        st.divider()
        section_header("🔬 Validasi Cross-Validation", level=3)
        col2, = st.columns(1)
        with col2:
            st.markdown(metric_card("Avg RMSE (10-Fold Validasi)", f"{id_format(results.get('cv_avg_rmse_10fold', 0), 2)}", "📈"), unsafe_allow_html=True)
        
        # Per-fold results table
        cv_results = st.session_state.get('cv_results', None)
        if cv_results and cv_results.get('fold_results'):
            with st.expander("📋 Detail RMSE per Fold (10-Fold Validasi)", expanded=False):
                fold_df = pd.DataFrame(cv_results['fold_results'])
                fold_df.columns = ['Fold', 'RMSE', 'N Train', 'N Val']
                fold_df['RMSE'] = fold_df['RMSE'].apply(lambda x: f"{x:,.2f}")
                st.dataframe(fold_df, use_container_width=True, hide_index=True)
                
                # Bar chart per fold
                fig_fold, ax_fold = plt.subplots(figsize=(10, 3))
                folds_plot = cv_results['fold_results']
                rmse_vals = [f['rmse'] for f in folds_plot]
                ax_fold.bar(range(1, len(rmse_vals)+1), rmse_vals, color='#2e7d32', alpha=0.7, edgecolor='#1b5e20')
                ax_fold.axhline(y=cv_results['avg_val_rmse'], color='#d32f2f', linestyle='--', linewidth=2, label=f"Rata-rata: {cv_results['avg_val_rmse']:.2f}")
                ax_fold.set_xlabel("Fold ke-", fontsize=10, fontweight='bold')
                ax_fold.set_ylabel("RMSE", fontsize=10, fontweight='bold')
                ax_fold.set_title("RMSE per Fold (10-Fold Validasi)", fontsize=11, fontweight='bold')
                ax_fold.legend(fontsize=9)
                ax_fold.grid(True, alpha=0.3)
                st.pyplot(fig_fold, use_container_width=True)
    
    st.divider()
    section_header("📈 Metrik Performa Model", level=3)

    # Use only RMSE and R² for evaluation (training & testing)
    train_rmse = results.get('train_rmse', float('nan'))
    train_r2 = results.get('train_r2', float('nan'))
    test_rmse = results.get('rmse', float('nan'))
    test_r2 = results.get('r2', float('nan'))

    # Compact, responsive two-column layout
    
    col_train, col_test = st.columns([1, 1], gap='large')

    train_title = """
    <div style="background: #e8f5e9; padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; color: #1b5e20; font-weight: 700; font-size: 1rem;">
        📚 Metrik Training
    </div>
    """

    test_title = """
    <div style="background: #e3f2fd; padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; color: #0d47a1; font-weight: 700; font-size: 1rem;">
        🧪 Metrik Testing
    </div>
    """

    with col_train:
        st.markdown(train_title, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="RMSE", value=f"{id_format(train_rmse, 2)}")
        with c2:
            st.metric(label="R² Score", value=f"{id_format(train_r2, 4)}")

    with col_test:
        st.markdown(test_title, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="RMSE", value=f"{id_format(test_rmse, 2)}")
        with c2:
            st.metric(label="R² Score", value=f"{id_format(test_r2, 4)}")
    
    # Best Parameters
    section_header("⚙️ Parameter Terbaik", level=3)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**C:** {id_format(results['best_params']['C'], 4)}")
    with col2:
        st.write(f"**Epsilon:** {id_format(results['best_params']['epsilon'], 6)}")
    with col3:
        st.write(f"**Gamma:** {results['best_params']['gamma']}")
    
    # Graph: RMSE Progress
    section_header("📊 Grafik: Progres RMSE per Iterasi PSO", level=3)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(results['rmse_history'], marker='o', linewidth=2, markersize=4, color='#2e7d32')
    ax.set_xlabel("Iterasi", fontsize=11, fontweight='bold')
    ax.set_ylabel("RMSE", fontsize=11, fontweight='bold')
    ax.set_title(f"Konvergensi PSO - Kernel: {results.get('kernel', 'RBF')}", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    # Graph: Prediksi vs Aktual
    section_header("📊 Grafik: Prediksi vs Data Aktual", level=3)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x_axis = range(len(results['y_test']))
    ax.plot(x_axis, results['y_test'], label='Data Aktual', marker='o', linewidth=2, markersize=5, color='#d32f2f')
    ax.plot(x_axis, results['predictions'], label='Prediksi', marker='s', linewidth=2, markersize=5, color='#2e7d32', alpha=0.8)
    ax.set_xlabel("Index Data Testing", fontsize=11, fontweight='bold')
    ax.set_ylabel("Produksi (Ton)", fontsize=11, fontweight='bold')
    ax.set_title(f"Perbandingan Prediksi Model - Kernel: {results.get('kernel', 'RBF')}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    # Graph: Scatter Plot
    section_header("📊 Grafik: Scatter Plot Prediksi vs Aktual", level=3)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results['y_test'], results['predictions'], alpha=0.6, s=80, color='#2e7d32')
    
    min_val = min(results['y_test'].min(), results['predictions'].min())
    max_val = max(results['y_test'].max(), results['predictions'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel("Data Aktual (Ton)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Prediksi Model (Ton)", fontsize=11, fontweight='bold')
    ax.set_title(f"Scatter Plot: Akurasi Prediksi - Kernel: {results.get('kernel', 'RBF')}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, use_container_width=True)
    
    # Tabel Preview
    section_header("📋 Tabel: Preview Prediksi", level=3)
    
    X_test_orig = st.session_state.split_data['X_test'].reset_index(drop=True)
    
    kab_col = next((c for c in ['Kabupaten/Kota', 'Kabupaten', 'kabupaten'] if c in X_test_orig.columns), None)
    bulan_col = next((c for c in ['Bulan', 'bulan'] if c in X_test_orig.columns), None)
    tahun_col = next((c for c in ['Tahun', 'tahun'] if c in X_test_orig.columns), None)
    
    abs_error = np.abs(results['y_test'] - results['predictions'])
    
    preview_data = {}
    if kab_col:
        preview_data['Kabupaten/Kota'] = X_test_orig[kab_col].values
    if bulan_col:
        preview_data['Bulan'] = X_test_orig[bulan_col].values
    if tahun_col:
        preview_data['Tahun'] = X_test_orig[tahun_col].values
    preview_data['Data Aktual (Ton)'] = [id_format(x, 2) for x in results['y_test']]
    preview_data['Prediksi (Ton)'] = [id_format(x, 2) for x in results['predictions']]
    preview_data['Error (Ton)'] = [id_format(x, 2) for x in abs_error]
    
    preview_df = pd.DataFrame(preview_data)
    st.dataframe(preview_df, use_container_width=True)
    
    # Download CSV & Model
    section_header("💾 Download Hasil", level=3)
    
    st.caption(
        "Model joblib memakai pipeline training halaman ini (scaler + fitur hasil encoding). "
        "Berbeda dari `model_final_padi.save` di menu Prediksi Cepat."
    )
    
    col_csv, col_model = st.columns(2)
    with col_csv:
        csv = preview_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Hasil Prediksi (CSV)",
            data=csv,
            file_name="hasil_prediksi_svr_pso.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_csv_results",
        )
    with col_model:
        model_ok = results.get("model") is not None
        if model_ok:
            kernel_slug = str(results.get("kernel", "model")).replace(" ", "_").lower()
            bundle = build_model_bundle(results)
            model_buffer = io.BytesIO()
            joblib.dump(bundle, model_buffer)
            model_buffer.seek(0)
            st.download_button(
                label="📦 Download Model (joblib)",
                data=model_buffer,
                file_name=f"model_svr_pso_{kernel_slug}.joblib",
                mime="application/octet-stream",
                use_container_width=True,
                key="download_joblib_model",
            )
        else:
            st.button(
                "📦 Download Model (joblib)",
                disabled=True,
                use_container_width=True,
                help="Model tidak tersedia — training gagal atau belum selesai.",
            )


# ==================== MAIN APP ====================

def main():
    """Main application dengan desain profesional"""
    
    
    
    # Sidebar Header Profesional
    st.sidebar.markdown("""
    <div style="padding: 20px; text-align: center; background: linear-gradient(135deg, #2e7d32, #388e3c); 
                border-radius: 10px; color: white; margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 28px;">🌾 SI-PADI JATIM</h1>
    <p style="margin: 5px 0 0 0; font-size: 12px; opacity: 0.9;">Prediksi Produksi Padi Jawa Timur</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigasi
    st.sidebar.markdown("### 📋 NAVIGASI HALAMAN")
    page = st.sidebar.radio(
        "Pilih halaman",
        ["Beranda", "Prediksi Cepat", "Proses Model SVR-PSO",
        "Visualisasi", "Tentang Model", "About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Info Section
    st.sidebar.markdown("### 🌟 Fitur Utama")
    st.sidebar.markdown("""
    **Input Fleksibel**  
    Masukkan data manual 
    
    **Prediksi SVR-PSO-ANOVA-RBF**  
    Algoritma machine learning canggih
    
    **Analytics Lengkap**  
    Visualisasi data dan insight mendalam

    **User-Friendly**
    """)
    
    st.sidebar.markdown("---")
    
    # System Info
    st.sidebar.markdown("### 📊 Informasi Sistem")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("**Versi**")
        st.markdown("1.0.0")
    with col2:
        st.markdown("**Status**")
        st.markdown("✅ Active")
    
    st.sidebar.markdown("")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown("**Framework**")
        st.markdown("Streamlit")
    with col2:
        st.markdown("**Algoritma**")
        st.markdown("SVR-PSO-ANOVA-RBF")
    
    st.sidebar.markdown("---")
    
    # Footer Info
    st.sidebar.markdown("""
    ___
    **SI-PADI JATIM v1.0**
    
    Sistem Informasi Prediksi Produksi Padi Jawa Timur
    
    Dibuat oleh: Nur Rohma Widiya Ningsih
    
    © 2026
    """)
    
    # Page Routing dengan professional header
    if page == "Beranda":
        home_page()
    elif page == "Prediksi Cepat":
        input_data_page()
    elif page == "Proses Model SVR-PSO":
        proses_model_svr_pso_page()
    elif page == "Visualisasi":
        visualisasi_page()
    elif page == "Tentang Model":
        tentang_model_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
