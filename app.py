"""
SI-PADI JATIM - Sistem Prediksi Produksi Padi Berbasis Machine Learning
Menggunakan SVR dengan PSO Optimization dan Kernel ANOVA RBF
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import joblib
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
                lambda x: f'{x:.2f}' if pd.notna(x) and x % 1 != 0 else f'{int(x)}'
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

def metric_card(label, value, icon="", change=None):
    """Custom metric card"""
    change_html = ""
    if change is not None:
        color = "#2e7d32" if change > 0 else "#d32f2f"
        arrow = "↑" if change > 0 else "↓"
        change_html = f'<span style="color: {color}; font-size: 0.9rem; font-weight: 600;">{arrow} {abs(change):.1f}%</span>'
    
    return f"""
    <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                padding: 24px; border-radius: 12px; border-left: 5px solid #2e7d32;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;">
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
    
    # List semua kabupaten/kota di Jawa Timur (34 total)
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
    if kabupaten in all_locations:
        idx = all_locations.index(kabupaten)
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

def load_pretrained_model():
    """Load model SVR yang sudah dilatih"""
    model_path = "model_svr_rbf_90_30_partikel_100_iterasi.save"
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
    st.markdown("### 💡 Manfaat Sistem")
    
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
    st.markdown("### 🚀 Mulai Menggunakan dalam 4 Langkah")
    
    steps = [
        ("� Data Masukan", " Input manual data iklim & lahan produksi padi"),
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
    st.markdown("### ✨ Fitur Unggulan")
    
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
    custom_header("📝 Input Data & Prediksi", "Input data iklim dan lahan untuk prediksi produksi padi")
    
    tab1, tab2 = st.tabs(["🎯 Prediksi Cepat", "📋 Batch Input"])
    
    # Tab 1: Prediksi Manual Cepat dengan Model
    with tab1:
        st.markdown("### 🔮 Prediksi Produksi Padi dengan Model SVR")
        
        # Load model
        model = load_pretrained_model()
        if model is None:
            st.error("❌ Model tidak ditemukan: model_svr_rbf_90_30_partikel_100_iterasi.save")
            st.stop()
        
        st.markdown("""
        Masukkan data iklim dan lahan untuk mendapatkan prediksi produksi padi menggunakan 
        model SVR-ANOVA-RBF yang telah dilatih dengan data 34 Kabupaten/Kota di Jawa Timur.
        """)
        
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
                        <p style="margin: 15px 0 0 0; font-size: 2.8rem; font-weight: 900; color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{produksi_prediksi:,.0f}</p>
                        <p style="margin: 5px 0 0 0; font-size: 1rem; font-weight: 600; color: #ffffff !important;">ton</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        produktivitas = produksi_prediksi / luas_panen * 100 if luas_panen > 0 else 0
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #fbc02d, #f9a825); 
                                    border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        <div style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #000000 !important; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">📊 Produktivitas</div>
                        <p style="margin: 15px 0 0 0; font-size: 2.8rem; font-weight: 900; color: #000000 !important; text-shadow: 1px 1px 2px rgba(255,255,255,0.3);">{produktivitas:,.1f}</p>
                        <p style="margin: 5px 0 0 0; font-size: 1rem; font-weight: 600; color: #000000 !important;">ku/ha</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="padding: 20px; background: linear-gradient(135deg, #1976d2, #1565c0); 
                                    border-radius: 10px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                        <div style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #ffffff !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">🌾 Luas Panen</div>
                        <p style="margin: 15px 0 0 0; font-size: 2.8rem; font-weight: 900; color: #ffffff !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{luas_panen:,.0f}</p>
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
                        st.write(f"**Curah Hujan**: {curah_hujan:.1f} mm")
                        st.write(f"**Kelembapan**: {kelembapan:.1f} %")
                        st.write(f"**Suhu**: {suhu:.1f} °C")
                    
                    with detail_cols[2]:
                        st.write(f"**Kecepatan Angin**: {kecepatan_angin:.1f} m/s")
                        st.write(f"**Sinar Matahari**: {sinar_matahari:.1f} jam/hari")
                    
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
        
        # Load model SVR terbaik
        model_svr = load_pretrained_model()
        if model_svr is None:
            st.error("❌ Model SVR tidak ditemukan: model_svr_rbf_90_30_partikel_100_iterasi.save")
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
                df_display['Luas_Panen'] = df_display['Luas_Panen'].apply(lambda x: f"{x:,.0f} ha")
                df_display['Curah_Hujan'] = df_display['Curah_Hujan'].apply(lambda x: f"{x:.1f} mm")
                df_display['Kelembapan'] = df_display['Kelembapan'].apply(lambda x: f"{x:.1f} %")
                df_display['Suhu'] = df_display['Suhu'].apply(lambda x: f"{x:.1f} °C")
                df_display['Kecepatan_Angin'] = df_display['Kecepatan_Angin'].apply(lambda x: f"{x:.1f} m/s")
                df_display['Sinar_Matahari'] = df_display['Sinar_Matahari'].apply(lambda x: f"{x:.1f} jam")
                
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
                df_result_display['Luas_Panen'] = df_result_display['Luas_Panen'].apply(lambda x: f"{x:,.0f} ha")
                df_result_display['Curah_Hujan'] = df_result_display['Curah_Hujan'].apply(lambda x: f"{x:.1f} mm")
                df_result_display['Kelembapan'] = df_result_display['Kelembapan'].apply(lambda x: f"{x:.1f} %")
                df_result_display['Suhu'] = df_result_display['Suhu'].apply(lambda x: f"{x:.1f} °C")
                df_result_display['Kecepatan_Angin'] = df_result_display['Kecepatan_Angin'].apply(lambda x: f"{x:.1f} m/s")
                df_result_display['Sinar_Matahari'] = df_result_display['Sinar_Matahari'].apply(lambda x: f"{x:.1f} jam")
                df_result_display['Produksi_Prediksi_Ton'] = df_result_display['Produksi_Prediksi_Ton'].apply(lambda x: f"{x:,.0f} ton")
                
                st.dataframe(df_result_display, use_container_width=True, hide_index=True)
                
                # Statistics
                st.markdown("#### 📈 Statistik Hasil Prediksi")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    total_produksi = df_results['Produksi_Prediksi_Ton'].sum()
                    st.metric("📦 Total Produksi", f"{total_produksi:,.0f} ton")
                
                with col_stat2:
                    rata_rata_produksi = df_results['Produksi_Prediksi_Ton'].mean()
                    st.metric("📊 Rata-rata", f"{rata_rata_produksi:,.0f} ton")
                
                with col_stat3:
                    max_produksi = df_results['Produksi_Prediksi_Ton'].max()
                    st.metric("📈 Maksimal", f"{max_produksi:,.0f} ton")
                
                with col_stat4:
                    min_produksi = df_results['Produksi_Prediksi_Ton'].min()
                    st.metric("📉 Minimal", f"{min_produksi:,.0f} ton")
                
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
            st.metric("🌾 Produksi", f"{df['Produksi'].sum():,.0f} ton")
        else:
            st.metric("⚠️ Produksi", "-")
    with col_stat4:
        if 'Luas Panen' in df.columns:
            st.metric("🌾 Luas", f"{df['Luas Panen'].sum():,.0f} ha")
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
                                <div style="color: #1b5e20; font-size: 24px; font-weight: bold;">{monthly.sum():,.0f} ton</div>
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
                                   f'{int(height):,}', ha='center', va='bottom', 
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
                            st.metric("📈 Produksi Tertinggi", f"{monthly.max():,.0f} ton", 
                                     delta=f"Bulan {bulan_names[int(monthly.idxmax())-1]}")
                        with col_stat2:
                            st.metric("📉 Produksi Terendah", f"{monthly.min():,.0f} ton",
                                     delta=f"Bulan {bulan_names[int(monthly.idxmin())-1]}")
                        with col_stat3:
                            st.metric("📊 Rata-rata", f"{monthly.mean():,.0f} ton")
                        with col_stat4:
                            selisih = monthly.max() - monthly.min()
                            st.metric("📌 Selisih", f"{selisih:,.0f} ton")
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
                        st.metric("📈 Nilai Tertinggi", f"{monthly.max():,.0f} ton")
                    
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
                            st.metric(f"#{idx+1}", display_name, f"{val:,.0f} ton")
                    
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
                               f' {int(val):,} ton', ha='left', va='center', 
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
                                <p style="margin: 5px 0 2px 0; font-size: 11px;"><b>Rata:</b> {mean_val:.2f}</p>
                                <p style="margin: 2px 0 2px 0; font-size: 11px;"><b>StdDev:</b> {std_val:.2f}</p>
                                <p style="margin: 2px 0 2px 0; font-size: 11px;"><b>Min:</b> {min_val:.2f}</p>
                                <p style="margin: 2px 0 0 0; font-size: 11px;"><b>Max:</b> {max_val:.2f}</p>
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
                axes[0].axvline(df[col_dist].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col_dist].mean():.2f}')
                axes[0].axvline(df[col_dist].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df[col_dist].median():.2f}')
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
                    text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
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
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: 900;">{yearly_stats['Total Produksi'].sum():,.0f}</p>
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
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: 900;">{growth_arrow} {abs(growth_rate):.1f}%</p>
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
            st.metric("✅ Komplit", f"{completion_rate:.1f}%")
        
        with col_dq3:
            st.metric("📊 Unique Values", df.nunique().sum())
        
        with col_dq4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("📈 Memory", f"{memory_mb:.2f} MB")

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


# ==================== MAIN APP ====================

def main():
    """Main application dengan desain profesional"""
    
    # Konfigurasi layout
    st.set_page_config(
        page_title="SI-PADI JATIM",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
        ["Beranda", "Input Data", 
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
    elif page == "Input Data":
        input_data_page()
    elif page == "Visualisasi":
        visualisasi_page()
    elif page == "Tentang Model":
        tentang_model_page()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    main()
