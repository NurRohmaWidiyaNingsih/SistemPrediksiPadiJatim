# 🌾 SI-PADI JATIM
Sistem Prediksi Produksi Padi Berbasis Machine Learning

## 📋 Requirements
- Python 3.8+
- Streamlit
- scikit-learn
- pandas, numpy, matplotlib

## 🚀 Instalasi & Menjalankan

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

Akses di: `http://localhost:8501`

## 📁 Struktur Project
```
SI-PADI JATIM/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── README.md             # Documentation
└── models/
    ├── svr_pso_model.py  # SVR + PSO Model
    └── svr_model.py      # Basic SVR
```

## 🎯 Fitur
- Input data manual atau upload CSV
- Prediksi dengan SVR-PSO-RBF
- Visualisasi data dan hasil
- Model explanation

