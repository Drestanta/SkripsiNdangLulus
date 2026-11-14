#!/bin/bash
# ============================================================
# SKRIP OTOMASI EKSEKUSI SISTEM PRESENSI BIOMETRIK
# ============================================================

echo "=================================================="
echo "  Sistem Presensi Wajah Siswa SMK PGRI Sukoharjo  "
echo "  Komparasi: ArcFace, InsightFace ONNX, FaceNet   "
echo "=================================================="

# 1. Pengecekan Lingkungan Virtual Python
if [ ! -d ".venv" ]; then
    echo "[INFO] Membuat Virtual Environment Python..."
    python3 -m venv .venv
fi

# Aktivasi Virtual Environment
source .venv/bin/activate || source .venv/Scripts/activate

# 2. Pembaruan dan Pemasangan Dependensi
echo "[INFO] Memeriksa paket dependensi..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
pip install insightface onnxruntime > /dev/null 2>&1

# 3. Menjalankan Server Web Streamlit
echo "[INFO] Menjalankan server aplikasi Streamlit..."
echo "Aplikasi dapat diakses pada: http://localhost:8501"
streamlit run app.py