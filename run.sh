#!/bin/bash
cd "$(dirname "$0")" || exit 1

echo "======================================================"
echo "  Sistem Presensi Wajah Siswa SMK PGRI Sukoharjo      "
echo "  Komparasi: ArcFace, InsightFace ONNX, FaceNet-512   "
echo "======================================================"

if [ ! -d ".venv" ]; then
    echo "[INFO] Membuat Virtual Environment Python (.venv)..."
    python3 -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
elif [ -f ".venv/Scripts/activate" ]; then
    source ".venv/Scripts/activate"
fi

echo "[INFO] Memeriksa kesiapan pustaka AI..."
if ! python -c "import streamlit, deepface, insightface, cv2, scipy" > /dev/null 2>&1; then
    echo "[INFO] Memasang paket dari requirements.txt..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
else
    echo "[OK] Seluruh pustaka AI siap digunakan."
fi

echo "[INFO] Menjalankan aplikasi pada: http://localhost:8501"
python -m streamlit run app.py