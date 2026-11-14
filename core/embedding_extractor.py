import time
import numpy as np
import cv2
from deepface import DeepFace

try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

def inisialisasi_insightface():
    """Memuat model InsightFace Buffalo_L via mesin ONNX Runtime"""
    if not HAS_INSIGHTFACE:
        return None
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception:
        return None

def ekstrak_insightface(app_insight, img_bgr):
    """Ekstraksi embedding 512-D InsightFace ONNX dari matriks citra BGR"""
    if app_insight is None:
        return None, 0.0
    start_t = time.time()
    try:
        faces = app_insight.get(img_bgr)
        durasi = (time.time() - start_t) * 1000
        if len(faces) > 0:
            wajah = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = wajah.embedding
            return (emb / np.linalg.norm(emb)), durasi
    except Exception:
        pass
    return None, 0.0

def ekstrak_deepface_model(img_path, model_name):
    """Ekstraksi embedding ArcFace atau FaceNet melalui DeepFace"""
    start_t = time.time()
    res = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False
    )
    durasi = (time.time() - start_t) * 1000
    emb = np.array(res[0]["embedding"], dtype=np.float64)
    return (emb / np.linalg.norm(emb)), durasi