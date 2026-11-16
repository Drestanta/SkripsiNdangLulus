import os
import time
import tempfile
import numpy as np
import cv2
from deepface import DeepFace

try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False

def inisialisasi_insightface():
    """Memuat model InsightFace Buffalo_L via mesin ONNX Runtime CPU"""
    if not HAS_INSIGHTFACE:
        return None
    try:
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception:
        return None

def deteksi_wajah_dan_insightface(img_bgr, app_insight=None):
    """
    Deteksi Wajah Multi-Backend & Ekstraksi InsightFace dalam 1x Jalan (Single-Pass):
    1. Prioritas Utama: SCRFD (mengembalikan crop wajah + vektor 512-D InsightFace sekaligus).
    2. Fallback: MTCNN -> Haar Cascade (jika SCRFD tidak aktif).
    """
    h_img, w_img = img_bgr.shape[:2]

    # 1. SCRFD + Recognition (InsightFace ONNX)
    if app_insight is not None:
        try:
            start_t = time.perf_counter()
            faces = app_insight.get(img_bgr)
            durasi_ins = (time.perf_counter() - start_t) * 1000
            if len(faces) > 0:
                wajah = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x1, y1, x2, y2 = wajah.bbox.astype(int)
                pad_w, pad_h = int((x2 - x1) * 0.10), int((y2 - y1) * 0.10)
                x = max(0, x1 - pad_w)
                y = max(0, y1 - pad_h)
                w = min(w_img - x, (x2 - x1) + (2 * pad_w))
                h = min(h_img - y, (y2 - y1) + (2 * pad_h))

                emb = np.array(wajah.embedding, dtype=np.float64)
                norm = np.linalg.norm(emb)
                emb_norm = (emb / norm) if norm != 0 else None
                return img_bgr[y:y+h, x:x+w], emb_norm, durasi_ins, True
        except Exception:
            pass

    # 2. MTCNN Fallback
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f_temp:
            temp_t = f_temp.name
        cv2.imwrite(temp_t, img_bgr)
        face_objs = DeepFace.extract_faces(img_path=temp_t, detector_backend="mtcnn", enforce_detection=False)
        if os.path.exists(temp_t):
            os.remove(temp_t)

        if len(face_objs) > 0:
            fa = max(face_objs, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])["facial_area"]
            if not (fa['w'] >= w_img * 0.95 and fa['h'] >= h_img * 0.95):
                x, y, w, h = max(0, fa['x']), max(0, fa['y']), fa['w'], fa['h']
                return img_bgr[y:y+h, x:x+w], None, 0.0, True
    except Exception:
        pass

    # 3. Haar Cascade Fallback
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
        if len(h_faces) > 0:
            x, y, w, h = max(h_faces, key=lambda f: f[2] * f[3])
            return img_bgr[y:y+h, x:x+w], None, 0.0, True
    except Exception:
        pass

    return None, None, 0.0, False

def ekstrak_deepface_model(img_path, model_name="ArcFace"):
    """Ekstraksi embedding ArcFace atau FaceNet-512 ternormalisasi L2 via DeepFace"""
    start_t = time.perf_counter()
    try:
        res = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend="skip",
            enforce_detection=False
        )
        durasi = (time.perf_counter() - start_t) * 1000
        if res and len(res) > 0:
            emb = np.array(res[0]["embedding"], dtype=np.float64)
            norm = np.linalg.norm(emb)
            return ((emb / norm) if norm != 0 else None), durasi
    except Exception:
        pass
    return None, 0.0