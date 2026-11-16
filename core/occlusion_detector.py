import cv2
import numpy as np

def perbaiki_iluminasi_adaptif(img_bgr):
    """Penyelarasan pencahayaan adaptif menggunakan CLAHE pada ruang warna LAB & koreksi Gamma"""
    try:
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        mean_l, std_l = np.mean(l), np.std(l)

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        if mean_l < 100 or std_l > 60:
            gamma = 0.72
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            l_clahe = cv2.LUT(l_clahe, table)

        return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return img_bgr

def deteksi_penutup_wajah_otomatis(img_bgr):
    """
    Verifikasi Biometrik Anatomis Bertingkat (Zero-Touch):
    1. Disparitas warna kulit YCrCb (masker medis / kain non-kulit).
    2. Verifikasi mukosa bibir (Delta Cr) & gradien celah oral (Sobel-Y) untuk oklusi tangan.
    """
    try:
        if img_bgr is None or img_bgr.size == 0:
            return False

        fh, fw = img_bgr.shape[:2]
        if fh < 30 or fw < 30:
            return False

        upper_face = img_bgr[int(fh * 0.05):int(fh * 0.40), int(fw * 0.20):int(fw * 0.80)]
        lower_face = img_bgr[int(fh * 0.48):int(fh * 0.95), int(fw * 0.15):int(fw * 0.85)]

        if upper_face.size == 0 or lower_face.size == 0:
            return False

        ycrcb_upper = cv2.cvtColor(upper_face, cv2.COLOR_BGR2YCrCb)
        ycrcb_lower = cv2.cvtColor(lower_face, cv2.COLOR_BGR2YCrCb)

        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        ratio_upper = np.sum(cv2.inRange(ycrcb_upper, lower_skin, upper_skin) > 0) / upper_face.shape[0] / upper_face.shape[1]
        ratio_lower = np.sum(cv2.inRange(ycrcb_lower, lower_skin, upper_skin) > 0) / lower_face.shape[0] / lower_face.shape[1]

        # Tahap 1: Deteksi Masker Medis / Kain Non-Kulit
        if ratio_lower < 0.25 or (ratio_upper > 0.15 and ratio_lower < (ratio_upper * 0.50)):
            return True

        # Tahap 2: Deteksi Oklusi Tangan (Cek Mukosa Bibir & Celah Oral)
        lh = lower_face.shape[0]
        mouth_zone = lower_face[int(lh * 0.25):int(lh * 0.80), :]
        if mouth_zone.size == 0:
            return False

        cr_dahi = np.median(ycrcb_upper[:, :, 1])
        cr_mouth = cv2.cvtColor(mouth_zone, cv2.COLOR_BGR2YCrCb)[:, :, 1]
        delta_cr = np.percentile(cr_mouth, 90) - cr_dahi

        gray_mouth = cv2.cvtColor(mouth_zone, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray_mouth, cv2.CV_64F, 0, 1, ksize=3)
        magnitudo_celah = np.percentile(np.abs(sobel_y), 90)

        if delta_cr < 4.0 and magnitudo_celah < 10.0:
            return True

        return False
    except Exception:
        return False