import cv2
import numpy as np

def deteksi_penutup_wajah_otomatis(img_bgr):
    """
    Verifikasi Biometrik Anatomis Bertingkat:
    Mendeteksi masker medis, masker kain, dan oklusi tangan tanpa intervensi manual.
    """
    try:
        fh, fw, _ = img_bgr.shape
        if fh < 40 or fw < 40:
            return False, "Resolusi Rendah"

        # Partisi Wilayah Anatomis Wajah
        dahi_roi = img_bgr[int(fh * 0.08):int(fh * 0.28), int(fw * 0.25):int(fw * 0.75)]
        bibir_roi = img_bgr[int(fh * 0.65):int(fh * 0.90), int(fw * 0.25):int(fw * 0.75)]
        bawah_roi = img_bgr[int(fh * 0.45):fh, :]

        ycrcb_dahi = cv2.cvtColor(dahi_roi, cv2.COLOR_BGR2YCrCb)
        ycrcb_bawah = cv2.cvtColor(bawah_roi, cv2.COLOR_BGR2YCrCb)
        ycrcb_bibir = cv2.cvtColor(bibir_roi, cv2.COLOR_BGR2YCrCb)

        # 1. Deteksi Masker Non-Kulit (Medis / Kain Gelap / Putih)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        mask_bawah = cv2.inRange(ycrcb_bawah, lower_skin, upper_skin)
        rasio_kulit_bawah = np.sum(mask_bawah > 0) / mask_bawah.size

        if rasio_kulit_bawah < 0.18:
            return True, "Masker Non-Kulit (Medis/Kain)"

        # 2. Deteksi Oklusi Tangan (Verifikasi Mukosa Bibir Delta Cr & Celah Oral)
        baseline_cr = np.median(ycrcb_dahi[:, :, 1])
        top_cr_bibir = np.percentile(ycrcb_bibir[:, :, 1], 90)
        delta_cr = top_cr_bibir - baseline_cr

        gray_bibir = cv2.cvtColor(bibir_roi, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray_bibir, cv2.CV_64F, 0, 1, ksize=3)
        magnitudo_celah = np.percentile(np.abs(sobel_y), 90)

        # Oklusi tangan terbukti jika Delta Cr sangat rendah DAN celah bibir hilang
        if delta_cr < 3.5 and magnitudo_celah < 7.0:
            return True, "Oklusi Tangan / Penutup Wajah"

        return False, "Wajah Terbuka (Normal)"
    except Exception:
        return False, "Normal"