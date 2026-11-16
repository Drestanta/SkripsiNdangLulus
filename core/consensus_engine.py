import numpy as np

def hitung_jarak_cosine(v1, v2):
    """Menghitung Jarak Cosine Teroptimasi dengan Clamping Numerik [-1.0, 1.0]"""
    vec1 = np.array(v1, dtype=np.float64)
    vec2 = np.array(v2, dtype=np.float64)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    dot_product = float(np.clip(np.dot(vec1 / norm1, vec2 / norm2), -1.0, 1.0))
    return max(0.0, min(2.0, float(1.0 - dot_product)))

def hitung_persentase_kemiripan(jarak, threshold):
    """Mengonversi jarak Cosine ke persentase kemiripan terkalibrasi"""
    if jarak <= 0:
        return 100.0
    if jarak <= threshold:
        persen = 100.0 - (jarak / threshold) * 25.0
    else:
        persen = max(0.0, 75.0 - ((jarak - threshold) / (threshold * 0.8)) * 75.0)
    return round(persen, 1)

def ambil_vektor_secara_aman(siswa_record, model_target):
    """Mengambil vektor referensi secara aman dengan auto-fallback multi-generasi"""
    v_dict = siswa_record.get("vektor_dict", {})
    if isinstance(v_dict, dict):
        v = v_dict.get(model_target)
        if v is not None:
            return v
        if "InsightFace" in model_target:
            return v_dict.get("ArcFace")
        if "Facenet512" in model_target:
            return v_dict.get("Facenet")
        if "Facenet" in model_target:
            return v_dict.get("Facenet512")
    return siswa_record.get("vektor")

def evaluasi_konsensus(daftar_siswa, vektor_hasil, ambang_batas, durasi_model, fallback_fn128=None):
    """
    Algoritma Single Identity Consensus Lock (Standar 512-D & Cosine Distance):
    1. Mengunci 1 kandidat identitas siswa terbaik berdasarkan jarak terkecil ArcFace/InsightFace.
    2. Menguji kesepakatan ketiga model terhadap kandidat tunggal tersebut.
    """
    kandidat_terbaik = None
    skor_terkecil = float("inf")

    for s in daftar_siswa:
        v_arc = ambil_vektor_secara_aman(s, "ArcFace")
        v_ins = ambil_vektor_secara_aman(s, "InsightFace")

        j_arc = (
            hitung_jarak_cosine(vektor_hasil["ArcFace"], v_arc)
            if (v_arc is not None and vektor_hasil.get("ArcFace") is not None and len(vektor_hasil["ArcFace"]) == len(v_arc))
            else 999.0
        )
        j_ins = (
            hitung_jarak_cosine(vektor_hasil["InsightFace"], v_ins)
            if (v_ins is not None and vektor_hasil.get("InsightFace") is not None and len(vektor_hasil["InsightFace"]) == len(v_ins))
            else 999.0
        )

        j_min = min(j_arc, j_ins)
        if j_min < skor_terkecil:
            skor_terkecil = j_min
            kandidat_terbaik = s

    hasil_komparasi = {}
    siswa_lolos = None

    if kandidat_terbaik is not None:
        for m_name in ["ArcFace", "InsightFace", "Facenet512"]:
            t_aktif = ambang_batas[m_name]
            v_ref = ambil_vektor_secara_aman(kandidat_terbaik, m_name)
            v_uji = vektor_hasil.get(m_name)

            if v_uji is not None and v_ref is not None and len(v_uji) == len(v_ref):
                jarak_final = hitung_jarak_cosine(v_uji, v_ref)
            elif m_name == "Facenet512" and v_ref is not None and len(v_ref) == 128 and fallback_fn128 is not None:
                v_128 = fallback_fn128()
                jarak_final = hitung_jarak_cosine(v_128, v_ref) if v_128 is not None else 999.0
            else:
                jarak_final = 999.0

            hasil_komparasi[m_name] = {
                "status": "Dikenali" if jarak_final <= t_aktif else "Ditolak",
                "jarak": jarak_final,
                "kemiripan": hitung_persentase_kemiripan(jarak_final, t_aktif),
                "threshold": t_aktif,
                "durasi": durasi_model.get(m_name, 0.0)
            }

        model_sepakat = sum(1 for m in hasil_komparasi.values() if m["status"] == "Dikenali")
        if (
            hasil_komparasi["ArcFace"]["status"] == "Dikenali"
            or hasil_komparasi["InsightFace"]["status"] == "Dikenali"
            or model_sepakat >= 2
        ):
            siswa_lolos = kandidat_terbaik

    return kandidat_terbaik, siswa_lolos, hasil_komparasi