import numpy as np

def hitung_jarak_cosine(v1, v2):
    """Menghitung jarak Cosine: 1 - Cosine Similarity"""
    vec1 = np.array(v1, dtype=np.float64)
    vec2 = np.array(v2, dtype=np.float64)
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 1.0
    return float(1.0 - (dot / norm))

def hitung_jarak_euclidean(v1, v2):
    """Menghitung jarak Euclidean"""
    vec1 = np.array(v1, dtype=np.float64)
    vec2 = np.array(v2, dtype=np.float64)
    return float(np.linalg.norm(vec1 - vec2))

def evaluasi_konsensus(daftar_siswa, vektor_hasil, ambang_batas, mode_aktif="normal"):
    """
    Algoritma Single Identity Consensus Lock:
    Menemukan 1 kandidat utama dan menguji kesepakatan ketiga model terhadap kandidat yang sama.
    """
    kunci_vektor = "ArcFace_periocular" if mode_aktif == "periocular" else "ArcFace"
    kandidat_terbaik = None
    jarak_terkecil = float("inf")

    # 1. Kunci Identitas Utama Berdasarkan ArcFace
    for s in daftar_siswa:
        v_dict = s.get("vektor_dict", {})
        v_ref = v_dict.get(kunci_vektor, v_dict.get("ArcFace"))
        if v_ref is not None and len(vektor_hasil["ArcFace"]) == len(v_ref):
            jarak = hitung_jarak_cosine(vektor_hasil["ArcFace"], v_ref)
            if jarak < jarak_terkecil:
                jarak_terkecil = jarak
                kandidat_terbaik = s

    # 2. Uji Kesepakatan Model terhadap Identitas Terpilih
    hasil_komparasi = {}
    siswa_lolos = None

    if kandidat_terbaik is not None:
        v_dict = kandidat_terbaik.get("vektor_dict", {})
        
        # Evaluasi ArcFace
        j_arc = hitung_jarak_cosine(vektor_hasil["ArcFace"], v_dict.get(kunci_vektor, v_dict.get("ArcFace")))
        hasil_komparasi["ArcFace"] = {"jarak": j_arc, "status": "Dikenali" if j_arc <= ambang_batas["ArcFace"] else "Ditolak"}

        # Evaluasi InsightFace
        k_ins = "InsightFace_periocular" if mode_aktif == "periocular" else "InsightFace"
        v_ins = v_dict.get(k_ins, v_dict.get("InsightFace", v_dict.get("ArcFace")))
        j_ins = hitung_jarak_cosine(vektor_hasil["InsightFace"], v_ins)
        hasil_komparasi["InsightFace"] = {"jarak": j_ins, "status": "Dikenali" if j_ins <= ambang_batas["InsightFace"] else "Ditolak"}

        # Evaluasi FaceNet
        k_fn = "Facenet_periocular" if mode_aktif == "periocular" else "Facenet"
        v_fn = v_dict.get(k_fn, v_dict.get("Facenet"))
        j_fn = hitung_jarak_euclidean(vektor_hasil["Facenet"], v_fn)
        hasil_komparasi["Facenet"] = {"jarak": j_fn, "status": "Dikenali" if j_fn <= ambang_batas["Facenet"] else "Ditolak"}

        # Verifikasi Konsensus: Lolos jika ArcFace setuju atau minimal 2 model sepakat
        total_sepakat = sum(1 for m in hasil_komparasi.values() if m["status"] == "Dikenali")
        if hasil_komparasi["ArcFace"]["status"] == "Dikenali" or total_sepakat >= 2:
            siswa_lolos = kandidat_terbaik

    return kandidat_terbaik, siswa_lolos, hasil_komparasi