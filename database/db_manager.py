import sqlite3
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "presensi_smk.db")
SCHEMA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

def get_connection():
    """Membuka koneksi ke basis data relasional SQLite"""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Inisialisasi tabel dan data tolok ukur dari schema.sql + auto-migration"""
    conn = get_connection()
    cursor = conn.cursor()

    if os.path.exists(SCHEMA_FILE):
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            conn.executescript(f.read())

    # Auto-migration jika file DB lama belum memiliki kolom foto_blob
    cursor.execute("PRAGMA table_info(siswa_referensi)")
    kolom_ada = [info[1] for info in cursor.fetchall()]
    if "foto_blob" not in kolom_ada:
        cursor.execute("ALTER TABLE siswa_referensi ADD COLUMN foto_blob BLOB")

    # Bersihkan otomatis baris sampel lama jika masih memuat Hijab / Impostor
    cursor.execute("SELECT COUNT(*) FROM log_komparasi_scan WHERE mode_masker LIKE '%Hijab%' OR mode_masker LIKE '%Impostor%'")
    if cursor.fetchone()[0] > 0:
        cursor.execute("DELETE FROM log_komparasi_scan WHERE id_komparasi <= 6 AND (mode_masker LIKE '%Hijab%' OR mode_masker LIKE '%Impostor%')")

    conn.commit()
    conn.close()

def simpan_siswa_multi_model(nisn, nama_siswa, kelas, vektor_dict, foto_bytes=None):
    """Menyimpan data siswa, kamus vektor 512-D, dan foto BLOB"""
    conn = get_connection()
    cursor = conn.cursor()
    vektor_serial = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in vektor_dict.items()}
    vektor_str = json.dumps(vektor_serial)
    tanggal_daftar = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cursor.execute('''
            INSERT INTO siswa_referensi (nisn, nama_siswa, kelas, vektor_wajah, tanggal_daftar, foto_blob)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (nisn, nama_siswa, kelas, vektor_str, tanggal_daftar, foto_bytes))
        conn.commit()
        return True, f"Siswa {nama_siswa} (NISN: {nisn}) berhasil disimpan ke database!"
    except sqlite3.IntegrityError:
        return False, f"Error: NISN {nisn} sudah terdaftar di database."
    finally:
        conn.close()

def ambil_semua_siswa():
    """Mengambil seluruh siswa terdaftar beserta vektor dan foto BLOB"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id_siswa, nisn, nama_siswa, kelas, vektor_wajah, foto_blob FROM siswa_referensi')
    rows = cursor.fetchall()
    conn.close()

    daftar = []
    for r in rows:
        raw_v = json.loads(r[4])
        vektor_dict = {k: np.array(v) for k, v in raw_v.items()} if isinstance(raw_v, dict) else {"ArcFace": np.array(raw_v)}
        daftar.append({
            "id_siswa": r[0], "nisn": r[1], "nama_siswa": r[2],
            "kelas": r[3], "vektor_dict": vektor_dict, "foto_blob": r[5]
        })
    return daftar

def catat_presensi(id_siswa, nama_siswa, model="InsightFace", skor_jarak=0.0):
    """Mencatat kehadiran resmi harian dengan pencegahan duplikasi"""
    conn = get_connection()
    cursor = conn.cursor()
    sekarang = datetime.now()
    tgl = sekarang.strftime("%Y-%m-%d")
    jam = sekarang.strftime("%H:%M:%S")

    cursor.execute('SELECT id_presensi FROM log_kehadiran WHERE id_siswa = ? AND tanggal = ?', (id_siswa, tgl))
    if cursor.fetchone():
        conn.close()
        return False, f"{nama_siswa} sudah melakukan presensi hari ini!"

    cursor.execute('''
        INSERT INTO log_kehadiran (id_siswa, nama_siswa, tanggal, waktu_masuk, status, model_digunakan, skor_jarak)
        VALUES (?, ?, ?, ?, 'Hadir', ?, ?)
    ''', (id_siswa, nama_siswa, tgl, jam, model, skor_jarak))
    conn.commit()
    conn.close()
    return True, f"Presensi berhasil dicatat: {nama_siswa} ({jam})"

def ambil_log_presensi_df():
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT id_presensi as "ID", nama_siswa as "Nama Siswa", tanggal as "Tanggal", 
               waktu_masuk as "Jam Masuk", status as "Status", 
               model_digunakan as "Model Penentu", ROUND(skor_jarak, 4) as "Skor Jarak"
        FROM log_kehadiran ORDER BY id_presensi DESC
    ''', conn)
    conn.close()
    return df

def ambil_log_komparasi_df():
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT id_komparasi as "ID", tanggal as "Tanggal", waktu as "Waktu", 
               mode_masker as "Kondisi", siswa_teridentifikasi as "Siswa Teridentifikasi",
               arcface_status as "ArcFace Status", ROUND(arcface_jarak, 4) as "ArcFace Jarak", ROUND(arcface_durasi, 1) as "ArcFace ms",
               insight_status as "InsightFace Status", ROUND(insight_jarak, 4) as "InsightFace Jarak", ROUND(insight_durasi, 1) as "InsightFace ms",
               facenet_status as "FaceNet-512 Status", ROUND(facenet_jarak, 4) as "FaceNet-512 Jarak", ROUND(facenet_durasi, 1) as "FaceNet-512 ms"
        FROM log_komparasi_scan ORDER BY id_komparasi ASC
    ''', conn)
    conn.close()
    return df

def ambil_log_benchmark_df():
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT id_bench as "ID", nama_dataset as "Dataset Baseline Literatur", 
               sumber_literatur as "Rujukan Ilmiah", kondisi_evaluasi as "Fokus Pengujian",
               ROUND(insight_acc, 2) as "InsightFace (%)", 
               ROUND(arcface_acc, 2) as "ArcFace (%)", 
               ROUND(facenet_acc, 2) as "FaceNet-512 (%)"
        FROM log_benchmark_penelitian ORDER BY id_bench ASC
    ''', conn)
    conn.close()
    return df

def simpan_log_komparasi(data_log):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO log_komparasi_scan (
            tanggal, waktu, mode_masker, siswa_teridentifikasi,
            arcface_status, arcface_jarak, arcface_durasi,
            insight_status, insight_jarak, insight_durasi,
            facenet_status, facenet_jarak, facenet_durasi
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data_log["tanggal"], data_log["waktu"], data_log["mode_masker"], data_log["siswa_teridentifikasi"],
        data_log["arcface_status"], data_log["arcface_jarak"], data_log["arcface_durasi"],
        data_log["insight_status"], data_log["insight_jarak"], data_log["insight_durasi"],
        data_log["facenet_status"], data_log["facenet_jarak"], data_log["facenet_durasi"]
    ))
    conn.commit()
    conn.close()

def ubah_kelas_siswa(id_siswa, kelas_baru):
    """Memperbarui tingkat kelas siswa pada tabel siswa_referensi"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        'UPDATE siswa_referensi SET kelas = ? WHERE id_siswa = ?',
        (kelas_baru, id_siswa)
    )
    conn.commit()
    berhasil = cursor.rowcount > 0
    conn.close()
    return berhasil

def hapus_siswa(id_siswa):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM log_kehadiran WHERE id_siswa = ?', (id_siswa,))
    cursor.execute('DELETE FROM siswa_referensi WHERE id_siswa = ?', (id_siswa,))
    conn.commit()
    conn.close()
    return True