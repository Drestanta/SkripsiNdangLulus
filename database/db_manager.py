import sqlite3
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Mengunci lokasi database di folder utama proyek
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(BASE_DIR, "presensi_smk.db")

def get_connection():
    """Membuka koneksi ke basis data relasional SQLite"""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Inisialisasi tabel basis data relasional"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Tabel Referensi Siswa (Menyimpan Vektor JSON dan FOTO BLOB langsung di database)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS siswa_referensi (
            id_siswa INTEGER PRIMARY KEY AUTOINCREMENT,
            nisn TEXT UNIQUE NOT NULL,
            nama_siswa TEXT NOT NULL,
            kelas TEXT NOT NULL,
            vektor_wajah TEXT NOT NULL,
            tanggal_daftar TEXT NOT NULL,
            foto_blob BLOB
        )
    ''')
    
    # Auto-migration: Cek jika kolom foto_blob belum ada pada tabel lama
    cursor.execute("PRAGMA table_info(siswa_referensi)")
    kolom_ada = [info[1] for info in cursor.fetchall()]
    if "foto_blob" not in kolom_ada:
        cursor.execute("ALTER TABLE siswa_referensi ADD COLUMN foto_blob BLOB")

    # 2. Tabel Log Kehadiran Resmi Siswa
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS log_kehadiran (
            id_presensi INTEGER PRIMARY KEY AUTOINCREMENT,
            id_siswa INTEGER,
            nama_siswa TEXT NOT NULL,
            tanggal TEXT NOT NULL,
            waktu_masuk TEXT NOT NULL,
            status TEXT NOT NULL,
            model_digunakan TEXT NOT NULL,
            skor_jarak REAL,
            FOREIGN KEY (id_siswa) REFERENCES siswa_referensi (id_siswa) ON DELETE CASCADE
        )
    ''')

    # 3. Tabel Log Evaluasi Komparasi Multi-Model
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS log_komparasi_scan (
            id_komparasi INTEGER PRIMARY KEY AUTOINCREMENT,
            tanggal TEXT NOT NULL,
            waktu TEXT NOT NULL,
            mode_masker TEXT NOT NULL,
            siswa_teridentifikasi TEXT NOT NULL,
            arcface_status TEXT,
            arcface_jarak REAL,
            arcface_durasi REAL,
            insight_status TEXT,
            insight_jarak REAL,
            insight_durasi REAL,
            facenet_status TEXT,
            facenet_jarak REAL,
            facenet_durasi REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def simpan_siswa_multi_model(nisn, nama_siswa, kelas, vektor_dict, foto_bytes=None):
    """Menyimpan data siswa, vektor 3 model, dan FOTO BLOB langsung ke file presensi_smk.db"""
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
        sukses = True
        pesan = f"Siswa {nama_siswa} (NISN: {nisn}) berhasil disimpan ke dalam database!"
    except sqlite3.IntegrityError:
        sukses = False
        pesan = f"Error: NISN {nisn} sudah terdaftar di database."
    finally:
        conn.close()
        
    return sukses, pesan

def ambil_semua_siswa():
    """Mengambil semua data siswa beserta vektor dan foto langsung dari database"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id_siswa, nisn, nama_siswa, kelas, vektor_wajah, foto_blob FROM siswa_referensi')
    rows = cursor.fetchall()
    conn.close()
    
    daftar_siswa = []
    for r in rows:
        raw_vektor = json.loads(r[4])
        if isinstance(raw_vektor, dict):
            vektor_dict = {k: np.array(v) for k, v in raw_vektor.items()}
        else:
            vektor_dict = {"ArcFace": np.array(raw_vektor)}
            
        daftar_siswa.append({
            "id_siswa": r[0],
            "nisn": r[1],
            "nama_siswa": r[2],
            "kelas": r[3],
            "vektor_dict": vektor_dict,
            "foto_blob": r[5]  # Foto biner yang tersimpan di dalam database
        })
    return daftar_siswa

def catat_presensi(id_siswa, nama_siswa, model="ArcFace", skor_jarak=0.0):
    conn = get_connection()
    cursor = conn.cursor()
    sekarang = datetime.now()
    tanggal_hari_ini = sekarang.strftime("%Y-%m-%d")
    waktu_sekarang = sekarang.strftime("%H:%M:%S")
    
    cursor.execute('''
        SELECT id_presensi FROM log_kehadiran 
        WHERE id_siswa = ? AND tanggal = ?
    ''', (id_siswa, tanggal_hari_ini))
    
    if cursor.fetchone():
        conn.close()
        return False, f"{nama_siswa} sudah melakukan presensi hari ini!"
    
    cursor.execute('''
        INSERT INTO log_kehadiran (id_siswa, nama_siswa, tanggal, waktu_masuk, status, model_digunakan, skor_jarak)
        VALUES (?, ?, ?, ?, 'Hadir', ?, ?)
    ''', (id_siswa, nama_siswa, tanggal_hari_ini, waktu_sekarang, model, skor_jarak))
    conn.commit()
    conn.close()
    return True, f"Presensi berhasil dicatat: {nama_siswa} ({waktu_sekarang})"

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

def isi_data_sampel_komparasi():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM log_komparasi_scan")
    if cursor.fetchone()[0] == 0:
        data_sampel = [
            ("2026-10-19", "07:15:20", "Normal (Frontal)", "Budi Santoso", "Dikenali", 0.2814, 385.0, "Dikenali", 0.2410, 85.0, "Dikenali", 0.3120, 290.0),
            ("2026-10-19", "07:18:45", "Variasi Hijab", "Siti Nurhaliza", "Dikenali", 0.3421, 395.0, "Dikenali", 0.3120, 90.0, "Dikenali", 0.3850, 310.0),
            ("2026-10-19", "07:22:10", "Variasi Kacamata", "Rizky Pratama", "Dikenali", 0.3210, 410.0, "Dikenali", 0.2950, 88.0, "Dikenali", 0.3420, 305.0),
            ("2026-10-19", "07:25:33", "Bermasker (Periokular)", "Dewi Lestari", "Dikenali", 0.3650, 420.0, "Dikenali", 0.3810, 95.0, "Ditolak", 0.5420, 315.0),
            ("2026-10-19", "07:30:15", "Pencahayaan Rendah", "Ahmad Dahlan", "Dikenali", 0.3980, 405.0, "Dikenali", 0.3540, 89.0, "Dikenali", 0.3950, 300.0),
            ("2026-10-19", "07:34:50", "Uji Impostor", "Tidak Dikenali", "Ditolak", 0.7820, 380.0, "Ditolak", 0.8120, 82.0, "Ditolak", 0.8410, 295.0)
        ]
        cursor.executemany('''
            INSERT INTO log_komparasi_scan (
                tanggal, waktu, mode_masker, siswa_teridentifikasi,
                arcface_status, arcface_jarak, arcface_durasi,
                insight_status, insight_jarak, insight_durasi,
                facenet_status, facenet_jarak, facenet_durasi
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_sampel)
        conn.commit()
    conn.close()

def hapus_siswa(id_siswa):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM log_kehadiran WHERE id_siswa = ?', (id_siswa,))
    cursor.execute('DELETE FROM siswa_referensi WHERE id_siswa = ?', (id_siswa,))
    conn.commit()
    conn.close()
    return True