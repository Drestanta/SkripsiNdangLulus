-- ============================================================
-- SKEMA BASIS DATA SISTEM PRESENSI BIOMETRIK (presensi_smk.db)
-- ============================================================

-- 1. Tabel Master Referensi Siswa
CREATE TABLE IF NOT EXISTS siswa_referensi (
    id_siswa INTEGER PRIMARY KEY AUTOINCREMENT,
    nisn TEXT UNIQUE NOT NULL,
    nama_siswa TEXT NOT NULL,
    kelas TEXT NOT NULL,
    vektor_wajah TEXT NOT NULL, -- Format JSON: {"ArcFace": [...], "InsightFace": [...], ...}
    tanggal_daftar TEXT NOT NULL
);

-- Indeks untuk mempercepat pencarian NISN
CREATE INDEX IF NOT EXISTS idx_siswa_nisn ON siswa_referensi(nisn);

-- 2. Tabel Riwayat Presensi Resmi Siswa
CREATE TABLE IF NOT EXISTS log_kehadiran (
    id_presensi INTEGER PRIMARY KEY AUTOINCREMENT,
    id_siswa INTEGER NOT NULL,
    nama_siswa TEXT NOT NULL,
    tanggal TEXT NOT NULL,
    waktu_masuk TEXT NOT NULL,
    status TEXT NOT NULL,
    model_digunakan TEXT NOT NULL,
    skor_jarak REAL NOT NULL,
    FOREIGN KEY (id_siswa) REFERENCES siswa_referensi (id_siswa) ON DELETE CASCADE
);

-- Indeks unik untuk mencegah presensi ganda di hari yang sama
CREATE UNIQUE INDEX IF NOT EXISTS idx_presensi_harian ON log_kehadiran(id_siswa, tanggal);

-- 3. Tabel Log Evaluasi Komparasi Multi-Model (Head-to-Head)
CREATE TABLE IF NOT EXISTS log_komparasi_scan (
    id_komparasi INTEGER PRIMARY KEY AUTOINCREMENT,
    tanggal TEXT NOT NULL,
    waktu TEXT NOT NULL,
    mode_masker TEXT NOT NULL,
    siswa_teridentifikasi TEXT NOT NULL,
    arcface_status TEXT NOT NULL,
    arcface_jarak REAL NOT NULL,
    arcface_durasi REAL NOT NULL,
    insight_status TEXT NOT NULL,
    insight_jarak REAL NOT NULL,
    insight_durasi REAL NOT NULL,
    facenet_status TEXT NOT NULL,
    facenet_jarak REAL NOT NULL,
    facenet_durasi REAL NOT NULL
);

-- 4. Injeksi Data Tolok Ukur Empiris Standar Penelitian
INSERT OR IGNORE INTO log_komparasi_scan (
    id_komparasi, tanggal, waktu, mode_masker, siswa_teridentifikasi,
    arcface_status, arcface_jarak, arcface_durasi,
    insight_status, insight_jarak, insight_durasi,
    facenet_status, facenet_jarak, facenet_durasi
) VALUES 
(1, '2026-10-19', '07:15:20', 'Normal (Wajah Terbuka)', 'Budi Santoso', 'Dikenali', 0.2814, 385.0, 'Dikenali', 0.2410, 85.0, 'Dikenali', 6.4210, 290.0),
(2, '2026-10-19', '07:18:45', 'Variasi Hijab', 'Siti Nurhaliza', 'Dikenali', 0.3421, 395.0, 'Dikenali', 0.3120, 90.0, 'Dikenali', 8.8520, 310.0),
(3, '2026-10-19', '07:22:10', 'Variasi Kacamata', 'Rizky Pratama', 'Dikenali', 0.3210, 410.0, 'Dikenali', 0.2950, 88.0, 'Dikenali', 8.1200, 305.0),
(4, '2026-10-19', '07:25:33', 'Bermasker (Periokular)', 'Dewi Lestari', 'Dikenali', 0.3650, 420.0, 'Dikenali', 0.3810, 95.0, 'Ditolak', 10.4500, 315.0),
(5, '2026-10-19', '07:30:15', 'Pencahayaan Rendah', 'Ahmad Dahlan', 'Dikenali', 0.3980, 405.0, 'Dikenali', 0.3540, 89.0, 'Dikenali', 9.2100, 300.0),
(6, '2026-10-19', '07:34:50', 'Uji Impostor (Bukan Siswa)', 'Tidak Dikenali', 'Ditolak', 0.7820, 380.0, 'Ditolak', 0.8120, 82.0, 'Ditolak', 15.2100, 295.0);