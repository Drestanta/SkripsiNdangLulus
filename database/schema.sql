-- ============================================================
-- SKEMA BASIS DATA SISTEM PRESENSI BIOMETRIK (presensi_smk.db)
-- ============================================================

CREATE TABLE IF NOT EXISTS siswa_referensi (
    id_siswa INTEGER PRIMARY KEY AUTOINCREMENT,
    nisn TEXT UNIQUE NOT NULL,
    nama_siswa TEXT NOT NULL,
    kelas TEXT NOT NULL,
    vektor_wajah TEXT NOT NULL,
    tanggal_daftar TEXT NOT NULL,
    foto_blob BLOB
);

CREATE INDEX IF NOT EXISTS idx_siswa_nisn ON siswa_referensi(nisn);

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

CREATE UNIQUE INDEX IF NOT EXISTS idx_presensi_harian ON log_kehadiran(id_siswa, tanggal);

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

CREATE TABLE IF NOT EXISTS log_benchmark_penelitian (
    id_bench INTEGER PRIMARY KEY AUTOINCREMENT,
    nama_dataset TEXT NOT NULL,
    sumber_literatur TEXT NOT NULL,
    kondisi_evaluasi TEXT NOT NULL,
    insight_acc REAL,
    arcface_acc REAL,
    facenet_acc REAL
);

INSERT OR IGNORE INTO log_komparasi_scan (
    id_komparasi, tanggal, waktu, mode_masker, siswa_teridentifikasi,
    arcface_status, arcface_jarak, arcface_durasi,
    insight_status, insight_jarak, insight_durasi,
    facenet_status, facenet_jarak, facenet_durasi
) VALUES 
(1, '2026-10-19', '07:15:20', 'Normal (Frontal, Terang)', 'Budi Santoso', 'Dikenali', 0.2814, 385.0, 'Dikenali', 0.2410, 85.0, 'Dikenali', 0.3120, 290.0),
(2, '2026-10-19', '07:18:45', 'Pencahayaan Redup (<100 lux)', 'Siti Nurhaliza', 'Dikenali', 0.3320, 395.0, 'Dikenali', 0.3010, 90.0, 'Dikenali', 0.3750, 310.0),
(3, '2026-10-19', '07:22:10', 'Variasi Kacamata', 'Rizky Pratama', 'Dikenali', 0.3210, 410.0, 'Dikenali', 0.2950, 88.0, 'Dikenali', 0.3420, 305.0),
(4, '2026-10-19', '07:25:33', 'Wajah Bermasker Medis/Kain', 'Dewi Lestari', 'Dikenali', 0.3650, 420.0, 'Dikenali', 0.3810, 95.0, 'Dikenali', 0.4720, 315.0),
(5, '2026-10-19', '07:30:15', 'Pencahayaan Silau (Backlight)', 'Ahmad Dahlan', 'Dikenali', 0.3980, 405.0, 'Dikenali', 0.3540, 89.0, 'Dikenali', 0.3950, 300.0),
(6, '2026-10-19', '07:34:50', 'Oklusi Tangan (Hand-to-Face)', 'Fajar Nugroho', 'Dikenali', 0.4120, 415.0, 'Dikenali', 0.3940, 93.0, 'Dikenali', 0.4890, 312.0);

INSERT OR IGNORE INTO log_benchmark_penelitian (
    id_bench, nama_dataset, sumber_literatur, kondisi_evaluasi, insight_acc, arcface_acc, facenet_acc
) VALUES
(1, 'LFW Benchmark', 'Huang et al. (2007)', 'Wajah Normal & Kacamata', 99.83, 99.53, 99.20),
(2, 'RMFD Benchmark', 'Wang et al. (2020)', 'Wajah Bermasker Nyata', 97.40, 96.20, 91.50),
(3, 'MS1MV2 / Glint360k', 'Deng et al. (2019) / Deng et al. (2022)', 'Pelatihan Skala Global', 99.80, 99.60, 98.70),
(4, 'WIDER FACE (Hard)', 'Yang et al. (2016)', 'Deteksi Wajah Teroklusi', 94.20, 91.50, 85.00);