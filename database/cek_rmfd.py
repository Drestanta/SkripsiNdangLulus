import os
import cv2

ARCHIVE_DIR = os.path.join("database", "archive")

jumlah_folder_orang = 0
jumlah_foto = 0
contoh_foto = []

for root, dirs, files in os.walk(ARCHIVE_DIR):
    foto_di_folder = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(foto_di_folder) > 0:
        jumlah_folder_orang += 1
        jumlah_foto += len(foto_di_folder)
        if len(contoh_foto) < 5:
            contoh_foto.append(os.path.join(root, foto_di_folder[0]))

print(f"Total Subfolder Identitas : {jumlah_folder_orang}")
print(f"Total File Foto Terbaca   : {jumlah_foto}")
print("\nContoh 5 file foto yang terbaca:")
for p in contoh_foto:
    img = cv2.imread(p)
    resolusi = f"{img.shape[1]}x{img.shape[0]}" if img is not None else "Gagal baca"
    print(f" - {p} ({resolusi})")