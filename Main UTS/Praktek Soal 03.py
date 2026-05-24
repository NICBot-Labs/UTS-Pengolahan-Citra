import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder_hasil = "image/hasil_kuantisasi_soal_03"
os.makedirs(folder_hasil, exist_ok=True)

# Membaca gambar dalam mode grayscale
img = cv2.imread('image/Soal 03.jpeg', 0)
if img is None:
    print("Gambar tidak ditemukan!")
    exit()
tinggi, lebar = img.shape


# Fungsi kuantisasi
def kuantisasi_manual(img, level):
    tinggi, lebar = img.shape
    # Membuat gambar kosong untuk hasil
    hasil = np.zeros((tinggi, lebar), dtype=np.uint8)

    # Rumus interval untuk membagi kuantitasi level
    interval = 256 / level

    for baris in range(tinggi):
        for kolom in range(lebar):
            piksel = img[baris, kolom]
            # Menentukan kelompok level
            nilai_level = int(piksel / interval)
            # Agar tidak melebihi level terakhir
            if nilai_level >= level:
                nilai_level = level - 1
            # Mengubah kembali ke rentang 0 - 255
            nilai_baru = int(nilai_level * (255 / (level - 1)))

            hasil[baris, kolom] = nilai_baru
    return hasil


# Kuantisasi menjadi beberapa level
img_2_level = kuantisasi_manual(img, 2)
img_4_level = kuantisasi_manual(img, 4)
img_8_level = kuantisasi_manual(img, 8)
img_16_level = kuantisasi_manual(img, 16)

cv2.imwrite(os.path.join(folder_hasil, "kuantisasi_2_level.png"), img_2_level)
cv2.imwrite(os.path.join(folder_hasil, "kuantisasi_4_level.png"), img_4_level)
cv2.imwrite(os.path.join(folder_hasil, "kuantisasi_8_level.png"), img_8_level)
cv2.imwrite(os.path.join(folder_hasil, "kuantisasi_16_level.png"), img_16_level)

# Menampilkan hasil
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Citra Asli Grayscale')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_2_level, cmap='gray')
plt.title('Kuantisasi 2 Level')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_4_level, cmap='gray')
plt.title('Kuantisasi 4 Level')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_8_level, cmap='gray')
plt.title('Kuantisasi 8 Level')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_16_level, cmap='gray')
plt.title('Kuantisasi 16 Level')
plt.axis('off')

plt.tight_layout()
plt.show()


# Menghitung estimasi ukuran data
jumlah_piksel = tinggi * lebar

print()
print("Ukuran citra:", lebar, "x", tinggi)
print("Jumlah piksel:", jumlah_piksel)
print()

# ==============================
# Ukuran citra asli berwarna RGB
# ==============================
bit_per_piksel_rgb = 24
ukuran_bit_rgb = jumlah_piksel * bit_per_piksel_rgb
ukuran_byte_rgb = ukuran_bit_rgb / 8
ukuran_kb_rgb = ukuran_byte_rgb / 1024

print("Citra Asli Berwarna / RGB")
print("Bit per piksel :", bit_per_piksel_rgb, "bit")
print("Ukuran data    :", ukuran_bit_rgb, "bit")
print("Ukuran data    :", ukuran_byte_rgb, "byte")
print("Ukuran data    :", ukuran_kb_rgb, "KB")
print()

# ==============================
# Ukuran citra grayscale asli
# ==============================
bit_per_piksel_gray = 8
ukuran_bit_gray = jumlah_piksel * bit_per_piksel_gray
ukuran_byte_gray = ukuran_bit_gray / 8
ukuran_kb_gray = ukuran_byte_gray / 1024

print("Citra Grayscale Asli")
print("Bit per piksel :", bit_per_piksel_gray, "bit")
print("Ukuran data    :", ukuran_bit_gray, "bit")
print("Ukuran data    :", ukuran_byte_gray, "byte")
print("Ukuran data    :", ukuran_kb_gray, "KB")
print()

# ==============================
# Ukuran hasil kuantisasi
# ==============================
level_list = [2, 4, 8, 16]

for level in level_list:
    bit_per_piksel = int(np.log2(level))
    ukuran_bit = jumlah_piksel * bit_per_piksel
    ukuran_byte = ukuran_bit / 8
    ukuran_kb = ukuran_byte / 1024

    print("Kuantisasi", level, "level")
    print("Bit per piksel :", bit_per_piksel, "bit")
    print("Ukuran data    :", ukuran_bit, "bit")
    print("Ukuran data    :", ukuran_byte, "byte")
    print("Ukuran data    :", ukuran_kb, "KB")
    print()