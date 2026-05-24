import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder_hasil = "image/hasil_konversi_soal_02"
os.makedirs(folder_hasil, exist_ok=True)


# Membaca gambar berwarna
img_bgr = cv2.imread('image/Soal 02.jpg')
if img_bgr is None:
    print("Gambar tidak ditemukan!")
    exit()

print("RUN pengolahan citra")

tinggi, lebar, channel = img_bgr.shape

# 1. Konversi BGR ke RGB
img_rgb = img_bgr[:, :, ::-1]

# 2. Konversi RGB ke Grayscale
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

img_gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# 3. Konversi Grayscale ke Biner
threshold = 128
img_biner = np.where(img_gray >= threshold, 255, 0).astype(np.uint8)


# Menampilkan ukuran array
print()
print("Ukuran Array")
print("RGB       :", img_rgb.shape)
print("Grayscale :", img_gray.shape)
print("Biner     :", img_biner.shape)

# Menghitung kebutuhan memori
jumlah_piksel = tinggi * lebar

bit_rgb = 24
bit_gray = 8
bit_biner = 1

ukuran_rgb_bit = jumlah_piksel * bit_rgb
ukuran_gray_bit = jumlah_piksel * bit_gray
ukuran_biner_bit = jumlah_piksel * bit_biner

ukuran_rgb_byte = ukuran_rgb_bit / 8
ukuran_gray_byte = ukuran_gray_bit / 8
ukuran_biner_byte = ukuran_biner_bit / 8

ukuran_rgb_kb = ukuran_rgb_byte / 1024
ukuran_gray_kb = ukuran_gray_byte / 1024
ukuran_biner_kb = ukuran_biner_byte / 1024

print()
print("Ukuran Citra:", lebar, "x", tinggi)
print("Jumlah Piksel:", jumlah_piksel)
print()

print("Citra RGB")
print("Bit per piksel :", bit_rgb, "bit")
print("Ukuran data    :", ukuran_rgb_bit, "bit")
print("Ukuran data    :", ukuran_rgb_byte, "byte")
print("Ukuran data    :", ukuran_rgb_kb, "KB")
print()

print("Citra Grayscale")
print("Bit per piksel :", bit_gray, "bit")
print("Ukuran data    :", ukuran_gray_bit, "bit")
print("Ukuran data    :", ukuran_gray_byte, "byte")
print("Ukuran data    :", ukuran_gray_kb, "KB")
print()

print("Citra Biner")
print("Bit per piksel :", bit_biner, "bit")
print("Ukuran data    :", ukuran_biner_bit, "bit")
print("Ukuran data    :", ukuran_biner_byte, "byte")
print("Ukuran data    :", ukuran_biner_kb, "KB")
print()



img_rgb_simpan = img_rgb[:, :, ::-1]

cv2.imwrite(os.path.join(folder_hasil, "Image_RGB.png"), img_rgb_simpan)
cv2.imwrite(os.path.join(folder_hasil, "Image_Grayscale.png"), img_gray)
cv2.imwrite(os.path.join(folder_hasil, "Image_Biner.png"), img_biner)

print("Hasil gambar berhasil disimpan di folder:", folder_hasil)

# Menampilkan gambar
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Citra RGB")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_gray, cmap="gray")
plt.title("Citra Grayscale")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img_biner, cmap="gray")
plt.title("Citra Biner")
plt.axis("off")

plt.tight_layout()
plt.show()