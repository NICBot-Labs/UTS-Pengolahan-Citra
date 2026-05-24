import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder_hasil = "image/hasil_histogram_equalization"
os.makedirs(folder_hasil, exist_ok=True)

# Membaca foto wajah
img_bgr = cv2.imread("image/Soal 05.jpeg")

if img_bgr is None:
    print("Gambar tidak ditemukan!")
    exit()

print("RUN histogram equalization pada foto wajah")

# Ubah BGR ke RGB untuk tampilan
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Ubah ke grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# A. Histogram Equalization
img_equalized = cv2.equalizeHist(img_gray)

# Simpan hasil
cv2.imwrite(os.path.join(folder_hasil, "01_wajah_grayscale.png"), img_gray)
cv2.imwrite(os.path.join(folder_hasil, "02_wajah_equalization.png"), img_equalized)

print("Hasil gambar berhasil disimpan di folder:", folder_hasil)

# B. Tampilkan citra dan histogram sebelum-sesudah
plt.figure(figsize=(12, 8))

# Foto asli berwarna
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Foto Wajah Asli")
plt.axis("off")

# Grayscale sebelum equalization
plt.subplot(2, 3, 2)
plt.imshow(img_gray, cmap="gray")
plt.title("Grayscale Sebelum")
plt.axis("off")

# Setelah histogram equalization
plt.subplot(2, 3, 3)
plt.imshow(img_equalized, cmap="gray")
plt.title("Setelah Equalization")
plt.axis("off")

# Histogram grayscale sebelum
plt.subplot(2, 3, 5)
plt.hist(img_gray.ravel(), bins=256, range=(0, 255))
plt.title("Histogram Sebelum (Gray)")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")

# Histogram setelah equalization
plt.subplot(2, 3, 6)
plt.hist(img_equalized.ravel(), bins=256, range=(0, 255))
plt.title("Histogram Sesudah (Equal)")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")

plt.tight_layout()
plt.show()