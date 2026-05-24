import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder_hasil = "image/hasil_brightness_contrast_soal_04"
os.makedirs(folder_hasil, exist_ok=True)

# Membaca gambar asli berwarna
img = cv2.imread("image/Soal 04.jpg")

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

print("RUN peningkatan brightness dan contrast stretching")

# A. Meningkatkan Brightness
# Rumus: g(x,y) = f(x,y) + beta
beta = 80

img_brightness = img.astype(np.int16) + beta
img_brightness = np.clip(img_brightness, 0, 255)
img_brightness = img_brightness.astype(np.uint8)

# B. Contrast Stretching
# Rumus: g(x,y) = ((f(x,y) - f_min) / (f_max - f_min)) * 255

f_min = np.min(img)
f_max = np.max(img)

img_contrast = ((img.astype(np.float32) - f_min) / (f_max - f_min)) * 255
img_contrast = np.clip(img_contrast, 0, 255)
img_contrast = img_contrast.astype(np.uint8)


# Simpan hasil gambar
cv2.imwrite(os.path.join(folder_hasil, "citra_asli.png"), img)
cv2.imwrite(os.path.join(folder_hasil, "brightness.png"), img_brightness)
cv2.imwrite(os.path.join(folder_hasil, "contrast_stretching.png"), img_contrast)

print("Hasil gambar berhasil disimpan di folder:", folder_hasil)

# Konversi BGR ke RGB hanya untuk tampilan matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_brightness_rgb = cv2.cvtColor(img_brightness, cv2.COLOR_BGR2RGB)
img_contrast_rgb = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2RGB)


# Membuat grayscale hanya untuk histogram
gray_asli = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_brightness = cv2.cvtColor(img_brightness, cv2.COLOR_BGR2GRAY)
gray_contrast = cv2.cvtColor(img_contrast, cv2.COLOR_BGR2GRAY)

#Tampilkan gambar dan histogram
plt.figure(figsize=(12, 8))

# Gambar asli
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")

# Brightness
plt.subplot(2, 3, 2)
plt.imshow(img_brightness_rgb)
plt.title("Brightness +50")
plt.axis("off")

# Contrast Stretching
plt.subplot(2, 3, 3)
plt.imshow(img_contrast_rgb)
plt.title("Contrast Stretching")
plt.axis("off")

# Histogram citra asli
plt.subplot(2, 3, 4)
plt.hist(gray_asli.ravel(), bins=256, range=(0, 255))
plt.title("Histogram Asli")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")

# Histogram brightness
plt.subplot(2, 3, 5)
plt.hist(gray_brightness.ravel(), bins=256, range=(0, 255))
plt.title("Histogram Brightness")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")

# Histogram contrast stretching
plt.subplot(2, 3, 6)
plt.hist(gray_contrast.ravel(), bins=256, range=(0, 255))
plt.title("Histogram Contrast Stretching")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Jumlah Piksel")

plt.tight_layout()
plt.show()