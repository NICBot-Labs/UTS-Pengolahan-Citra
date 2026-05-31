import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

folder_hasil = "../image/hasil_noise_salt_and_pepper_soal_06"
os.makedirs(folder_hasil, exist_ok=True)
img = cv2.imread("../image/Soal 06.jpeg")

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def tambah_salt_pepper_noise(img, jumlah_noise=0.05):
    hasil = img.copy()
    tinggi, lebar, channel = hasil.shape
    jumlah_piksel_noise = int(jumlah_noise * tinggi * lebar)
    for i in range(jumlah_piksel_noise):
        y = np.random.randint(0, tinggi)
        x = np.random.randint(0, lebar)
        if np.random.rand() < 0.5:
            hasil[y, x] = [0, 0, 0]
        else:
            hasil[y, x] = [255, 255, 255]    # salt / putih

    return hasil

img_noise = tambah_salt_pepper_noise(img_rgb, jumlah_noise=0.05)
cv2.imwrite(
    os.path.join(folder_hasil, "salt_pepper_noise.png"),
    cv2.cvtColor(img_noise, cv2.COLOR_RGB2BGR)
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_noise)
plt.title("Salt-and-Pepper Noise")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Hasil noise disimpan di folder:", folder_hasil)