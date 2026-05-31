import cv2
import matplotlib.pyplot as plt
import os

# Folder hasil
folder_hasil = "image/hasil_filter_noise_soal_06"
os.makedirs(folder_hasil, exist_ok=True)

img_noise = cv2.imread("image/hasil_noise_salt_and_pepper_soal_06/salt_pepper_noise.png")
img = cv2.imread("image/Soal 06.jpeg")

if img is None:
    print("Gambar tidak ditemukan!")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)

# img_rgb = cv2.resize(img_rgb, (400, 300))
# img_noise = cv2.resize(img_noise, (400, 300))

# Mean Filter 3x3 Manual
def mean_filter_3x3(img):
    tinggi, lebar, channel = img.shape
    hasil = img.copy()

    for y in range(1, tinggi - 1):
        for x in range(1, lebar - 1):
            for c in range(channel):
                total = 0

                # Mengambil tetangga 3x3
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        total += int(img[y + i, x + j, c])

                nilai_baru = total / 9
                hasil[y, x, c] = int(nilai_baru)
    return hasil

# Median Filter 3x3 Manual
def median_filter_3x3(img):
    tinggi, lebar, channel = img.shape
    hasil = img.copy()

    for y in range(1, tinggi - 1):
        for x in range(1, lebar - 1):
            for c in range(channel):
                data_tetangga = []

                # Mengambil tetangga 3x3
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        data_tetangga.append(img[y + i, x + j, c])

                # Urutkan nilai piksel
                data_tetangga.sort()

                # Ambil nilai tengah
                nilai_median = data_tetangga[4]

                hasil[y, x, c] = nilai_median
    return hasil



# Terapkan Mean Filter 3x3
img_mean = mean_filter_3x3(img_noise)
# Terapkan Median Filter 3x3
img_median = median_filter_3x3(img_noise)

# Simpan hasil gambar
cv2.imwrite(
    os.path.join(folder_hasil, "01_citra_asli.png"),
    cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(folder_hasil, "02_citra_noise.png"),
    cv2.cvtColor(img_noise, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(folder_hasil, "03_mean_filter_3x3.png"),
    cv2.cvtColor(img_mean, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    os.path.join(folder_hasil, "04_median_filter_3x3.png"),
    cv2.cvtColor(img_median, cv2.COLOR_RGB2BGR)
)

# Tampilkan hasil
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img_noise)
plt.title("Citra dengan Salt-and-Pepper Noise")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(img_mean)
plt.title("Mean Filter 3x3")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(img_median)
plt.title("Median Filter 3x3")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Hasil gambar berhasil disimpan di folder:", folder_hasil)