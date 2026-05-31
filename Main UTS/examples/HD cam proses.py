import cv2
import numpy as np
import os
from datetime import datetime


# ============================================================
# 1. IMAGE ENHANCE
# Memperbaiki kualitas gambar secara umum:
# - lebih terang
# - kontras lebih jelas
# - noise sedikit berkurang
# ============================================================

def image_enhance(frame):
    frame = cv2.resize(frame, (640, 480))

    # Naikkan brightness dan contrast
    alpha = 1.2   # kontras
    beta = 25     # kecerahan
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Kurangi noise ringan
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Tajamkan kembali gambar
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

    return enhanced


# ============================================================
# 2. FACE FILTER
# Membuat wajah terlihat lebih cerah, bersih, dan halus
# Efek seperti filter kamera / filter IG sederhana
# ============================================================

def face_filter(frame):
    frame = cv2.resize(frame, (640, 480))

    # Cerahkan gambar
    alpha = 1.15
    beta = 30
    bright = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Menghaluskan wajah tetapi tepi gambar tetap cukup jelas
    smooth = cv2.bilateralFilter(bright, 9, 75, 75)

    # Naikkan warna sedikit agar tidak pucat
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = cv2.convertScaleAbs(s, alpha=1.15, beta=0)
    v = cv2.convertScaleAbs(v, alpha=1.05, beta=10)

    hasil_hsv = cv2.merge([h, s, v])
    hasil = cv2.cvtColor(hasil_hsv, cv2.COLOR_HSV2BGR)

    return hasil


# ============================================================
# 3. IMAGE CAPTURE
# Menyimpan gambar ke folder hasil_capture
# ============================================================

def image_capture(frame, nama_filter):
    folder = "hasil_capture"

    if not os.path.exists(folder):
        os.makedirs(folder)

    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    nama_file = f"{folder}/{nama_filter}_{waktu}.jpg"

    cv2.imwrite(nama_file, frame)

    print("Gambar berhasil disimpan:", nama_file)


# ============================================================
# PROGRAM UTAMA TANPA GUI
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka")
    exit()

mode = 1

print("===================================")
print("PROGRAM IMAGE PROCESSING TANPA GUI")
print("===================================")
print("Tekan tombol:")
print("1 = Image Enhance")
print("2 = Face Filter")
print("3 = Kamera Asli")
print("s = Simpan gambar")
print("q = Keluar")
print("===================================")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame kamera tidak terbaca")
        break

    frame = cv2.resize(frame, (640, 480))

    if mode == 1:
        hasil = image_enhance(frame)
        nama_window = "Image Enhance"
        nama_simpan = "image_enhance"

    elif mode == 2:
        hasil = face_filter(frame)
        nama_window = "Face Filter"
        nama_simpan = "face_filter"

    else:
        hasil = frame
        nama_window = "Kamera Asli"
        nama_simpan = "kamera_asli"

    cv2.imshow(nama_window, hasil)

    tombol = cv2.waitKey(1) & 0xFF

    if tombol == ord('1'):
        cv2.destroyAllWindows()
        mode = 1
        print("Mode: Image Enhance")

    elif tombol == ord('2'):
        cv2.destroyAllWindows()
        mode = 2
        print("Mode: Face Filter")

    elif tombol == ord('3'):
        cv2.destroyAllWindows()
        mode = 3
        print("Mode: Kamera Asli")

    elif tombol == ord('s'):
        image_capture(hasil, nama_simpan)

    elif tombol == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()