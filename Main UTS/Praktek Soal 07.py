import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_PATH = "image/Soal 07.jpeg"
OUTPUT_DIR = "image/process_OCR_07"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def simpan(nama_file, image):
    path = os.path.join(OUTPUT_DIR, nama_file)

    if not cv2.imwrite(path, image):
        raise IOError(f"Gagal menyimpan gambar: {path}")


def urutkan_titik(points):
    """Urutan: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    points = np.array(points, dtype="float32")
    hasil = np.zeros((4, 2), dtype="float32")

    jumlah = points.sum(axis=1)
    selisih = np.diff(points, axis=1).reshape(-1)

    hasil[0] = points[np.argmin(jumlah)]
    hasil[2] = points[np.argmax(jumlah)]
    hasil[1] = points[np.argmin(selisih)]
    hasil[3] = points[np.argmax(selisih)]

    return hasil


def deteksi_dokumen(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edge = cv2.Canny(blur, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=2)
    edge = cv2.dilate(edge, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edge,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Menghindari area kecil dan area yang hampir memenuhi seluruh foto.
        if area < 0.15 * image_area or area > 0.95 * image_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append((area, approx.reshape(4, 2)))

    if candidates:
        _, points = max(candidates, key=lambda item: item[0])
        return urutkan_titik(points), edge

    # Fallback: kotak minimum dari kontur terbesar yang cukup luas.
    valid_contours = [
        contour for contour in contours
        if cv2.contourArea(contour) >= 0.15 * image_area
    ]

    if not valid_contours:
        raise ValueError(
            "Dokumen tidak terdeteksi. Gunakan latar belakang yang lebih kontras."
        )

    largest = max(valid_contours, key=cv2.contourArea)
    rectangle = cv2.minAreaRect(largest)
    points = cv2.boxPoints(rectangle)

    return urutkan_titik(points), edge


def hitung_sudut(points):
    tl, tr, _, _ = urutkan_titik(points)

    delta_x = tr[0] - tl[0]
    delta_y = tr[1] - tl[1]

    return np.degrees(np.arctan2(delta_y, delta_x))


def rotasi_gambar(image, angle):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )


def perspective_transform(image, points):
    tl, tr, br, bl = urutkan_titik(points)

    width = int(max(
        np.linalg.norm(tr - tl),
        np.linalg.norm(br - bl)
    ))

    height = int(max(
        np.linalg.norm(bl - tl),
        np.linalg.norm(br - tr)
    ))

    if width <= 0 or height <= 0:
        raise ValueError("Ukuran dokumen hasil transformasi tidak valid.")

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        urutkan_titik(points),
        destination
    )

    return cv2.warpPerspective(image, matrix, (width, height))


# 1. Membaca citra
original = cv2.imread(INPUT_PATH)
if original is None:
    raise FileNotFoundError(f"File tidak ditemukan: {INPUT_PATH}")

simpan("01_foto_asli.jpg", original)

# 2. Deteksi dokumen
corners_original, edge = deteksi_dokumen(original)
simpan("00_hasil_edge_detection.jpg", edge)

# Visualisasi empat titik yang dipilih program

detected = original.copy()
for index, point in enumerate(corners_original):
    x, y = point.astype(int)
    cv2.circle(detected, (x, y), 12, (0, 0, 255), -1)
    cv2.putText(detected, str(index + 1), (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.polylines(detected, [corners_original.astype(np.int32)], True, (0, 255, 0), 4)
simpan("00b_sudut_dokumen.jpg", detected)

# 3. Koreksi orientasi
angle = hitung_sudut(corners_original)
print(f"Sudut kemiringan dokumen: {angle:.2f} derajat")

# Rotasi berlawanan dengan arah kemiringan.
oriented = rotasi_gambar(original, angle)
simpan("02_koreksi_orientasi.jpg", oriented)

# 4. Cropping otomatis
# Mengambil kotak pembatas dari empat sudut dokumen pada foto asli.
# Cropping hanya membuang latar belakang, belum meluruskan perspektif.

x, y, width, height = cv2.boundingRect(
    corners_original.astype(np.int32)
)
cropped = original[y:y + height, x:x + width]
if cropped.size == 0:
    raise ValueError("Hasil cropping kosong.")
simpan("03_cropping_otomatis.jpg", cropped)

# 5. Perspective correction
# Mengubah bentuk trapesium menjadi persegi panjang.
scanned = perspective_transform(original, corners_original)
simpan("04_perspective_correction.jpg", scanned)

# 6. Menampilkan perbandingan hasil
images = [
    original,
    oriented,
    cropped,
    scanned
]

titles = [
    "1. Foto Asli",
    "2. Koreksi Orientasi",
    "3. Cropping Otomatis",
    "4. Perspective Correction"
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, image, title in zip(axes.ravel(), images, titles):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=14, pad=10)
    ax.axis("off")

fig.suptitle(
    "Perbandingan Tahapan Transformasi Geometrik Dokumen",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_perbandingan_hasil.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.show()