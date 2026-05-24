import cv2

# Load XML Haar Cascade
face_detection = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
eye_detection = cv2.CascadeClassifier("models/haarcascade_eye_tree_eyeglasses.xml")

# Cek model XML
if face_detection.empty():
    print("Model wajah gagal dimuat!")

if eye_detection.empty():
    print("Model mata gagal dimuat!")

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak dapat dibuka")
    exit()

# Variabel deteksi
counter_mengantuk = 0
BATAS_MENGANTUK = 20

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kamera tidak terbaca")
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    status = "WAJAH TIDAK TERDETEKSI"
    warna_status = (0, 0, 255)

    for (x, y, w, h) in faces:
        # Gambar kotak wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Ambil area wajah bagian atas untuk mencari mata
        roi_gray = gray[y:y + int(h * 0.55), x:x + w]
        roi_color = frame[y:y + int(h * 0.55), x:x + w]

        eyes = eye_detection.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )

        # Gambar kotak mata
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

        # Logika sederhana:
        # Jika mata terdeteksi = tidak mengantuk
        # Jika mata tidak terdeteksi beberapa frame = mengantuk
        if len(eyes) >= 1:
            counter_mengantuk = 0
            status = "TIDAK MENGANTUK"
            warna_status = (0, 255, 0)
        else:
            counter_mengantuk += 1

            if counter_mengantuk >= BATAS_MENGANTUK:
                status = "MENGANTUK"
                warna_status = (0, 0, 255)
            else:
                status = "MATA TERTUTUP SEMENTARA"
                warna_status = (0, 255, 255)

        break

    # Tampilkan status
    cv2.putText(
        frame,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        warna_status,
        2
    )

    cv2.putText(
        frame,
        "Counter: " + str(counter_mengantuk),
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to Exit",
        (30, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Deteksi Mengantuk XML", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()