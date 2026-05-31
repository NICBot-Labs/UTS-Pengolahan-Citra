from tkinter import *
from PIL import Image, ImageTk
import cv2
import pygame


########################################################################################################################
# FUCTION
########################################################################################################################

# VARIABEL global warna
background_header = "#FCFCFC"
background_sidebar = "#0D2AAB"
background_konten = "#E7EDF0"

# variabel global untuk kamera
cap = None
camera_running = False
label_camera = None

# variabel global deteksi mengantuk
counter_mengantuk = 0
BATAS_MENGANTUK = 10

# variabel global deteksi suara
status_var = None
suara_var = None

# variabel global alarm
alarm_sedang_berjalan = False

alarm_path = "alarm/Asistan.mp3"  # ganti sesuai lokasi file musik kamu

daftar_suara_alarm = {
    "Asistan": "alarm/Asistan.mp3",
    "Ambulan": "alarm/ambulan.mp3",
    "Warning": "alarm/Warning.mp3",
}

pygame.mixer.init()

# variabel code untuk warna menu aktif
menu_buttons = []
active_menu = None

def hover_masuk(event):
    if event.widget != active_menu:
        event.widget.config(bg="#3048B7")

def hover_keluar(event):
    if event.widget != active_menu:
        event.widget.config(bg=background_sidebar)

def alarm_nyala(time_waktu):
    global alarm_sedang_berjalan
    if alarm_sedang_berjalan:
        return
    alarm_sedang_berjalan = True
    try:
        pygame.mixer.music.load(alarm_path)
        pygame.mixer.music.play()
        window.after(time_waktu, stop_alarm)
    except Exception as e:
        print("Gagal memutar alarm:", e)

def stop_alarm():
    global alarm_sedang_berjalan
    pygame.mixer.music.stop()
    alarm_sedang_berjalan = False

face_detection = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
eye_detection = cv2.CascadeClassifier("models/haarcascade_eye_tree_eyeglasses.xml")

def set_active_menu(button):
    global active_menu
    active_menu = button
    for btn in menu_buttons:
        if btn == active_menu:
            btn.config(
                bg="#3048B7",
                fg="white",
                activebackground="white",
                activeforeground=background_sidebar
            )
        else:
            btn.config(
                bg=background_sidebar,
                fg="white",
                activebackground="#1E40D0",
                activeforeground="white"
            )

def ganti_suara_alarm(pilihan):
    global alarm_path
    if pilihan in daftar_suara_alarm:
        alarm_path = daftar_suara_alarm[pilihan]
        print("Suara alarm dipilih:", pilihan)
        print("Path alarm:", alarm_path)

########################################################################################################################
# FUCTION CAMERA
########################################################################################################################
def start_camera():
    global cap, camera_running, label_camera
    if label_camera is None:
        return
    try:
        if not label_camera.winfo_exists():
            return
    except:
        return

    if camera_running:
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        label_camera.config(text="Kamera tidak dapat dibuka")
        cap = None
        return

    camera_running = True
    update_camera()


def update_camera():
    global cap, camera_running, label_camera
    if not camera_running:
        return
    if cap is None or label_camera is None:
        return
    try:
        if not label_camera.winfo_exists():
            return
    except:
        return

    ret, frame = cap.read()
    if ret:
        # width = 600, height = 350
        frame = cv2.resize(frame, (600, 350))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        label_camera.config(image=img_tk, text="")
        label_camera.image = img_tk

    label_camera.after(10, update_camera)


def stop_camera():
    global cap, camera_running, label_camera

    camera_running = False

    if cap is not None:
        cap.release()
        cap = None

    if label_camera is not None:
        try:
            if label_camera.winfo_exists():
                label_camera.config(image="", text="Camera stopped")
                label_camera.image = None
        except:
            pass

########################################################################################################################
# FUCTION CAMERA dengan MODEL DETEKSI
########################################################################################################################

def start_camera_deteksi():
    global cap, camera_running, label_camera
    if label_camera is None:
        return
    try:
        if not label_camera.winfo_exists():
            return
    except:
        return

    if camera_running:
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        label_camera.config(text="Kamera tidak dapat dibuka")
        cap = None
        return
    camera_running = True
    update_camera_deteksi()

def update_camera_deteksi():
    global cap, camera_running, label_camera
    global counter_mengantuk, status_var

    if not camera_running:
        return

    if cap is None or label_camera is None:
        return

    try:
        if not label_camera.winfo_exists():
            return
    except:
        return

    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (560, 330))
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + int(h * 0.55), x:x + w]
            roi_color = frame[y:y + int(h * 0.55), x:x + w]

            eyes = eye_detection.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_color,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    (0, 255, 0),
                    2
                )

            if len(eyes) >= 1:
                counter_mengantuk = 0
                status = "TIDAK MENGANTUK"
                warna_status = (0, 255, 0)
            else:
                counter_mengantuk += 1
                if counter_mengantuk >= BATAS_MENGANTUK:
                    status = "MENGANTUK"
                    warna_status = (0, 0, 255)
                    alarm_nyala(5000)
                else:
                    status = "MATA TERTUTUP SEMENTARA"
                    warna_status = (0, 255, 255)

            break

        if len(faces) == 0:
            counter_mengantuk = 0

        if status_var is not None:
            status_var.set(status)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            warna_status,
            2
        )

        cv2.putText(
            frame,
            "Counter: " + str(counter_mengantuk),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        label_camera.config(image=img_tk, text="")
        label_camera.image = img_tk

    label_camera.after(10, update_camera_deteksi)

########################################################################################################################
# PREPARE---------------------------------------------------------------------------------------------------------------
# Membuat window utama
window = Tk()
# Ukuran window
lebar = 1200
tinggi = 600

window.resizable(False, False)
window.title("Deteksi Supir Mengantuk pada kendaran mobil")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Menghitung posisi agar window muncul di tengah layar
posisi_x = int((screen_width / 2) - (lebar / 2))
posisi_y = int((screen_height / 2) - (tinggi / 2))

# Mengatur ukuran dan posisi window
window.geometry(f"{lebar}x{tinggi}+{posisi_x}+{posisi_y}")


########################################################################################################################
# design header
########################################################################################################################
# membuat header
header = Frame(window, bg=background_header, height=60)
header.pack(fill="x")
header.pack_propagate(False)

# logo
logo_path_unu = "image/bonus Image/lOGO unu.png"
logo_path_ilkom = "image/bonus Image/ilkom logo.jpg"

# LOGO UNU HEADER-------------------------------------------------------------------------------------------------------
logo_img_unu = Image.open(logo_path_unu)
logo_img_unu = logo_img_unu.resize((40, 40))
logo_tk_unu = ImageTk.PhotoImage(logo_img_unu)

label_logo_unu = Label(header, image=logo_tk_unu, bg=background_header)
label_logo_unu.image = logo_tk_unu
label_logo_unu.pack(side="left", padx=20)

# LOGO ILKOM HEADER-----------------------------------------------------------------------------------------------------
logo_img_ilkom = Image.open(logo_path_ilkom)
logo_img_ilkom = logo_img_ilkom.resize((40, 40))
logo_tk_ilkom = ImageTk.PhotoImage(logo_img_ilkom)

label_logo_ilkom = Label(header, image=logo_tk_ilkom, bg=background_header)
label_logo_ilkom.image = logo_tk_ilkom
label_logo_ilkom.pack(side="left")


frame_judul = Frame(header, width=350, height=60, bg=background_header)
frame_judul.pack(side="right", padx=20, pady=2)
frame_judul.pack_propagate(False)

judul = Label(frame_judul,text="ALARM DETEKSI DRIVER LELAH",fg="black",bg=background_header, font=("Arial", 15, "bold"))
judul.pack(anchor="e") # anchor e = kanan, w = kiri
judul_ket = Label(frame_judul,text="By: M. NIKO NUR CAHYONO",fg="black",bg=background_header,font=("Arial", 9))
judul_ket.pack(anchor="e") # anchor e = kanan, w = kiri


########################################################################################################################
# design KONTEN UTAMA
########################################################################################################################
# Body utama
# Body utama
body = Frame(window, bg=background_konten)
body.pack(fill="both", expand=True)

# Konten utama
content = Frame(body, bg=background_konten)
content.pack(side="right", fill="both", expand=True)

# Frame judul di area konten
content_judul = Frame(content, bg=background_konten, height=80)
content_judul.pack(side="top", fill="x")
content_judul.pack_propagate(False)

def set_judul_halaman(judul, keterangan):
    for widget in content_judul.winfo_children():
        widget.destroy()

    label_judul = Label(content_judul,text=judul,bg=background_konten,fg="#213CB2",font=("Arial", 18, "bold"))
    label_judul.pack(pady=(20, 0), padx=20, anchor="w")
    label_keterangan = Label(content_judul, text=keterangan, bg=background_konten, fg="#545960", font=("Arial", 12))
    label_keterangan.pack(padx=20, anchor="w")

# Frame isi konten
content_isi = Frame(content, bg=background_konten)
content_isi.pack(side="top", fill="both", expand=True)

def clear_content():
    global label_camera
    stop_camera()
    for widget in content_isi.winfo_children():
        widget.destroy()
    label_camera = None

def show_dashboard():
    global label_camera, status_var, suara_var
    clear_content()
    set_judul_halaman(
        "HOME PAGE",
        "Sistem deteksi wajah untuk peringatan dini kepada driver"
    )

    frame_dashboard = Frame(content_isi, bg=background_konten)
    frame_dashboard.pack(fill="both", expand=True, padx=20, pady=10)


    frame_camera = Frame(frame_dashboard, bg="black", width=560, height=330)
    frame_camera.pack(side="left", padx=(0, 15), pady=10)
    frame_camera.pack_propagate(False)



    label_camera = Label(frame_camera, text="Camera Preview", bg="black", fg="white", font=("Arial", 12))
    label_camera.pack(fill="both", expand=True)

    # ==================================================================================
    # AREA KANAN: STATUS DAN PILIHAN SUARA
    # ==================================================================================
    frame_panel_kanan = Frame(frame_dashboard, bg=background_konten, width=300, height=330)
    frame_panel_kanan.pack(side="left", pady=10)
    frame_panel_kanan.pack_propagate(False)

    # Kotak status
    frame_status = Frame(frame_panel_kanan, bg=background_header, height=70, highlightbackground=background_sidebar, highlightthickness=2)
    frame_status.pack(fill="x", pady=(0, 15))
    frame_status.pack_propagate(False)

    Label(frame_status, text="STATUS DETEKSI", bg="white", fg="#213CB2", font=("Arial", 12, "bold")).pack(anchor="w", padx=15, pady=(10, 0))
    status_var = StringVar()
    status_var.set("Normal - Mata Terdeteksi Terbuka")

    label_status = Label(frame_status, textvariable=status_var, bg="white", fg="green", font=("Arial", 11))
    label_status.pack(anchor="w", padx=15, pady=5)

    # Kotak pilihan suara
    frame_suara = Frame(frame_panel_kanan, bg=background_header, height=100, highlightbackground=background_sidebar, highlightthickness=2)
    frame_suara.pack(fill="x")
    frame_suara.pack_propagate(False)

    Label(
        frame_suara,
        text="PILIHAN SUARA PERINGATAN",
        bg="white",
        fg="#213CB2",
        font=("Arial", 12, "bold")
    ).pack(anchor="w", padx=15, pady=(10, 0))

    suara_var = StringVar()
    suara_var.set("Asistan")

    # "Asistan": "alarm/Asistan.mp3",
    # "Ambulan": "alarm/ambulan.mp3",
    # "Warning": "alarm/Warning.mp3",
    pilihan_suara = [
        "Asistan",
        "Ambulan",
        "Warning"
    ]

    menu_suara = OptionMenu(
        frame_suara,
        suara_var,
        *pilihan_suara,
        command=ganti_suara_alarm
    )
    menu_suara.config(
        width=25,
        bg="#E7EDF0",
        fg="black",
        font=("Arial", 10)
    )
    menu_suara.pack(anchor="w", padx=15, pady=10)

    # Tombol kamera
    frame_tombol = Frame(frame_panel_kanan, bg=background_konten)
    frame_tombol.pack(fill="x", pady=20)

    Button(
        frame_tombol,
        text="Start Camera",
        width=16,
        command=start_camera_deteksi
    ).pack(side="left", padx=5)

    Button(
        frame_tombol,
        text="Stop Camera",
        width=16,
        command=stop_camera
    ).pack(side="left", padx=5)

    # Kamera langsung hidup saat halaman dibuka
    start_camera_deteksi()

def show_hd_image():
    clear_content()
    set_judul_halaman(
        "HD IMAGE PAGE",
        "Cerahkan wajah anda untuk meningkatkan kepercaan saat berkendara"
    )

def show_tes_cam():
    global label_camera

    clear_content()

    set_judul_halaman(
        "TES CAMERA PAGE",
        "Menampilkan kamera secara langsung untuk pengujian deteksi wajah driver"
    )

    frame_camera = Frame(content_isi, bg="black", width=600, height=350)
    frame_camera.pack(pady=20, anchor="w", padx= 20)
    frame_camera.pack_propagate(False)

    label_camera = Label(
        frame_camera,
        text="Camera Preview",
        bg="black",
        fg="white",
        font=("Arial", 12)
    )
    label_camera.pack(fill="both", expand=True)

    frame_tombol = Frame(content_isi, bg=background_konten)
    frame_tombol.pack(pady=10, anchor="w", padx= 20)

    Button(
        frame_tombol,
        text="Start Camera",
        width=18,
        command=start_camera
    ).grid(row=0, column=0)

    Button(
        frame_tombol,
        text="Stop Camera",
        width=18,
        command=stop_camera
    ).grid(row=0, column=1, padx=10)
show_dashboard()

########################################################################################################################
# design sidebar
########################################################################################################################
# membuat sidebar di samping kiri
sidebar = Frame(body, bg=background_sidebar, width=200)
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

logo_path_driver = "image/bonus Image/driver.png"

logo_img_driver = Image.open(logo_path_driver)
logo_img_driver = logo_img_driver.resize((80, 80))
logo_tk_driver = ImageTk.PhotoImage(logo_img_driver)

label_logo_driver = Label(sidebar, image=logo_tk_driver, bg=background_sidebar)
label_logo_driver.image = logo_tk_driver
label_logo_driver.pack(side="top", pady=10)

icon_home_img = Image.open("image/bonus Image/home.png")
icon_hd_img = Image.open("image/bonus Image/hd-sign.png")
icon_conv_img = Image.open("image/bonus Image/exchange.png")
icon_tes_cam = Image.open("image/bonus Image/videocam.png")

icon_home_img = icon_home_img.resize((18, 18))
icon_home_tk = ImageTk.PhotoImage(icon_home_img)

icon_hd_img = icon_hd_img.resize((18, 18))
icon_hd_tk = ImageTk.PhotoImage(icon_hd_img)

icon_conv_img = icon_conv_img.resize((18, 18))
icon_conv_tk = ImageTk.PhotoImage(icon_conv_img)

icon_tes_cam = icon_tes_cam.resize((18, 18))
icon_cam_tk = ImageTk.PhotoImage(icon_tes_cam)

def main_menu(teks, image_icon, command=None):
    tombol = Button(
        sidebar,
        image=image_icon,
        text=teks,
        compound="left",
        font=("Arial", 12),
        bg=background_sidebar,
        fg="#B2BAE4",
        activebackground="#1E40D0",
        activeforeground="white",
        relief="flat",
        bd=0,
        anchor="w",
        padx=10,
        pady=16
    )

    def klik_menu():
        set_active_menu(tombol)
        if command is not None:
            command()

    tombol.config(command=klik_menu)

    tombol.image = image_icon
    tombol.pack(fill="x")
    tombol.bind("<Enter>", hover_masuk)
    tombol.bind("<Leave>", hover_keluar)

    menu_buttons.append(tombol)

    return tombol

btn_home = main_menu("Home", icon_home_tk, command=show_dashboard)
# btn_hd = main_menu("HD Image", icon_hd_tk, command=show_hd_image)
# btn_convert = main_menu("Image Convert", icon_conv_tk)
btn_tes_cam = main_menu("Tes Cam", icon_cam_tk, command=show_tes_cam)

set_active_menu(btn_home)

# Menjalankan GUI
window.mainloop()
