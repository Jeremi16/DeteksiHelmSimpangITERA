# Deteksi Pelanggaran Helm YOLOv8

Project Python untuk deteksi realtime pengendara motor yang memakai helm dan tidak memakai helm dari video CCTV, video file, webcam, RTSP, atau HTTP stream. Sistem menggunakan YOLOv8 custom model dan OpenCV.

## Struktur Project

```text
app/
├── main.py
├── detector.py
├── utils.py
├── requirements.txt
├── violations/
model/
└── best.pt
```

`model/best.pt` tidak disertakan karena biasanya berukuran besar. Letakkan file model YOLOv8 custom Anda di path tersebut.

## Install

Gunakan Python 3.10 atau lebih baru.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r app/requirements.txt
```

Untuk GPU NVIDIA, install PyTorch sesuai CUDA yang terpasang dari dokumentasi resmi PyTorch, lalu install requirements di atas.

## Cara Run

Pastikan model tersedia di:

```text
model/best.pt
```

Jalankan:

```bash
python app/main.py
```

Keyboard control:

- `q` untuk keluar dari aplikasi.

## Mengganti Source CCTV / Video

Buka `app/main.py`, lalu ubah variable `VIDEO_SOURCE`.

Contoh video file:

```python
VIDEO_SOURCE = "data/cctv.mp4"
```

Contoh webcam:

```python
VIDEO_SOURCE = 0
```

Contoh RTSP CCTV:

```python
VIDEO_SOURCE = "rtsp://username:password@192.168.1.10:554/stream1"
```

Contoh HTTP stream:

```python
VIDEO_SOURCE = "http://192.168.1.10:8080/video"
```

## Mengganti Model

Default model berada di:

```python
MODEL_PATH = Path("model/best.pt")
```

Jika model Anda berada di lokasi lain, ubah `MODEL_PATH` pada `app/main.py`.

Model harus memiliki class:

- `helmet`
- `no-helmet`

Beberapa variasi nama class seperti `no helmet` dan `no_helmet` tetap dinormalisasi menjadi `no-helmet`.

## Output Pelanggaran

Ketika class `no-helmet` terdeteksi, sistem akan:

- menampilkan bounding box merah
- menampilkan teks `NO HELMET`
- menambah counter pelanggaran
- menyimpan screenshot otomatis ke `app/violations/`

Format nama file:

```text
violation_TIMESTAMP.jpg
```

Screenshot memakai cooldown default 2 detik agar satu objek yang sama tidak menyimpan file berulang di setiap frame.

## Konfigurasi Penting

Semua konfigurasi utama ada di `app/main.py`:

```python
VIDEO_SOURCE = "video.mp4"
MODEL_PATH = Path("model/best.pt")
CONFIDENCE_THRESHOLD = 0.45
IMAGE_SIZE = 640
RETRY_DELAY_SECONDS = 2.0
MAX_STREAM_RETRIES = 0
VIOLATION_COOLDOWN_SECONDS = 2.0
```

Untuk laptop biasa, gunakan `IMAGE_SIZE = 640` atau turunkan ke `512` jika FPS terlalu rendah. Naikkan `CONFIDENCE_THRESHOLD` jika terlalu banyak false positive.

## Catatan Production

- Stream dibaca menggunakan thread terpisah agar proses inferensi tidak terlalu menghambat pembacaan frame.
- Jika stream CCTV putus, sistem akan mencoba reconnect otomatis.
- Buffer OpenCV diset kecil supaya frame yang diproses lebih mendekati realtime.
- Jika `model/best.pt` tidak ditemukan, aplikasi akan berhenti dengan pesan error yang jelas.
