from __future__ import annotations

import time
from pathlib import Path

import cv2

from detector import HelmetDetector
from utils import (
    FPSCounter,
    ThreadedVideoStream,
    ViolationRecorder,
    draw_status_panel,
    format_exception,
    safe_destroy_windows,
    source_to_string,
)


# Ganti nilai ini ke path video, indeks kamera, RTSP, atau HTTP stream CCTV.
VIDEO_SOURCE = "https://stream.lihatcctv.com/stream/0196ebdf-efdd-7307-adfa-b0273b207a09"

# Letakkan model custom YOLOv8 Anda di model/best.pt.
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "best.pt"
VIOLATIONS_DIR = BASE_DIR / "app" / "violations"

CONFIDENCE_THRESHOLD = 0.45
IMAGE_SIZE = 640
RETRY_DELAY_SECONDS = 2.0
MAX_STREAM_RETRIES = 0  # 0 berarti retry terus untuk stream CCTV.
VIOLATION_COOLDOWN_SECONDS = 2.0
WINDOW_NAME = "Helmet Violation Detection"


def main() -> int:
    try:
        detector = HelmetDetector(
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            image_size=IMAGE_SIZE,
        )
    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        return 1
    except Exception as error:
        print(f"[ERROR] Gagal memuat model YOLO: {format_exception(error)}")
        return 1

    stream = ThreadedVideoStream(
        source=VIDEO_SOURCE,
        retry_delay=RETRY_DELAY_SECONDS,
        max_retries=MAX_STREAM_RETRIES,
    ).start()
    recorder = ViolationRecorder(
        output_dir=VIOLATIONS_DIR,
        cooldown_seconds=VIOLATION_COOLDOWN_SECONDS,
    )
    fps_counter = FPSCounter()

    print(f"[INFO] Membuka video source: {source_to_string(VIDEO_SOURCE)}")
    if not stream.wait_until_opened(timeout=10.0):
        print("[WARN] Stream belum terbuka. Sistem tetap menunggu dan akan retry otomatis.")

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                if stream.status in {"failed", "ended"}:
                    print(f"[ERROR] Stream berhenti dengan status: {stream.status}")
                    return 1
                time.sleep(0.03)
                continue

            detections = detector.detect(frame)
            annotated_frame = detector.draw_detections(frame, detections)

            has_violation = any(item.label == "no-helmet" for item in detections)
            recorder.record_if_needed(annotated_frame, has_violation)

            fps = fps_counter.update()
            draw_status_panel(
                annotated_frame,
                fps=fps,
                violation_count=recorder.count,
                stream_status=stream.status,
            )

            cv2.imshow(WINDOW_NAME, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit requested.")
                break

    except KeyboardInterrupt:
        print("[INFO] Dihentikan oleh user.")
    except Exception as error:
        print(f"[ERROR] Runtime error: {format_exception(error)}")
        return 1
    finally:
        stream.stop()
        safe_destroy_windows()

    print(f"[INFO] Total pelanggaran tersimpan: {recorder.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
