from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


HELMET_ALIASES = {"helmet", "helm"}
NO_HELMET_ALIASES = {
    "no-helmet",
    "no helmet",
    "no_helmet",
    "without-helmet",
    "without helmet",
    "without_helmet",
    "not-helmet",
    "not helmet",
    "not_helmet",
}


@dataclass(frozen=True)
class Detection:
    """Normalized detection output used by the UI and violation recorder."""

    label: str
    confidence: float
    box: tuple[int, int, int, int]


class HelmetDetector:
    """YOLOv8 wrapper focused on helmet and no-helmet classes."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.45,
        image_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model YOLO tidak ditemukan: {self.model_path.resolve()}. "
                "Letakkan model custom Anda di path tersebut, misalnya model/best.pt."
            )

        self.model = YOLO(str(self.model_path))
        self.names = self._normalize_names(self.model.names)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference and return only helmet/no-helmet detections."""
        predict_kwargs = {
            "source": frame,
            "conf": self.confidence_threshold,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        return list(self._parse_result(results[0]))

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels directly onto a copied frame."""
        output = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            is_violation = detection.label == "no-helmet"
            color = (0, 0, 255) if is_violation else (0, 180, 0)
            text = (
                f"NO HELMET {detection.confidence:.2f}"
                if is_violation
                else f"helmet {detection.confidence:.2f}"
            )

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            self._draw_label(output, text, x1, y1, color)

        return output

    def _parse_result(self, result) -> Iterable[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return

        for box in boxes:
            class_id = int(box.cls[0].item())
            raw_label = self.names.get(class_id, str(class_id))
            label = self._normalize_label(raw_label)
            if label not in {"helmet", "no-helmet"}:
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            yield Detection(label=label, confidence=confidence, box=(x1, y1, x2, y2))

    @staticmethod
    def _normalize_names(names) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}

    @staticmethod
    def _normalize_label(label: str) -> str:
        normalized = label.strip().lower().replace("_", "-")
        if normalized in HELMET_ALIASES:
            return "helmet"
        if normalized in {item.replace("_", "-") for item in NO_HELMET_ALIASES}:
            return "no-helmet"
        return normalized

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        padding = 6
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

        label_y1 = max(0, y - text_height - baseline - padding * 2)
        label_y2 = max(text_height + baseline + padding * 2, y)
        label_x2 = min(frame.shape[1] - 1, x + text_width + padding * 2)

        cv2.rectangle(frame, (x, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            frame,
            text,
            (x + padding, label_y2 - baseline - padding),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
