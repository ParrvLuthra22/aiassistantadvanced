"""Local face authentication helper for startup access control."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False


@dataclass
class FaceAuthResult:
    granted: bool
    message: str
    similarity: float = 0.0
    enrolled: bool = False


class FaceAuthenticator:
    """
    Lightweight local face authentication using OpenCV + cosine similarity.

    This is intentionally lightweight for Apple Silicon laptops with limited RAM.
    It stores a normalized face embedding in local files under data/face_auth.
    """

    def __init__(
        self,
        data_dir: str = "data/face_auth",
        camera_id: int = 0,
        threshold: float = 0.82,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.camera_id = int(camera_id)
        self.threshold = float(threshold)

        self.embedding_path = self.data_dir / "reference_embedding.npy"
        self.meta_path = self.data_dir / "reference_meta.json"
        self._last_capture_error: str = ""

    def verify_or_enroll(self) -> FaceAuthResult:
        if not CV2_AVAILABLE or cv2 is None:
            return FaceAuthResult(
                granted=False,
                message="Face auth unavailable: opencv-python is not installed.",
            )

        current = self._capture_embedding()
        if current is None:
            if self._last_capture_error == "camera_unavailable":
                return FaceAuthResult(
                    granted=False,
                    message=(
                        "Face auth failed: camera is unavailable or permission is denied. "
                        "Allow camera access for your terminal/IDE in System Settings."
                    ),
                )
            return FaceAuthResult(
                granted=False,
                message="Face auth failed: no clear face detected from camera.",
            )

        if not self.embedding_path.exists():
            return self._enroll_from_embedding(current)

        saved = self._load_reference()
        if saved is None:
            return self._enroll_from_embedding(
                current,
                message="Face profile was corrupted and has been recreated. Access granted, Sir.",
            )

        similarity = self._cosine_similarity(saved, current)
        if similarity >= self.threshold:
            return FaceAuthResult(
                granted=True,
                message=f"Identity verified (similarity {similarity:.2f}). Access granted, Sir.",
                similarity=similarity,
            )

        return FaceAuthResult(
            granted=False,
            message=(
                f"Identity check failed (similarity {similarity:.2f}). "
                "Access denied until the registered user is recognized."
            ),
            similarity=similarity,
        )

    def force_enroll(self) -> FaceAuthResult:
        """Capture current face and overwrite stored profile."""
        if not CV2_AVAILABLE or cv2 is None:
            return FaceAuthResult(
                granted=False,
                message="Face enrollment unavailable: opencv-python is not installed.",
            )
        current = self._capture_embedding(allow_center_crop_fallback=True)
        if current is None:
            if self._last_capture_error == "camera_unavailable":
                return FaceAuthResult(
                    granted=False,
                    message=(
                        "Face enrollment failed: camera is unavailable or permission is denied. "
                        "Allow camera access for your terminal/IDE in System Settings."
                    ),
                )
            return FaceAuthResult(
                granted=False,
                message="Face enrollment failed: no clear face detected from camera.",
            )
        return self._enroll_from_embedding(
            current,
            message="Face profile updated successfully. Access granted, Sir.",
        )

    def _capture_embedding(self, allow_center_crop_fallback: bool = False) -> Optional[np.ndarray]:
        if not CV2_AVAILABLE or cv2 is None:
            return None

        self._last_capture_error = ""
        cap = self._open_camera()
        if not cap.isOpened():
            self._last_capture_error = "camera_unavailable"
            return None

        frame = None
        try:
            # Warm up camera and try multiple frames to improve face detection reliability.
            for _ in range(40):
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.03)
                    continue
                face_crop = self._extract_largest_face(frame)
                if face_crop is not None:
                    return self._build_embedding(face_crop)
                time.sleep(0.03)
            if frame is None:
                return None
            if allow_center_crop_fallback:
                center_crop = self._extract_center_crop(frame)
                if center_crop is not None:
                    return self._build_embedding(center_crop)
            return None
        finally:
            cap.release()

    def _open_camera(self):
        if cv2 is None:
            return None
        candidates = []
        if hasattr(cv2, "CAP_AVFOUNDATION"):
            candidates.append(cv2.CAP_AVFOUNDATION)
        if hasattr(cv2, "CAP_ANY"):
            candidates.append(cv2.CAP_ANY)
        candidates.append(None)

        for backend in candidates:
            if backend is None:
                cap = cv2.VideoCapture(self.camera_id)
            else:
                cap = cv2.VideoCapture(self.camera_id, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(self.camera_id)

    def _extract_largest_face(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if not CV2_AVAILABLE or cv2 is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_dim = min(gray.shape[:2])
        min_size = max(56, int(min_dim * 0.14))

        cascade_files = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
        ]
        faces = []
        for cascade_file in cascade_files:
            cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / cascade_file))
            if cascade.empty():
                continue
            for scale_factor, min_neighbors in ((1.1, 5), (1.08, 4), (1.05, 3)):
                found = cascade.detectMultiScale(
                    gray,
                    scaleFactor=scale_factor,
                    minNeighbors=min_neighbors,
                    minSize=(min_size, min_size),
                )
                if len(found) > 0:
                    faces = found
                    break
            if len(faces) > 0:
                break

        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        return gray[y : y + h, x : x + w]

    @staticmethod
    def _extract_center_crop(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if h < 40 or w < 40:
            return None
        crop_w = int(w * 0.55)
        crop_h = int(h * 0.70)
        x1 = max(0, (w - crop_w) // 2)
        y1 = max(0, (h - crop_h) // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        return gray[y1:y2, x1:x2]

    @staticmethod
    def _build_embedding(face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (96, 96))  # type: ignore[arg-type]
        normalized = resized.astype(np.float32) / 255.0
        emb = normalized.flatten()
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def _save_reference(self, embedding: np.ndarray) -> None:
        np.save(self.embedding_path, embedding)
        payload = {"threshold": self.threshold}
        self.meta_path.write_text(json.dumps(payload, indent=2))

    def _enroll_from_embedding(
        self,
        embedding: np.ndarray,
        message: str = "Face profile created. Access granted, Sir.",
    ) -> FaceAuthResult:
        self._save_reference(embedding)
        return FaceAuthResult(
            granted=True,
            message=message,
            similarity=1.0,
            enrolled=True,
        )

    def _load_reference(self) -> Optional[np.ndarray]:
        try:
            return np.load(self.embedding_path)
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
