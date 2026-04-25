"""Local face authentication helper for startup access control."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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

    Robustness improvements:
    - Multi-frame capture for both enroll and verify
    - Reference embedding bank instead of single embedding
    - Retry pass before deny to reduce false negatives
    - Optional incremental bank update when verified
    """

    def __init__(
        self,
        data_dir: str = "data/face_auth",
        camera_id: int = 0,
        threshold: float = 0.78,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.camera_id = int(camera_id)
        self.threshold = float(threshold)

        self.bank_path = self.data_dir / "reference_embeddings.npy"
        self.embedding_path = self.data_dir / "reference_embedding.npy"  # legacy
        self.meta_path = self.data_dir / "reference_meta.json"
        self._last_capture_error: str = ""
        self._loaded_legacy_profile = False

    def verify_or_enroll(self) -> FaceAuthResult:
        if not CV2_AVAILABLE or cv2 is None:
            return FaceAuthResult(
                granted=False,
                message="Face auth unavailable: opencv-python is not installed.",
            )

        captured = self._capture_embeddings(
            sample_count=6,
            max_frames=140,
            allow_center_crop_fallback=True,
        )
        if not captured:
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

        bank = self._load_reference_bank()
        if bank is None:
            return self._enroll_from_embeddings(captured)

        best_similarity, best_embedding = self._best_similarity(captured, bank)
        bank_size = int(bank.shape[0]) if bank.ndim > 1 else 1
        effective_threshold = self._effective_threshold(bank_size)
        if best_embedding is None:
            return FaceAuthResult(
                granted=False,
                message="Face auth failed: could not compare captured samples.",
            )

        if best_similarity >= effective_threshold:
            self._promote_bank_on_success(bank=bank, captured=captured, best_embedding=best_embedding)
            return FaceAuthResult(
                granted=True,
                message=f"Identity verified (similarity {best_similarity:.2f}). Access granted, Sir.",
                similarity=best_similarity,
            )

        # Retry one more capture pass before denying.
        retry = self._capture_embeddings(
            sample_count=6,
            max_frames=140,
            allow_center_crop_fallback=True,
        )
        if retry:
            retry_similarity, retry_best = self._best_similarity(retry, bank)
            if retry_best is not None and retry_similarity >= effective_threshold:
                self._promote_bank_on_success(bank=bank, captured=retry, best_embedding=retry_best)
                return FaceAuthResult(
                    granted=True,
                    message=f"Identity verified (similarity {retry_similarity:.2f}). Access granted, Sir.",
                    similarity=retry_similarity,
                )
            if retry_similarity > best_similarity:
                best_similarity = retry_similarity

        return FaceAuthResult(
            granted=False,
            message=(
                f"Identity check failed (similarity {best_similarity:.2f}, threshold {effective_threshold:.2f}). "
                "Access denied until the registered user is recognized."
            ),
            similarity=best_similarity,
        )

    def force_enroll(self) -> FaceAuthResult:
        """Capture current face and overwrite stored profile."""
        if not CV2_AVAILABLE or cv2 is None:
            return FaceAuthResult(
                granted=False,
                message="Face enrollment unavailable: opencv-python is not installed.",
            )

        captured = self._capture_embeddings(
            sample_count=8,
            max_frames=180,
            allow_center_crop_fallback=True,
        )
        if not captured:
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

        return self._enroll_from_embeddings(
            captured,
            message="Face profile updated successfully. Access granted, Sir.",
        )

    def _capture_embeddings(
        self,
        sample_count: int = 6,
        max_frames: int = 140,
        allow_center_crop_fallback: bool = False,
    ) -> List[np.ndarray]:
        if not CV2_AVAILABLE or cv2 is None:
            return []

        self._last_capture_error = ""
        cap = self._open_camera()
        if not cap.isOpened():
            self._last_capture_error = "camera_unavailable"
            return []

        samples: List[np.ndarray] = []
        frame = None
        frame_idx = 0
        try:
            while frame_idx < max_frames and len(samples) < sample_count:
                ok, frame = cap.read()
                frame_idx += 1
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue

                face_crop = self._extract_largest_face(frame)
                if face_crop is None:
                    time.sleep(0.02)
                    continue

                emb = self._build_embedding(face_crop)
                if emb.size == 0:
                    continue

                # Keep diverse samples, avoid identical duplicates.
                if samples:
                    dup = max(self._cosine_similarity(emb, prev) for prev in samples)
                    if dup > 0.999:
                        continue

                samples.append(emb)
                time.sleep(0.03)

            if not samples and frame is not None and allow_center_crop_fallback:
                center_crop = self._extract_center_crop(frame)
                if center_crop is not None:
                    emb = self._build_embedding(center_crop)
                    if emb.size > 0:
                        samples.append(emb)
            return samples
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

        faces: List[Tuple[int, int, int, int]] = []
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
                    faces = found.tolist() if hasattr(found, "tolist") else list(found)
                    break
            if faces:
                break

        if not faces:
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

    @staticmethod
    def _best_similarity(captured: List[np.ndarray], bank: np.ndarray) -> tuple[float, Optional[np.ndarray]]:
        if bank.ndim == 1:
            bank = bank.reshape(1, -1)

        best = 0.0
        best_emb: Optional[np.ndarray] = None
        for emb in captured:
            for ref in bank:
                sim = FaceAuthenticator._cosine_similarity(emb, ref)
                if sim > best:
                    best = sim
                    best_emb = emb
        return best, best_emb

    def _save_reference_bank(self, bank: np.ndarray) -> None:
        if bank.ndim == 1:
            bank = bank.reshape(1, -1)
        np.save(self.bank_path, bank)

        # Legacy single embedding for compatibility.
        centroid = np.mean(bank, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        np.save(self.embedding_path, centroid)

        payload = {
            "threshold": self.threshold,
            "samples": int(bank.shape[0]),
        }
        self.meta_path.write_text(json.dumps(payload, indent=2))

    def _append_reference(self, embedding: np.ndarray, max_size: int = 24) -> None:
        bank = self._load_reference_bank()
        if bank is None:
            new_bank = embedding.reshape(1, -1)
        else:
            if bank.ndim == 1:
                bank = bank.reshape(1, -1)
            new_bank = np.vstack([bank, embedding.reshape(1, -1)])
            if len(new_bank) > max_size:
                new_bank = new_bank[-max_size:]
        self._save_reference_bank(new_bank)

    @staticmethod
    def _merge_unique_embeddings(bank: np.ndarray, captured: List[np.ndarray], max_size: int = 24) -> np.ndarray:
        """Merge captured embeddings into bank while avoiding near-duplicates."""
        if bank.ndim == 1:
            bank = bank.reshape(1, -1)
        merged = [ref.copy() for ref in bank]
        for emb in captured:
            if emb.size == 0:
                continue
            if merged:
                dup = max(FaceAuthenticator._cosine_similarity(emb, ref) for ref in merged)
                if dup > 0.999:
                    continue
            merged.append(emb.copy())
            if len(merged) >= max_size:
                break
        if not merged:
            return bank
        arr = np.vstack([m.reshape(1, -1) for m in merged])
        if len(arr) > max_size:
            arr = arr[-max_size:]
        return arr

    def _promote_bank_on_success(
        self,
        bank: np.ndarray,
        captured: List[np.ndarray],
        best_embedding: np.ndarray,
    ) -> None:
        """
        Update the profile after a successful verify.

        For legacy single-vector profiles, promote to a richer multi-sample bank
        using current captures. This greatly improves stability on future boots.
        """
        if bank.ndim == 1:
            bank = bank.reshape(1, -1)

        if self._loaded_legacy_profile and not self.bank_path.exists():
            promoted = self._merge_unique_embeddings(bank, captured, max_size=24)
            self._save_reference_bank(promoted)
            self._loaded_legacy_profile = False
            return

        self._append_reference(best_embedding, max_size=24)

    def _enroll_from_embeddings(
        self,
        embeddings: List[np.ndarray],
        message: str = "Face profile created. Access granted, Sir.",
    ) -> FaceAuthResult:
        valid = [emb for emb in embeddings if emb.size > 0]
        if not valid:
            return FaceAuthResult(granted=False, message="No valid face embeddings captured.")
        bank = np.vstack([emb.reshape(1, -1) for emb in valid])
        self._save_reference_bank(bank)
        return FaceAuthResult(
            granted=True,
            message=message,
            similarity=1.0,
            enrolled=True,
        )

    def _load_reference_bank(self) -> Optional[np.ndarray]:
        try:
            if self.bank_path.exists():
                bank = np.load(self.bank_path)
                if bank.ndim == 1:
                    bank = bank.reshape(1, -1)
                self._loaded_legacy_profile = False
                return bank
            if self.embedding_path.exists():
                emb = np.load(self.embedding_path)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                self._loaded_legacy_profile = True
                return emb
        except Exception:
            return None
        return None

    def _effective_threshold(self, bank_size: int) -> float:
        """
        Compute a robust threshold.

        Single-sample legacy profiles are noisy, so allow a slightly lower
        threshold and then immediately upgrade the profile on success.
        """
        base = float(self.threshold)
        if bank_size <= 1:
            return max(0.72, base - 0.06)
        if bank_size == 2:
            return max(0.74, base - 0.03)
        return base

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
