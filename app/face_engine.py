# Optimized for CPU inference and multipart uploads
import base64
import binascii
from typing import Any, Dict, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config import (
    FACE_MODEL_NAME,
    FACE_DEVICE,
    FACE_QUALITY_THRESHOLD,
    FACE_DET_SIZE,
    MAX_IMAGE_SIZE_MB,
)


EMBEDDING_SIZE = 512


class FaceEngine:
    def __init__(self):
        """
        Initializes InsightFace FaceAnalysis model.

        FACE_DEVICE:
            cpu -> CPUExecutionProvider
            gpu -> CUDAExecutionProvider + CPUExecutionProvider fallback
        """

        device = FACE_DEVICE.lower().strip()

        if device == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self.app = FaceAnalysis(
            name=FACE_MODEL_NAME,
            providers=providers,
        )

        self.app.prepare(
            ctx_id=ctx_id,
            det_size=(FACE_DET_SIZE, FACE_DET_SIZE),
        )

    def decode_base64_image(self, image_base64: str):
        """
        Converts base64 image string into OpenCV image.

        Supports both:
            raw base64 string
            data URL format: data:image/jpeg;base64,...
        """

        try:
            if not image_base64 or not isinstance(image_base64, str):
                raise ValueError("image_base64 must be a non-empty string")

            image_base64 = image_base64.strip()

            if len(image_base64) * 0.75 > MAX_IMAGE_SIZE_MB * 1024 * 1024:
                from fastapi import HTTPException, status as http_status
                raise HTTPException(
                    status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image exceeds the {MAX_IMAGE_SIZE_MB} MB limit",
                )

            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            image_bytes = base64.b64decode(image_base64, validate=True)
            max_bytes = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)
            if len(image_bytes) > max_bytes:
                from fastapi import HTTPException, status as http_status
                raise HTTPException(
                    status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Image exceeds the {MAX_IMAGE_SIZE_MB} MB limit",
                )
            np_arr = np.frombuffer(image_bytes, np.uint8)

            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Invalid image data")

            return image

        except binascii.Error:
            raise ValueError("Invalid base64 image string")

        except Exception as e:
            raise ValueError(f"Image decode failed: {str(e)}")

    def preprocess_image(self, image):
        """Normalise contrast using CLAHE for better detection in poor lighting."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def is_blurry(self, image) -> bool:
        """Returns True if the image is too blurry for reliable face embedding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < 80.0

    def extract_faces_from_array(self, image) -> list:
        """Extracts valid faces from an already-decoded OpenCV image array."""
        image = self.preprocess_image(image)
        raw_faces = self.app.get(image)

        if not raw_faces:
            return []

        valid_faces = []

        for face in raw_faces:
            det_score = float(face.det_score)

            if det_score < FACE_QUALITY_THRESHOLD:
                continue

            if face.normed_embedding is None:
                continue

            embedding = face.normed_embedding.astype(np.float32).tolist()

            if len(embedding) != EMBEDDING_SIZE:
                continue

            bbox = face.bbox.astype(float).tolist()

            valid_faces.append({
                "bbox": bbox,
                "det_score": det_score,
                "embedding": embedding,
            })

        valid_faces.sort(key=lambda item: item["det_score"], reverse=True)
        return valid_faces

    def extract_faces(self, image_base64: str) -> List[Dict[str, Any]]:
        """
        Extracts all valid faces from an image.

        Returns:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "det_score": 0.98,
                    "embedding": [512 float values]
                }
            ]

        Used by:
            /api/v1/extract-embedding
            /api/v1/extract-faces
            /api/v1/verify-one-to-one
            /api/v1/identify
        """

        image = self.decode_base64_image(image_base64)
        return self.extract_faces_from_array(image)

    def cosine_similarity(self, emb1, emb2) -> float:
        """
        Calculates cosine similarity between two face embeddings.

        Since InsightFace normed_embedding is already normalized,
        this still re-normalizes safely to avoid bad input issues.
        """

        a = np.array(emb1, dtype=np.float32)
        b = np.array(emb2, dtype=np.float32)

        if a.shape[0] != EMBEDDING_SIZE or b.shape[0] != EMBEDDING_SIZE:
            raise ValueError(
                f"Both embeddings must contain exactly {EMBEDDING_SIZE} values"
            )

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError("Embedding vector cannot be zero vector")

        a = a / norm_a
        b = b / norm_b

        return float(np.dot(a, b))

    def verify_one_to_one(self, live_embedding, stored_embedding, threshold: float):
        """
        Compares live embedding with stored embedding.

        Returns:
            {
                "matched": true/false,
                "similarity": 0.72,
                "threshold": 0.60
            }
        """

        if threshold is None:
            raise ValueError("threshold is required")

        threshold = float(threshold)

        if threshold < 0.10 or threshold > 1.00:
            raise ValueError("threshold must be between 0.10 and 1.00")

        similarity = self.cosine_similarity(live_embedding, stored_embedding)

        return {
            "matched": similarity >= threshold,
            "similarity": similarity,
            "threshold": threshold,
        }

    def identify_one_to_many(self, live_embedding, candidates, threshold: float) -> dict:
        """Vectorised one-to-many matching using numpy matrix operations."""
        if not candidates:
            return {"matched": False, "best_match_id": None, "similarity": 0.0, "threshold": threshold}

        ids = [c.get("id") for c in candidates]
        matrix = np.array([c.get("embedding") for c in candidates], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        live = np.array(live_embedding, dtype=np.float32)
        live = live / np.linalg.norm(live)

        scores = matrix @ live
        best_idx = int(np.argmax(scores))
        best_similarity = float(scores[best_idx])
        best_match_id = ids[best_idx]
        matched = best_similarity >= threshold

        return {
            "matched": matched,
            "best_match_id": best_match_id if matched else None,
            "similarity": best_similarity,
            "threshold": threshold,
        }


face_engine = FaceEngine()