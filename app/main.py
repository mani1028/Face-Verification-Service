from fastapi import FastAPI, HTTPException, status, UploadFile, File

from app.face_engine import face_engine
from app.schemas import (
    ExtractEmbeddingRequest,
    ExtractFacesRequest,
    VerifyOneToOneRequest,
    CompareEmbeddingsRequest,
    IdentifyRequest,
)
from app.config import FACE_MATCH_THRESHOLD, MAX_IMAGE_SIZE_MB, APP_NAME, APP_VERSION
import logging
import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("frs")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="A universal InsightFace ArcFace-based face recognition API service. No API key required.",
)


from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    errors = exc.errors()
    first = errors[0] if errors else {}
    return JSONResponse(status_code=422, content={
        "success": False,
        "message": first.get("msg", "Invalid request payload"),
        "field": " -> ".join(str(x) for x in first.get("loc", [])),
    })


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "model": "InsightFace ArcFace buffalo_l",
        "version": APP_VERSION,
    }


@app.post("/api/v1/extract-embedding")
def extract_embedding(
    payload: ExtractEmbeddingRequest,
):
    """
    Extract one face embedding from a single-face image.

    Use this endpoint for:
    - employee registration
    - teacher registration
    - student registration
    - face login registration

    This endpoint expects exactly one face in the image.
    """

    try:
        faces = face_engine.extract_faces(payload.image_base64)

        if len(faces) == 0:
            return {
                "success": False,
                "face_count": 0,
                "reason": "no_face_detected",
                "message": "No face found. Ensure the face is well-lit, not blurry, and clearly visible.",
            }

        if len(faces) > 1:
            return {
                "success": False,
                "face_count": len(faces),
                "reason": "multiple_faces",
                "message": f"{len(faces)} faces detected. Only one face is allowed for registration.",
            }

        face = faces[0]

        return {
            "success": True,
            "face_count": 1,
            "embedding": face["embedding"],
            "bbox": face["bbox"],
            "det_score": face["det_score"],
            "message": "Face embedding extracted successfully",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract face embedding: {str(e)}",
        )


@app.post("/api/v1/extract-faces")
def extract_faces(
    payload: ExtractFacesRequest,
):
    """
    Extract all face embeddings from an image.

    Use this endpoint for:
    - group photo recognition
    - classroom attendance
    - multiple student upload verification
    - video frame face extraction

    This endpoint allows multiple faces.
    """

    try:
        faces = face_engine.extract_faces(payload.image_base64)

        if len(faces) == 0:
            return {
                "success": False,
                "face_count": 0,
                "faces": [],
                "reason": "no_face_detected",
                "message": "No face found. Ensure faces are well-lit and clearly visible.",
            }

        return {
            "success": True,
            "face_count": len(faces),
            "faces": faces,
            "message": "Faces extracted successfully",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract faces: {str(e)}",
        )


@app.post("/api/v1/verify-one-to-one")
def verify_one_to_one(
    payload: VerifyOneToOneRequest,
):
    """
    Verify one live/captured face image against one stored embedding.

    Use this endpoint for:
    - teacher verification
    - employee verification
    - face login
    - one-to-one identity verification

    This endpoint expects exactly one face in the image.
    """

    try:
        faces = face_engine.extract_faces(payload.image_base64)

        if len(faces) == 0:
            return {
                "success": False,
                "matched": False,
                "face_count": 0,
                "reason": "no_face_detected",
                "message": "No face found. Ensure the face is well-lit, not blurry, and clearly visible.",
            }

        if len(faces) > 1:
            return {
                "success": False,
                "matched": False,
                "face_count": len(faces),
                "reason": "multiple_faces",
                "message": f"{len(faces)} faces detected. Only one face is allowed for verification.",
            }

        threshold = payload.threshold if payload.threshold is not None else FACE_MATCH_THRESHOLD
        live_embedding = faces[0]["embedding"]

        result = face_engine.verify_one_to_one(
            live_embedding=live_embedding,
            stored_embedding=payload.stored_embedding,
            threshold=threshold,
        )

        logger.info(f"verify | det={faces[0]['det_score']:.3f} sim={result['similarity']:.4f} matched={result['matched']}")

        return {
            "success": True,
            "matched": result["matched"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "face_count": 1,
            "bbox": faces[0]["bbox"],
            "det_score": faces[0]["det_score"],
            "message": "Face matched" if result["matched"] else "Face not matched",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify face: {str(e)}",
        )


@app.post("/api/v1/compare-embeddings")
def compare_embeddings(
    payload: CompareEmbeddingsRequest,
):
    """
    Compare two existing face embeddings.

    Use this endpoint when your project already has two embeddings
    and only wants similarity/match result.
    """

    try:
        threshold = payload.threshold if payload.threshold is not None else FACE_MATCH_THRESHOLD

        result = face_engine.verify_one_to_one(
            live_embedding=payload.embedding_1,
            stored_embedding=payload.embedding_2,
            threshold=threshold,
        )

        return {
            "success": True,
            "matched": result["matched"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "message": "Embeddings matched" if result["matched"] else "Embeddings not matched",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare embeddings: {str(e)}",
        )


@app.post("/api/v1/identify")
def identify(
    payload: IdentifyRequest,
):
    """
    Identify one live/captured face against many candidate embeddings.

    Use this endpoint for:
    - employee identification
    - student identification
    - visitor identification
    - one-to-many recognition

    This endpoint expects exactly one face in the input image.
    The candidates should come from the client project's own database.
    """

    try:
        faces = face_engine.extract_faces(payload.image_base64)

        if len(faces) == 0:
            return {
                "success": False,
                "matched": False,
                "face_count": 0,
                "best_match_id": None,
                "similarity": 0.0,
                "reason": "no_face_detected",
                "message": "No face found. Ensure the face is well-lit, not blurry, and clearly visible.",
            }

        if len(faces) > 1:
            return {
                "success": False,
                "matched": False,
                "face_count": len(faces),
                "best_match_id": None,
                "similarity": 0.0,
                "reason": "multiple_faces",
                "message": f"{len(faces)} faces detected. Only one face is allowed for identification.",
            }

        threshold = payload.threshold if payload.threshold is not None else FACE_MATCH_THRESHOLD
        live_embedding = faces[0]["embedding"]

        candidates_dicts = [{"id": c.id, "embedding": c.embedding} for c in payload.candidates]
        result = face_engine.identify_one_to_many(
            live_embedding=live_embedding,
            candidates=candidates_dicts,
            threshold=threshold,
        )

        logger.info(f"identify | candidates={len(payload.candidates)} sim={result['similarity']:.4f} matched={result['matched']}")

        return {
            "success": True,
            "matched": result["matched"],
            "face_count": 1,
            "best_match_id": result["best_match_id"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "bbox": faces[0]["bbox"],
            "det_score": faces[0]["det_score"],
            "message": "Face identified" if result["matched"] else "No matching face found",
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to identify face: {str(e)}",
        )

@app.post("/api/v2/extract-embedding-upload")
async def extract_embedding_upload(image: UploadFile = File(...)):
    try:
        max_bytes = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)
        contents = await image.read(max_bytes + 1)

        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds the {MAX_IMAGE_SIZE_MB} MB limit",
            )

        np_arr = np.frombuffer(contents, np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if decoded is None:
            return {"success": False, "message": "Invalid image file — could not decode"}

        faces = face_engine.extract_faces_from_array(decoded)

        if len(faces) == 0:
            return {
                "success": False,
                "face_count": 0,
                "reason": "no_face_detected",
                "message": "No face found. Ensure the face is well-lit, not blurry, and clearly visible.",
            }

        if len(faces) > 1:
            return {
                "success": False,
                "face_count": len(faces),
                "reason": "multiple_faces",
                "message": f"{len(faces)} faces detected. Only one face is allowed.",
            }

        face = faces[0]
        return {
            "success": True,
            "face_count": 1,
            "embedding": face["embedding"],
            "bbox": face["bbox"],
            "det_score": face["det_score"],
            "message": "Face embedding extracted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to extract face embedding: {str(e)}")
