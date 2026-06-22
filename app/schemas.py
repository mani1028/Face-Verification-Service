from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


EMBEDDING_SIZE = 512
DEFAULT_THRESHOLD = 0.60


def validate_embedding_vector(value: List[float], field_name: str) -> List[float]:
    """
    InsightFace ArcFace buffalo_l returns a 512-dimensional embedding.
    This validation prevents wrong or corrupted embeddings from reaching
    the recognition logic.
    """

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of float values")

    if len(value) != EMBEDDING_SIZE:
        raise ValueError(
            f"{field_name} must contain exactly {EMBEDDING_SIZE} values"
        )

    return value


class ExtractEmbeddingRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=20,
        description="Base64 encoded image string",
    )


class ExtractFacesRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=20,
        description="Base64 encoded image string for multi-face extraction",
    )


class VerifyOneToOneRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=20,
        description="Base64 encoded live/captured image",
    )

    stored_embedding: List[float] = Field(
        ...,
        description="Stored 512-dimensional face embedding",
    )

    threshold: Optional[float] = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.10,
        le=1.00,
        description="Face match threshold. Higher value means stricter matching.",
    )

    @field_validator("stored_embedding")
    @classmethod
    def validate_stored_embedding(cls, value: List[float]) -> List[float]:
        return validate_embedding_vector(value, "stored_embedding")


class CompareEmbeddingsRequest(BaseModel):
    embedding_1: List[float] = Field(
        ...,
        description="First 512-dimensional face embedding",
    )

    embedding_2: List[float] = Field(
        ...,
        description="Second 512-dimensional face embedding",
    )

    threshold: Optional[float] = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.10,
        le=1.00,
        description="Face match threshold. Higher value means stricter matching.",
    )

    @field_validator("embedding_1")
    @classmethod
    def validate_embedding_1(cls, value: List[float]) -> List[float]:
        return validate_embedding_vector(value, "embedding_1")

    @field_validator("embedding_2")
    @classmethod
    def validate_embedding_2(cls, value: List[float]) -> List[float]:
        return validate_embedding_vector(value, "embedding_2")


class CandidateEmbedding(BaseModel):
    id: str = Field(
        ...,
        min_length=1,
        description="Unique ID from your project database, for example employee_id or student_id",
    )

    embedding: List[float] = Field(
        ...,
        description="Candidate 512-dimensional face embedding",
    )

    @field_validator("embedding")
    @classmethod
    def validate_candidate_embedding(cls, value: List[float]) -> List[float]:
        return validate_embedding_vector(value, "candidate embedding")


class IdentifyRequest(BaseModel):
    image_base64: str = Field(
        ...,
        min_length=20,
        description="Base64 encoded live/captured image",
    )

    candidates: List[CandidateEmbedding] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of known candidate embeddings from your project database. Max 5000 per request.",
    )

    threshold: Optional[float] = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.10,
        le=1.00,
        description="Face match threshold. Higher value means stricter matching.",
    )

# Multipart/form-data upload support added.
# Existing base64 endpoints kept for backward compatibility.
