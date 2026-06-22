# Universal Face Recognition Service

## README.md Content

# Universal Face Recognition Service

Production-ready Face Recognition API service using:

- InsightFace ArcFace
- buffalo_l ONNX model
- ONNX Runtime
- FastAPI
- Docker

This service is designed to work as a separate face recognition model server.  
Any project can use this service by calling its APIs.

---

## 1. Project Purpose

This service provides face recognition features through API calls.

It can be used for:

- Face registration
- Face verification
- Employee verification
- Student attendance
- Teacher attendance
- Login verification
- Visitor verification
- Any future face recognition project

The service does not directly depend on any specific project database.

Each client project should manage its own:

- Users
- Database
- Attendance records
- Business logic
- Stored embeddings

This service only handles:

- Face detection
- Face embedding extraction
- Face comparison
- Match / no-match result

---

## 2. Architecture

```text
Client Project
    |
    | API Request
    v
Universal Face Recognition Service
    |
    | InsightFace + buffalo_l + ONNX Runtime
    v
Response: embedding / similarity / matched resu

## OPTIMIZATION_NOTES.md Content

# Optimizations Applied

## Added
- multipart/form-data upload endpoint
- configurable FACE_DET_SIZE
- improved Gunicorn worker count
- async image upload handling
- CPU optimized detection size

## Recommended Next Steps
- Redis embedding cache
- Rate limiting
- pgvector / FAISS support
- Liveness detectio

## Readme.md Content

# Universal Face Verification Service

A high-performance open-source face verification API built with FastAPI and InsightFace.

It provides:

- Face Detection
- Face Embedding Extraction
- 1:1 Face Verification
- Embedding Comparison
- Docker Deployment
- OpenAPI Documentation

Designed for developers building:

- Attendance Systems
- Authentication Platforms
- Access Control
- KYC Verification
- Visitor Management
- Identity Verification Solutions

---

## Features

✅ Face Detection

✅ Face Embedding Extraction

✅ 1:1 Face Verification

✅ Embedding Similarity Comparison

✅ FastAPI REST APIs

✅ Docker Support

✅ Swagger Documentation

✅ Production Ready

---

## Architecture

```text
Image
   ↓
Face Detection
   ↓
Face Embedding
   ↓
Similarity Comparison
   ↓
Match Result
```

---

## Technology Stack

- Python
- FastAPI
- InsightFace
- ONNX Runtime
- Docker
- Gunicorn

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/universal-face-verification-service.git

cd universal-face-verification-service
```

### Create Environment File

```bash
cp .env.example .env
```

### Run with Docker

```bash
docker compose up -d
```

Service will start at:

```text
http://localhost:8001
```

---

## API Documentation

Swagger UI

```text
http://localhost:8001/docs
```

Redoc

```text
http://localhost:8001/redoc
```

---

## Authentication

All endpoints require:

```http
x-api-key: YOUR_API_KEY
```

---

## Endpoints

### Health Check

```http
GET /health
```

### Extract Face Embedding

```http
POST /extract-embedding
```

Returns:

```json
{
  "embedding": [...]
}
```

---

### Verify Two Faces

```http
POST /verify
```

Request:

```json
{
  "image1": "base64",
  "image2": "base64"
}
```

Response:

```json
{
  "matched": true,
  "score": 0.87
}
```

---

### Compare Embeddings

```http
POST /compare-embeddings
```

---

## Docker

Build:

```bash
docker build -t universal-face-service .
```

Run:

```bash
docker run -p 8001:8001 universal-face-service
```

---

## Performance

Typical Verification Time:

- CPU: 200-500ms
- GPU: 30-100ms

Embedding Size:

```text
512 Dimensions
```

Model:

```text
InsightFace Buffalo_L
```

---

## Example Use Cases

### Attendance System

```text
Student Face
      ↓
Stored Embedding
      ↓
Verification
      ↓
Attendance Marked
```

### Authentication

```text
User Face
      ↓
Stored Face
      ↓
Verification
      ↓
Login Approved
```

### KYC Verification

```text
Selfie
      ↓
ID Photo
      ↓
Verification
      ↓
Identity Confirmed
```

---

## Security Notes

- Do not store user images without consent.
- Use HTTPS in production.
- Rotate API keys regularly.
- Follow local privacy regulations.

---

## Roadmap

- Multipart Upload Support
- Rate Limiting
- Liveness Detection
- Batch Verification
- GPU Auto Scaling

---

## Contributing

Contributions are welcome.

1. Fork Repository
2. Create Feature Branch
3. Commit Changes
4. Open Pull Request

---

## License

MIT Licen

