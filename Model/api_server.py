"""
SatQuery AI - FastAPI Production API Reference Implementation
Demonstrates how the backend team can integrate `run_specialist_model` from `model_inference.py`.

Run locally with:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import time
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from model_inference import run_specialist_model

app = FastAPI(
    title="SatQuery AI - Vision-Language Satellite Intelligence API",
    description="Local open-source Vision-Language Model serving remote sensing VQA, Captioning, Grounding, and Change Detection.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceResponse(BaseModel):
    task_type: str = Field(..., example="vqa")
    query: str = Field(..., example="How many aircraft are on the runway?")
    response: str = Field(..., example="There are 3 aircraft detected on the primary runway.")
    confidence: float = Field(..., example=0.94)
    latency_ms: float = Field(..., example=420.5)

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint confirming VRAM and model status."""
    try:
        import torch
        cuda_ready = torch.cuda.is_available()
        vram_allocated = torch.cuda.memory_allocated() / (1024**2) if cuda_ready else 0.0
    except ImportError:
        cuda_ready = False
        vram_allocated = 0.0
        
    return {
        "status": "online",
        "service": "SatQuery AI",
        "cuda_available": cuda_ready,
        "vram_allocated_mb": round(vram_allocated, 2)
    }

@app.post("/v1/analyze", response_model=InferenceResponse, tags=["Vision-Language Inference"])
async def analyze_single_satellite_image(
    image: UploadFile = File(..., description="Satellite image tile (PNG, JPEG, GeoTIFF)"),
    query: str = Form("Describe this satellite scene in detail.", description="Question or prompt"),
    task_type: str = Form("vqa", description="vqa | captioning | grounding")
):
    """
    Execute Single-Image VQA, Dense Captioning, or Visual Grounding.
    """
    t0 = time.perf_counter()
    
    # Read & parse image
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )

    # Call specialist model inference function
    answer, confidence = run_specialist_model(
        images=[pil_img],
        query=query,
        task_type=task_type
    )

    latency = round((time.perf_counter() - t0) * 1000, 2)

    return InferenceResponse(
        task_type=task_type,
        query=query,
        response=answer,
        confidence=confidence,
        latency_ms=latency
    )

@app.post("/v1/change-detection", response_model=InferenceResponse, tags=["Bi-Temporal Change Detection"])
async def detect_temporal_changes(
    image_t1: UploadFile = File(..., description="Satellite Image at Time 1 (Before)"),
    image_t2: UploadFile = File(..., description="Satellite Image at Time 2 (After)"),
    query: str = Form("Identify all new buildings, infrastructure changes, and deforestation between Time 1 and Time 2.")
):
    """
    Execute Bi-Temporal Change Detection comparing two distinct satellite timestamps.
    """
    t0 = time.perf_counter()

    try:
        bytes_t1 = await image_t1.read()
        bytes_t2 = await image_t2.read()
        pil_t1 = Image.open(io.BytesIO(bytes_t1)).convert("RGB")
        pil_t2 = Image.open(io.BytesIO(bytes_t2)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image pair: {str(e)}"
        )

    # Execute bi-temporal specialist forward pass
    answer, confidence = run_specialist_model(
        images=[pil_t1, pil_t2],
        query=query,
        task_type="change_detection"
    )

    latency = round((time.perf_counter() - t0) * 1000, 2)

    return InferenceResponse(
        task_type="change_detection",
        query=query,
        response=answer,
        confidence=confidence,
        latency_ms=latency
    )
