"""
SatQuery AI - Backend Application Entrypoint
===========================================
FastAPI server providing endpoints for remote sensing vision-language query execution,
health checks, and sample demo preset catalogs.
"""

import os
import time
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

try:
    from schemas import QueryResponse, HealthResponse, TraceInfo, TraceStep, SamplePreset
    from router import route_query
    from llm_client import analyze_imagery
except ImportError:
    from backend.schemas import QueryResponse, HealthResponse, TraceInfo, TraceStep, SamplePreset
    from backend.router import route_query
    from backend.llm_client import analyze_imagery

app = FastAPI(
    title="SatQuery AI Backend",
    description="Agentic Vision-Language Assistant for Satellite and Remote Sensing Imagery (ISRO SIH 2026)",
    version="1.0.0"
)

# Enable CORS for local development and frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(os.path.dirname(BASE_DIR), "sample-images")

# Mount sample images directory for frontend previews
if os.path.exists(SAMPLE_DIR):
    app.mount("/sample-images", StaticFiles(directory=SAMPLE_DIR), name="sample-images")


# Sample Presets Catalog
SAMPLE_PRESETS = [
    SamplePreset(
        id="flood-change",
        title="🌊 Urban Flood Inundation Analysis",
        category="change_detection",
        tag="Bi-Temporal CDVQA (2 Images)",
        description="Compare pre-event and post-event satellite imagery to map inundated streets, submerged parks, and canal breach zones.",
        image_names=["urban_before.jpg", "urban_after_flood.jpg"],
        suggested_query="Compare these before and after satellite images and identify all flood-damaged areas and inundated roads."
    ),
    SamplePreset(
        id="sar-fusion",
        title="🛰️ Optical + SAR Crop Vitality Fusion",
        category="optical_sar_fusion",
        tag="Multi-Sensor Fusion (2 Images)",
        description="Fuse visible optical RGB reflectance with Sentinel-1 SAR microwave radar backscatter for agricultural moisture mapping.",
        image_names=["agri_optical.jpg", "agri_sar.jpg"],
        suggested_query="Analyze the multi-modal optical and SAR radar imagery to determine surface roughness, soil moisture, and crop health."
    ),
    SamplePreset(
        id="port-grounding",
        title="🎯 Naval Port & Storage Grounding",
        category="grounding",
        tag="Visual Grounding (1 Image)",
        description="Detect and spatially localize fuel oil storage tanks, container ships, and dockside gantry cranes with coordinates.",
        image_names=["naval_port_grounding.jpg"],
        suggested_query="Where are the fuel oil storage tanks, container ships, and gantry cranes located? Provide spatial coordinates and quadrants."
    ),
    SamplePreset(
        id="airport-captioning",
        title="📝 International Airport Scene Captioning",
        category="captioning",
        tag="Scene Summary (1 Image)",
        description="Generate a multi-scale land-use land-cover (LULC) scene taxonomy, transit corridors, and aviation infrastructure summary.",
        image_names=["airport_hub_captioning.jpg"],
        suggested_query="Describe this satellite scene in detail, including primary land cover taxonomy, transit corridors, and aviation infrastructure."
    ),
    SamplePreset(
        id="solar-vqa",
        title="🔍 Solar Park & Turbine Counting",
        category="single_image_vqa",
        tag="Single-Image RSVQA (1 Image)",
        description="Perform precision visual question answering for solar array grid structure, substation location, and wind turbine counts.",
        image_names=["solar_park_vqa.jpg"],
        suggested_query="How many wind turbines are visible on the western perimeter, and what is the layout of the solar photovoltaic array and substation?"
    )
]


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Returns the operational status and active LLM configuration of the SatQuery AI engine."""
    active_model = "SatQuery Specialist (Qwen2-VL-2B + LoRA)"
    return HealthResponse(
        status="ok",
        service="SatQuery AI Backend Engine",
        version="1.0.0",
        active_model=active_model,
        demo_mode_available=True
    )


@app.get("/api/samples", response_model=List[SamplePreset])
async def get_sample_presets():
    """Returns the curated catalog of demo satellite image presets for 1-click testing."""
    return SAMPLE_PRESETS


@app.post("/api/query", response_model=QueryResponse)
async def query_satellite_imagery(
    query: str = Form(..., description="The user's natural language question or remote sensing task instruction"),
    effort: str = Form("medium", description="Effort level for the AI response (min, medium, max)"),
    images: List[UploadFile] = File(..., description="1 or 2 satellite imagery files in PNG, JPG, or WebP format")
):
    """
    Main endpoint for SatQuery AI:
    1. Validates query and image inputs (rejects >2 images or corrupt files).
    2. Runs agentic task routing (single_image_vqa, captioning, grounding, change_detection, optical_sar_fusion).
    3. Executes specialist prompt + Vision LLM model.
    4. Returns evidence-grounded answer, confidence score, and full execution trace.
    """
    start_time = time.time()
    steps: List[TraceStep] = []

    # 1. Validate query text
    clean_query = query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty. Please provide a remote sensing question or instruction."
        )

    # 2. Validate image count
    image_count = len(images)
    if image_count < 1 or image_count > 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image count ({image_count}). SatQuery AI supports either 1 image (VQA/Captioning/Grounding) or 2 images (Change Detection / Optical+SAR Fusion)."
        )

    # 3. Read and validate image formats via PIL
    pil_images: List[Image.Image] = []
    image_details = []
    
    for idx, uploaded_file in enumerate(images):
        try:
            content = await uploaded_file.read()
            if len(content) == 0:
                raise ValueError(f"Uploaded file '{uploaded_file.filename}' is empty (0 bytes).")
                
            img = Image.open(io.BytesIO(content))
            img.load() # Verify image integrity
            pil_images.append(img)
            image_details.append(f"Image {idx+1} ('{uploaded_file.filename}'): {img.width}x{img.height}px, Mode={img.mode}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process image '{uploaded_file.filename}'. Unsupported or corrupt image format. Error: {str(e)}"
            )

    steps.append(TraceStep(
        stage="Input Ingestion & Compatibility Check",
        description=f"Successfully ingested and verified {image_count} satellite image(s).",
        details="; ".join(image_details)
    ))

    # 4. Agentic Task Routing
    try:
        task_type, routing_reason, routing_confidence = route_query(clean_query, image_count)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )

    steps.append(TraceStep(
        stage="Agentic Task Routing",
        description=f"Routed input to specialist pipeline: `{task_type}` (Routing Confidence: {int(routing_confidence*100)}%)",
        details=routing_reason
    ))

    # 5. Specialist Prompt Assembly
    steps.append(TraceStep(
        stage="Specialist Domain Conditioning",
        description=f"Applied remote sensing system prompt tailored for `{task_type}`.",
        details="Configured domain guardrails, structured output headers, and sensor modality specifications."
    ))

    # 6. Specialist Model Execution
    answer_text, model_confidence, model_invoked = analyze_imagery(pil_images, clean_query, task_type, effort)

    steps.append(TraceStep(
        stage="Specialist Model Inference",
        description=f"Executed vision-language analysis via `{model_invoked}`.",
        details=f"Analyzed {image_count} input tensor(s) against natural language query."
    ))

    # 7. Calculate total latency
    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    
    steps.append(TraceStep(
        stage="Evidence-Grounded Response Assembly",
        description="Packaged telemetry, confidence calibration, and structured markdown response.",
        details=f"Total end-to-end latency: {elapsed_ms} ms."
    ))

    # Determine confidence grading
    # Combined heuristic from routing and model
    final_confidence = round(routing_confidence * model_confidence, 2)
    if final_confidence >= 0.90:
        confidence_label = "High Confidence (Specialist Aligned)"
    elif final_confidence >= 0.75:
        confidence_label = "Moderate Confidence"
    else:
        confidence_label = "Low Confidence (Heuristic Review Needed)"

    trace_info = TraceInfo(
        task_type=task_type,
        model_invoked=model_invoked,
        routing_reason=routing_reason,
        image_count=image_count,
        execution_time_ms=elapsed_ms,
        pipeline_steps=steps
    )

    return QueryResponse(
        answer=answer_text,
        confidence=final_confidence,
        confidence_label=confidence_label,
        task_type=task_type,
        trace=trace_info
    )
