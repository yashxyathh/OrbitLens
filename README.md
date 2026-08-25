# SatQuery AI 🛰️
### Agentic Vision-Language Assistant for Satellite & Remote Sensing Imagery
**Smart India Hackathon (SIH 2026) Prototype — ISRO Problem Statement**

---

## 🌟 Overview

**SatQuery AI** is an agentic vision-language assistant engineered for satellite and earth-observation imagery. It enables defense analysts, disaster response commanders, and urban planners to ask natural language questions over single images, bi-temporal change pairs, or multi-modal Optical+SAR (Synthetic Aperture Radar) sensor combinations.

The system features a **5-stage modular agentic pipeline** with an intelligent routing layer that automatically directs queries to specialized remote sensing tasks and constructs domain-conditioned vision-language inferences with full execution telemetry.

---

## 🏗️ 5-Stage Agentic Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Query & Tensor Ingestion                                                 │
│    User uploads 1 or 2 satellite images (Optical, SAR, or Bi-Temporal Pair) │
│    + Natural language instruction or analytical question                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Compatibility & Integrity Check (backend/main.py)                        │
│    • Validates image count (1 or 2 images)                                  │
│    • Pillow PIL integrity check (PNG, JPEG, WebP)                           │
│    • Rejects corrupt inputs or >2 payload violations                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Agentic Task Routing (backend/router.py)                                 │
│    Analyzes image count and query semantics into 5 RS task pipelines:       │
│    • single_image_vqa    • captioning          • grounding                  │
│    • change_detection    • optical_sar_fusion                               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Specialist Model Execution (backend/llm_client.py)                       │
│    • Injects task-specific RS system prompts (RSVQA / VRSBench / CDVQA)     │
│    • Ingests image base64 blocks + query into Anthropic Claude Vision API   │
│    • Graceful fallback simulation engine for offline/demo reliability       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. Evidence-Grounded Response & Trace (backend/schemas.py)                  │
│    Returns JSON { answer, confidence, confidence_label, task_type, trace }  │
│    Interactive Telemetry: Step-by-step latency, routing reason & audit logs │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Task Routing Taxonomy (`backend/router.py`)

| Task Type | Image Count | Key Trigger Patterns | Target Downstream Specialist |
| :--- | :--- | :--- | :--- |
| **`change_detection`** | 2 Images | `change`, `before and after`, `flood`, `damage`, `expansion`, `compare` | Bi-Temporal CDVQA Specialist |
| **`optical_sar_fusion`** | 2 Images | `sar`, `radar`, `backscatter`, `microwave`, `soil moisture`, `roughness` | Multi-Sensor Optical-SAR Fusion |
| **`grounding`** | 1 Image | `where is`, `locate`, `bounding box`, `coordinates`, `pinpoint`, `quadrant` | Spatial Visual Grounding Specialist |
| **`captioning`** | 1 Image | `describe`, `caption`, `overview`, `summary`, `land use`, `land cover` | VRSBench Scene Captioner |
| **`single_image_vqa`** | 1 Image | `how many`, `count`, `what is`, `vessel presence`, `road width` | RSVQA Analytical Specialist |

---

## 🚀 Quickstart & Setup

### Prerequisites
- Python 3.11+
- Node.js v18+ & npm

### 1. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Install Python requirements
pip install -r requirements.txt

# (Optional) Set your Anthropic API Key in environment or .env
# Windows PowerShell:
$env:ANTHROPIC_API_KEY="your_api_key_here"

# Linux / macOS:
export ANTHROPIC_API_KEY="your_api_key_here"

# Launch FastAPI Server (uses python -m uvicorn for universal Windows/macOS/Linux compatibility)
python -m uvicorn main:app --reload --port 8000
```
> 💡 *Note: If `ANTHROPIC_API_KEY` is omitted, SatQuery AI automatically operates in high-fidelity **Remote Sensing Domain Simulation Mode**, ensuring a flawless, un-crashable presentation for hackathon evaluators.*

### 2. Frontend Setup

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite Development Server
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## 🔑 Environment Variables (`.env`)

Copy `.env.example` to `.env` in the root or backend directory:

```env
# Anthropic Claude API Key for Vision LLM (e.g. claude-3-7-sonnet-20250219 or claude-3-5-sonnet-latest)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Anthropic Model Identifier (optional, default: claude-3-7-sonnet-20250219)
ANTHROPIC_MODEL=claude-3-7-sonnet-20250219

# Port (optional, default: 8000)
PORT=8000
```

---

## 🎯 1-Click Evaluation Presets

The interface includes 5 preloaded sample benchmarks:

1. **🌊 Urban Flood Inundation Analysis** (`change_detection` — 2 Images):
   * Images: `urban_before.jpg` + `urban_after_flood.jpg`
   * Query: *"Compare these before and after satellite images and identify all flood-damaged areas and inundated roads."*
2. **🛰️ Optical + SAR Crop Vitality Fusion** (`optical_sar_fusion` — 2 Images):
   * Images: `agri_optical.jpg` + `agri_sar.jpg`
   * Query: *"Analyze the multi-modal optical and SAR radar imagery to determine surface roughness, soil moisture, and crop health."*
3. **🎯 Naval Port & Storage Grounding** (`grounding` — 1 Image):
   * Image: `naval_port_grounding.jpg`
   * Query: *"Where are the fuel oil storage tanks, container ships, and gantry cranes located? Provide spatial coordinates and quadrants."*
4. **📝 International Airport Scene Captioning** (`captioning` — 1 Image):
   * Image: `airport_hub_captioning.jpg`
   * Query: *"Describe this satellite scene in detail, including primary land cover taxonomy, transit corridors, and aviation infrastructure."*
5. **🔍 Solar Park & Turbine Counting** (`single_image_vqa` — 1 Image):
   * Image: `solar_park_vqa.jpg`
   * Query: *"How many wind turbines are visible on the western perimeter, and what is the layout of the solar photovoltaic array and substation?"*

---

## 🔌 Downstream Fine-Tuning Roadmap

SatQuery AI's execution layer (`backend/llm_client.py`) provides clean plugin seams to swap in specialized fine-tuned checkpoints:

- **BigEarthNet-MM**: Multi-spectral Sentinel-2 + Sentinel-1 SAR classification.
- **VRSBench & RSICD**: Dense satellite scene description and visual grounding.
- **RSVQA**: High/Low-resolution visual question answering.
- **CDVQA**: Bi-temporal disaster damage and flood change quantification.

---

## 📡 API Reference

### `POST /api/query`
- **Request**: `multipart/form-data`
  - `query`: string (natural language instruction)
  - `images`: array of 1 or 2 image files (`PNG`, `JPG`, `WebP`)
- **Response**:
```json
{
  "answer": "### 🛰️ Bi-Temporal Change Detection Analysis...",
  "confidence": 0.94,
  "confidence_label": "High Confidence (Specialist Aligned)",
  "task_type": "change_detection",
  "trace": {
    "task_type": "change_detection",
    "model_invoked": "Anthropic claude-3-7-sonnet-20250219 (Live Vision API)",
    "routing_reason": "Dual-image input detected with bi-temporal change keywords...",
    "image_count": 2,
    "execution_time_ms": 420.5,
    "pipeline_steps": [
      {
        "stage": "Input Ingestion & Compatibility Check",
        "description": "Successfully ingested and verified 2 satellite image(s).",
        "details": "Image 1: 1024x1024px, Mode=RGB; Image 2: 1024x1024px, Mode=RGB",
        "status": "completed"
      },
      {
        "stage": "Agentic Task Routing",
        "description": "Routed input to specialist pipeline: change_detection",
        "status": "completed"
      }
    ]
  }
}
```

### `GET /api/health`
- **Response**: `{"status": "ok", "service": "SatQuery AI Backend Engine", "version": "1.0.0", "active_model": "...", "demo_mode_available": true}`

### `GET /api/samples`
- **Response**: List of sample evaluation presets with pre-configured queries and image filenames.
