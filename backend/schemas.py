from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# Valid remote-sensing agentic task types
TaskType = Literal[
    "single_image_vqa",
    "captioning",
    "grounding",
    "change_detection",
    "optical_sar_fusion"
]

class TraceStep(BaseModel):
    stage: str
    description: str
    details: Optional[str] = None
    status: str = "completed"

class TraceInfo(BaseModel):
    task_type: TaskType
    model_invoked: str
    routing_reason: str
    image_count: int
    execution_time_ms: float
    pipeline_steps: List[TraceStep] = Field(default_factory=list)

class QueryResponse(BaseModel):
    answer: str
    confidence: float = Field(
        ..., 
        description="Confidence score heuristic (0.0 to 1.0) calibrated for the selected specialist task"
    )
    confidence_label: str = Field(
        ...,
        description="Human-readable confidence grade: High, Moderate, or Low"
    )
    task_type: TaskType
    trace: TraceInfo

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "SatQuery AI Backend Engine"
    version: str = "1.0.0"
    active_model: str
    demo_mode_available: bool = True

class SamplePreset(BaseModel):
    id: str
    title: str
    category: TaskType
    tag: str
    description: str
    image_names: List[str]
    suggested_query: str
