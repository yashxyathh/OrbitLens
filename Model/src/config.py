import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Optional torch import for flexible runtime / lightweight CLI checks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

def detect_runtime_environment() -> str:
    """Detect if running inside Google Colab, Kaggle, or Local Environment."""
    if "COLAB_GPU" in os.environ or "google.colab" in str(os.environ):
        return "colab"
    elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle/working"):
        return "kaggle"
    return "local"

def get_optimal_device_and_dtype() -> Tuple[str, Any]:
    """Determine best available torch device and precision."""
    if not HAS_TORCH:
        return "cpu", "float32"
        
    if torch.cuda.is_available():
        # Check for bfloat16 support (Ampere/Ada/Hopper GPUs e.g. A100, RTX 3090/4090)
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        # Fallback to float16 for Turing / Pascal (T4, P100, V100)
        return "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

@dataclass
class SatQueryModelConfig:
    """Core Model & LoRA Configuration."""
    # Primary Recommended Model for Remote Sensing & Multi-Image Change Detection
    base_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    # Alternative Lightweight Model for Ultra-Fast Single Image Tasks
    fallback_model_id: str = "microsoft/Florence-2-large"
    
    # Image Resolution Limits (prevents OOM on high-resolution satellite tiles)
    min_pixels: int = 256 * 28 * 28      # 200,704 pixels
    max_pixels: int = 1024 * 28 * 28     # 802,816 pixels (~896x896 equivalent)
    
    # LoRA Fine-Tuning Hyperparameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_qlora: bool = True  # 4-bit quantization for free Colab/Kaggle T4 VRAM savings (< 5GB VRAM)

@dataclass
class SatQueryDataConfig:
    """Dataset Storage & Processing Configuration."""
    root_dir: Path = Path("./satquery_data")
    raw_dir: Path = Path("./satquery_data/raw")
    processed_dir: Path = Path("./satquery_data/processed")
    cache_dir: Path = Path("./satquery_data/cache")
    
    # Task identifiers
    TASK_VQA: str = "vqa"
    TASK_CAPTIONING: str = "captioning"
    TASK_GROUNDING: str = "grounding"
    TASK_CHANGE_DETECTION: str = "change_detection"
    
    # Dataset Names
    DATASET_BIGEARTHNET: str = "bigearthnet"
    DATASET_RSVQA: str = "rsvqa"
    DATASET_VRSBENCH: str = "vrsbench"
    DATASET_CDVQA: str = "cdvqa"

    # Default Train/Val/Test Split
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42

# Default Prompt Templates for Remote Sensing Tasks
TASK_PROMPT_TEMPLATES: Dict[str, str] = {
    "vqa": "Analyze this satellite image and answer the following question precisely: {query}",
    "captioning": "Provide a detailed remote sensing description of this satellite scene, including terrain type, land cover, structures, and notable geographic features.",
    "grounding": "Locate all instances of '{query}' in this satellite image. Output normalized bounding boxes in [ymin, xmin, ymax, xmax] format with class labels.",
    "change_detection": "Compare the two bi-temporal satellite images (Image 1 captured at Time 1, and Image 2 captured at Time 2). Describe all structural, land-use, environmental, or infrastructural changes that occurred between Time 1 and Time 2: {query}"
}

# Instantiate global configs
RUNTIME_ENV = detect_runtime_environment()
DEVICE, DTYPE = get_optimal_device_and_dtype()
MODEL_CONFIG = SatQueryModelConfig()
DATA_CONFIG = SatQueryDataConfig()
