"""
SatQuery AI - Specialist Model Execution Layer (Production)
==============================================
This module serves as the bridge between the FastAPI routing layer
and the local fine-tuned Qwen2-VL specialist model.
"""

from typing import List, Tuple
from PIL import Image
import logging

try:
    from inference import run_specialist_model
except ImportError:
    from backend.inference import run_specialist_model

logger = logging.getLogger("SatQuery-Execution")

def analyze_imagery(
    images: List[Image.Image], 
    query: str, 
    task_type: str
) -> Tuple[str, float, str]:
    """
    Executes the local fine-tuned specialist vision model on the input satellite imagery.
    
    Returns:
        Tuple[str, float, str]:
            - answer text (markdown formatted)
            - confidence score (float 0.0 - 1.0)
            - model identifier string
    """
    try:
        # Run local inference
        answer_text, confidence_score = run_specialist_model(images, query, task_type)
        model_invoked = "SatQuery Specialist (Qwen2-VL-2B + LoRA)"
        
        return answer_text, confidence_score, model_invoked
        
    except Exception as e:
        logger.error(f"Failed to execute local model: {e}", exc_info=True)
        return f"[Error] Model execution failed: {str(e)}", 0.0, "SatQuery Specialist (Error)"
