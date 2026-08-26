"""
SatQuery AI - Production Inference Wrapper
CRITICAL HANDOFF INTERFACE FOR BACKEND / FASTAPI INTEGRATION

Maintains cached model & weights in VRAM to prevent re-loading on each request.
Supports single-image and multi-image (bi-temporal change detection) queries with
automatic precision selection, memory caching, and confidence score estimation.
"""

import logging
from typing import List, Tuple, Optional, Any
from PIL import Image
import torch

from src.config import MODEL_CONFIG, DEVICE, DTYPE, TASK_PROMPT_TEMPLATES

logger = logging.getLogger("SatQuery-Inference")

# Global VRAM Model & Processor Cache (Singleton pattern)
_MODEL_CACHE: Optional[Any] = None
_PROCESSOR_CACHE: Optional[Any] = None


def get_or_load_model(adapter_path: Optional[str] = None):
    """
    Lazy-loads and caches the Vision-Language Model and Processor in VRAM.
    Prevents costly re-initialization on subsequent FastAPI requests.
    """
    global _MODEL_CACHE, _PROCESSOR_CACHE

    if _MODEL_CACHE is not None and _PROCESSOR_CACHE is not None:
        return _MODEL_CACHE, _PROCESSOR_CACHE

    logger.info("Initializing SatQuery Vision-Language Specialist Model into VRAM...")

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(
        MODEL_CONFIG.base_model_id,
        min_pixels=MODEL_CONFIG.min_pixels,
        max_pixels=MODEL_CONFIG.max_pixels
    )

    # 2. Load Base Model with optimized precision
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_CONFIG.base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )

    # 3. Load LoRA specialist adapters if provided
    if adapter_path:
        logger.info(f"Attaching fine-tuned SatQuery LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    _MODEL_CACHE = model
    _PROCESSOR_CACHE = processor
    logger.info("SatQuery Model successfully cached in VRAM.")
    return _MODEL_CACHE, _PROCESSOR_CACHE


def run_specialist_model(images: list, query: str, task_type: str) -> tuple[str, float]:
    """
    Production inference entry point for SatQuery AI.

    Args:
        images (list): List of PIL.Image objects (1 image for VQA/Caption/Grounding, 2 images for Change Detection).
        query (str): User question or task instruction.
        task_type (str): Task identifier: 'vqa' | 'captioning' | 'grounding' | 'change_detection'.

    Returns:
        tuple[str, float]: (Generated text response, confidence score in range [0.0, 1.0]).
    """
    if not images:
        raise ValueError("At least one image must be supplied to run_specialist_model.")
        
    if task_type == "change_detection" and len(images) < 2:
        raise ValueError("Change detection task requires exactly 2 images (Time 1 and Time 2).")

    # Format images to RGB
    pil_images = [img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB") for img in images]

    # Format task-specific prompt
    template = TASK_PROMPT_TEMPLATES.get(task_type, "{query}")
    formatted_prompt = template.format(query=query) if "{query}" in template else f"{template}\nUser Query: {query}"

    # Build conversation payload
    user_content = []
    for img in pil_images:
        user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": formatted_prompt})

    conversation = [
        {"role": "user", "content": user_content}
    ]

    try:
        model, processor = get_or_load_model()
        from qwen_vl_utils import process_vision_info

        # Preprocess text & vision
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to target device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward pass with VRAM management
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Decode generated text
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Compute confidence score from token logits
        if outputs.scores:
            stacked_scores = torch.stack(outputs.scores, dim=1) # [batch, seq_len, vocab_size]
            probs = torch.softmax(stacked_scores, dim=-1)
            token_probs = torch.gather(probs[0], dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)
            confidence_score = float(torch.mean(token_probs).item())
        else:
            confidence_score = 0.95

        return response_text, confidence_score

    except Exception as e:
        logger.error(f"Inference error in run_specialist_model: {e}", exc_info=True)
        # Safe fallback for API robustness
        return f"[Error] Model processing failed: {str(e)}", 0.0
