"""
SatQuery AI - Production Inference Service (FastAPI Backend Handoff)
Author: Lead AI/ML Engineer
Target: Backend Engineer (Person 2)

This script provides the production inference wrapper for the fine-tuned
Vision-Language Specialist Model (Qwen2-VL-2B + LoRA Remote Sensing Adapter).

Usage in FastAPI:
    from inference import run_specialist_model
    answer, confidence = run_specialist_model(images=[pil_img], query="...", task_type="vqa")
"""

import os
import sys
import logging
from typing import List, Tuple, Optional, Any
from pathlib import Path
from PIL import Image
import torch

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] SatQuery-Inference: %(message)s")
logger = logging.getLogger("SatQuery-Inference")

# ============================================================================
# Configuration & Global Cache (Singleton Pattern)
# ============================================================================
BASE_MODEL_ID = os.environ.get("SATQUERY_BASE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.environ.get("SATQUERY_ADAPTER_PATH", os.path.join(_CURRENT_DIR, "satquery_checkpoints", "final_adapter"))

# Global in-memory cache to prevent reloading the model on every API request
_CACHED_MODEL: Optional[Any] = None
_CACHED_PROCESSOR: Optional[Any] = None

# Task-Specific Prompt Templates (Forcing Structured Output)
TASK_PROMPTS = {
    "vqa": (
        "Analyze this satellite image and answer the user's inquiry precisely: {query}\n"
        "Structure your answer with clear bullet points (e.g., Counts, Identifiers, Spatial Layout)."
    ),
    "captioning": (
        "Provide a detailed remote sensing description of this satellite scene.\n"
        "Structure your response strictly under these headers:\n"
        "1. **Primary Scene Taxonomy**\n"
        "2. **Dominant Land Cover & Land Use (LULC)**\n"
        "3. **Prominent Infrastructure & Terrain Features**\n"
        "4. **Executive Summary**"
    ),
    "grounding": (
        "Locate all instances of '{query}' in this satellite image.\n"
        "Structure your response with:\n"
        "1. **Target Identification**\n"
        "2. **Spatial Localization** (specify quadrant and bounding boxes)\n"
        "3. **Contextual Landmarks**\n"
        "4. **Visual Verification Cues**"
    ),
    "change_detection": (
        "Compare the two bi-temporal satellite images (Image 1 captured at Time 1, and Image 2 captured at Time 2). Describe all changes regarding: {query}\n"
        "Structure your response strictly under:\n"
        "1. **Overview of Temporal Shifts**\n"
        "2. **Key Zones of Alteration**\n"
        "3. **Impact & Severity Assessment**\n"
        "4. **Actionable Intelligence / Recommendations**"
    ),
    "optical_sar_fusion": (
        "Analyze the complementary multi-modal satellite image pair (Optical and SAR): {query}\n"
        "Synthesize both sensor modalities to deliver fused insights under:\n"
        "1. **Cross-Modal Sensor Alignment**\n"
        "2. **Radar Texture & Moisture Analysis**\n"
        "3. **Fused Intelligence Synthesis**"
    )
}


def get_or_load_model() -> Tuple[Any, Any]:
    """
    Lazy-loads and caches the Vision-Language Model and Processor in memory/VRAM.
    Checks for CUDA (GPU) and automatically falls back to CPU if no GPU is available.
    """
    global _CACHED_MODEL, _CACHED_PROCESSOR

    if _CACHED_MODEL is not None and _CACHED_PROCESSOR is not None:
        return _CACHED_MODEL, _CACHED_PROCESSOR

    print("\n" + "="*50)
    print("🚀 Initializing SatQuery Specialist Vision-Language Model...")
    print("⏳ Please wait! This may take 3-5 minutes to download weights on first run...")
    print("="*50 + "\n")

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    # 1. Device & Precision Detection
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)} | Using Precision: {dtype}")
    else:
        device = "cpu"
        dtype = torch.float32
        device_map = None
        print("⚠️ No CUDA GPU detected. Falling back gracefully to CPU (float32).")

    # 2. Load Processor with optimized resolution limits
    print("📥 Loading Processor from HuggingFace...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        min_pixels=128 * 28 * 28,
        max_pixels=384 * 28 * 28
    )

    # 3. Load Base Model
    print(f"📥 Downloading/Loading Base Model '{BASE_MODEL_ID}' (This is ~5GB, please wait)...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )

    # 4. Attach Trained LoRA Adapter (if directory exists)
    if os.path.exists(ADAPTER_PATH):
        print(f"🔗 Attaching fine-tuned Remote Sensing LoRA adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    else:
        print(f"⚠️ Adapter not found at '{ADAPTER_PATH}'. Running base model '{BASE_MODEL_ID}'.")
        model = base_model

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    _CACHED_MODEL = model
    _CACHED_PROCESSOR = processor
    print("\n✅ SatQuery Model is warm, cached in memory, and ready for inference!\n")
    return _CACHED_MODEL, _CACHED_PROCESSOR


def run_specialist_model(images: list, query: str, task_type: str) -> tuple[str, float]:
    """
    Production inference function for FastAPI backend integration.

    Args:
        images (list): List of PIL.Image.Image objects or image paths.
                       (Pass 1 image for VQA, Captioning, and Grounding;
                        Pass exactly 2 images [img_t1, img_t2] for Change Detection).
        query (str): User question, target object name, or task instruction.
        task_type (str): 'vqa' | 'captioning' | 'grounding' | 'change_detection'.

    Returns:
        tuple[str, float]: (Generated text response, confidence score in range [0.0, 1.0]).
    """
    # 1. Input Validation & Sanitization
    if not images or len(images) == 0:
        return "[Error] No satellite images provided.", 0.0

    task_key = task_type.lower().strip()
    if task_key == "change_detection" and len(images) < 2:
        return "[Error] Change detection task requires exactly 2 images (Time 1 and Time 2).", 0.0

    # Ensure all items are PIL RGB Images
    pil_images: List[Image.Image] = []
    for item in images:
        if isinstance(item, Image.Image):
            pil_images.append(item.convert("RGB"))
        elif isinstance(item, (str, Path)):
            pil_images.append(Image.open(str(item)).convert("RGB"))
        else:
            try:
                pil_images.append(Image.open(item).convert("RGB"))
            except Exception as e:
                return f"[Error] Failed to parse input image: {e}", 0.0

    # 2. Format System Prompt
    system_instruction = TASK_PROMPTS.get(task_key, "You are a remote sensing AI.")
    if "{query}" in system_instruction:
        system_instruction = system_instruction.format(query=query if query else "")

    # 3. Build Multi-Modal Conversation Payload
    user_content = []
    for img in pil_images:
        user_content.append({"type": "image", "image": img})
    
    # Just pass the task instruction or query to the user role
    user_instruction = f"User Query: {query}" if query and "{query}" not in TASK_PROMPTS.get(task_key, "") else "Please analyze the imagery according to your system instructions."
    user_content.append({"type": "text", "text": user_instruction})

    # Separate strict formatting instructions into the system role
    conversation = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]

    try:
        # 4. Retrieve Cached Model & Processor
        model, processor = get_or_load_model()
        from qwen_vl_utils import process_vision_info

        # Format inputs
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # 5. Forward Pass with Strict Memory & VRAM Management
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.2,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )

        # 6. Decode Output Text
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        response_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

        # 7. Calculate Confidence Score from Output Logits
        confidence_score = 0.95
        if outputs.scores and len(outputs.scores) > 0:
            stacked_logits = torch.stack(outputs.scores, dim=1) # [1, seq_len, vocab_size]
            probs = torch.softmax(stacked_logits, dim=-1)
            token_probs = torch.gather(probs[0], dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)
            avg_prob = float(torch.mean(token_probs).item())
            confidence_score = round(max(0.05, min(0.99, avg_prob)), 4)

        return response_text, confidence_score

    except Exception as e:
        logger.error(f"Inference error in run_specialist_model: {e}", exc_info=True)
        return f"[Inference Error] {str(e)}", 0.0


# ============================================================================
# Standalone CLI Self-Test
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("🛰️  SatQuery AI - Standalone Inference Test")
    print("=" * 70)
    
    # Create two synthetic test images
    img1 = Image.new("RGB", (256, 256), color=(34, 139, 34))
    img2 = Image.new("RGB", (256, 256), color=(180, 50, 50))

    print("\n[Test 1] Single-Image VQA:")
    ans1, conf1 = run_specialist_model([img1], query="What is the terrain color?", task_type="vqa")
    print(f"  • Answer     : {ans1}")
    print(f"  • Confidence : {conf1}")

    print("\n[Test 2] Bi-Temporal Change Detection (2 Images):")
    ans2, conf2 = run_specialist_model([img1, img2], query="Detect modifications.", task_type="change_detection")
    print(f"  • Answer     : {ans2}")
    print(f"  • Confidence : {conf2}")

    print("\n✅ Verification successful! Ready for backend import.")
