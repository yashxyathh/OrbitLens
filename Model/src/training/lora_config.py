import logging
from typing import Optional, List, Dict, Any
from src.config import MODEL_CONFIG

logger = logging.getLogger("SatQuery-LoRA")

def get_quantization_config(use_4bit: bool = True):
    """
    Configure BitsAndBytes 4-bit / 8-bit quantization for QLoRA.
    Drastically reduces VRAM footprint to < 4.5 GB on free Colab/Kaggle T4 GPUs.
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig
        
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        if use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        logger.warning("BitsAndBytes or Transformers not installed. Quantization config unavailable.")
        return None

def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
):
    """
    Generate PEFT LoRA configuration tailored for Qwen2-VL Vision-Language Model.
    Targets attention projection and MLP feed-forward layers.
    """
    try:
        from peft import LoraConfig, TaskType
        
        modules = target_modules or MODEL_CONFIG.target_modules
        
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=None # Vision tower frozen by default for maximum efficiency
        )
        return peft_config
    except ImportError:
        logger.warning("PEFT library not found.")
        return None

def print_trainable_parameters(model):
    """
    Utility function to display trainable vs total parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    pct = 100 * trainable_params / all_param if all_param > 0 else 0.0
    print(f"📊 Model Parameter Summary:")
    print(f"   • Trainable Params : {trainable_params:,}")
    print(f"   • All Params       : {all_param:,}")
    print(f"   • Trainable %      : {pct:.3f}% (LoRA efficiency)")
