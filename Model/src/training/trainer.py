import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch

from src.config import MODEL_CONFIG, DEVICE, DTYPE
from src.data.formatters import Qwen2VLDataCollator
from src.training.lora_config import get_lora_config, get_quantization_config, print_trainable_parameters

logger = logging.getLogger("SatQuery-Trainer")

try:
    from transformers import (
        Trainer,
        TrainingArguments,
        AutoProcessor,
        Qwen2VLForConditionalGeneration
    )
    from peft import get_peft_model, prepare_model_for_kbit_training
    HAS_TRAINER_DEPS = True
except ImportError:
    Trainer = object
    TrainingArguments = None
    AutoProcessor = None
    Qwen2VLForConditionalGeneration = None
    HAS_TRAINER_DEPS = False


class SatQueryTrainer:
    """
    High-level training orchestrator for SatQuery AI Vision-Language Models.
    Coordinates QLoRA setup, Hugging Face Trainer configuration, and checkpointing.
    """

    def __init__(
        self,
        base_model_id: Optional[str] = None,
        use_qlora: bool = True,
        output_dir: str = "./satquery_checkpoints"
    ):
        if not HAS_TRAINER_DEPS:
            raise RuntimeError("Transformers and PEFT must be installed to run SatQueryTrainer.")
            
        self.base_model_id = base_model_id or MODEL_CONFIG.base_model_id
        self.use_qlora = use_qlora
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.processor = None

    def initialize_model_and_processor(self):
        """
        Loads base Qwen2-VL model and applies LoRA / QLoRA adapters.
        """
        logger.info(f"Loading processor for {self.base_model_id}...")
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_id,
            min_pixels=MODEL_CONFIG.min_pixels,
            max_pixels=MODEL_CONFIG.max_pixels
        )

        quant_config = get_quantization_config(use_4bit=True) if self.use_qlora else None
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        logger.info(f"Loading base model {self.base_model_id} (QLoRA={self.use_qlora})...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.base_model_id,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )

        if self.use_qlora:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True
            )

        # Enable gradient checkpointing to save VRAM
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        # Apply LoRA configuration
        lora_config = get_lora_config(
            r=MODEL_CONFIG.lora_r,
            lora_alpha=MODEL_CONFIG.lora_alpha,
            lora_dropout=MODEL_CONFIG.lora_dropout
        )
        self.model = get_peft_model(model, lora_config)
        print_trainable_parameters(self.model)
        
        return self.model, self.processor

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        epochs: int = 3,
        batch_size: int = 2,
        grad_accum_steps: int = 4,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_steps: int = 100,
        warmup_ratio: float = 0.03
    ):
        """
        Executes fine-tuning using the HuggingFace Trainer.
        """
        if self.model is None or self.processor is None:
            self.initialize_model_and_processor()

        collator = Qwen2VLDataCollator(processor=self.processor)

        is_cuda = torch.cuda.is_available()
        use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
        use_fp16 = is_cuda and not use_bf16

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=save_steps if eval_dataset is not None else None,
            fp16=use_fp16,
            bf16=use_bf16,
            optim="paged_adamw_8bit" if self.use_qlora else "adamw_torch",
            remove_unused_columns=False,
            report_to="none",
            dataloader_pin_memory=False,
            gradient_checkpointing=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator
        )

        logger.info("Starting Vision-Language LoRA training run...")
        train_result = trainer.train()

        # Save the fine-tuned adapter weights
        final_adapter_dir = self.output_dir / "final_adapter"
        self.model.save_pretrained(str(final_adapter_dir))
        self.processor.save_pretrained(str(final_adapter_dir))
        logger.info(f"LoRA specialist adapter weights saved to {final_adapter_dir}")

        return train_result
