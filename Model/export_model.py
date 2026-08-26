#!/usr/bin/env python3
"""
SatQuery AI - Model Export & Weight Serialization (Step 4)
Exports standalone LoRA adapters or merges LoRA weights into base model (.safetensors).

Usage:
    # 1. Export standalone LoRA adapter:
    python export_model.py --adapter-dir ./satquery_checkpoints/final_adapter --output-dir ./exported_models/satquery_adapter

    # 2. Merge LoRA weights into base model for standalone production deployment:
    python export_model.py --adapter-dir ./satquery_checkpoints/final_adapter --merge --output-dir ./exported_models/satquery_merged_fp16
"""

import argparse
import sys
import os
from pathlib import Path
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import MODEL_CONFIG

def print_banner():
    print("=" * 70)
    print("[SatQuery AI] Model Export & Weight Serialization (Step 4)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="SatQuery Model Export & Serialization")
    parser.add_argument("--adapter-dir", type=str, required=True, help="Path to trained LoRA adapter directory.")
    parser.add_argument("--output-dir", type=str, default="./exported_models/satquery_vlm", help="Export target directory.")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights with base model into standalone FP16 safetensors.")
    parser.add_argument("--base-model-id", type=str, default=MODEL_CONFIG.base_model_id, help="Base model identifier.")
    parser.add_argument("--push-to-hub", type=str, default=None, help="Optional Hugging Face repository ID to upload weights.")

    args = parser.parse_args()
    print_banner()

    adapter_path = Path(args.adapter_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not adapter_path.exists():
        print(f"❌ Adapter directory not found at: {adapter_path}")
        sys.exit(1)

    print(f"📦 Source Adapter : {adapter_path.absolute()}")
    print(f"📁 Target Output  : {output_path.absolute()}")
    print(f"🔀 Merge Mode     : {args.merge}")

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    # 1. Load Processor
    print("\n[1/3] Loading Processor & Tokenizer...")
    processor = AutoProcessor.from_pretrained(args.base_model_id)

    if args.merge:
        print("\n[2/3] Merging LoRA weights into base model weights (FP16)...")
        # Load base model in FP16 for clean weight merging
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        # Attach LoRA adapter
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        
        # Merge weights into base layers
        print("  • Executing merge_and_unload()...")
        merged_model = model.merge_and_unload()

        print(f"\n[3/3] Saving merged standalone model to {output_path} (.safetensors format)...")
        merged_model.save_pretrained(str(output_path), safe_serialization=True)
        processor.save_pretrained(str(output_path))
    else:
        print(f"\n[2/3] Copying & saving adapter artifacts to {output_path}...")
        model = PeftModel.from_pretrained(
            Qwen2VLForConditionalGeneration.from_pretrained(
                args.base_model_id,
                torch_dtype=torch.float16,
                device_map="cpu"
            ),
            str(adapter_path)
        )
        model.save_pretrained(str(output_path), safe_serialization=True)
        processor.save_pretrained(str(output_path))

    if args.push_to_hub:
        print(f"\n🚀 Uploading export to Hugging Face Hub: {args.push_to_hub}...")
        merged_model.push_to_hub(args.push_to_hub)
        processor.push_to_hub(args.push_to_hub)

    print("\n" + "=" * 70)
    print(f"✅ Model export successfully completed!")
    print(f"Artifacts saved at: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
