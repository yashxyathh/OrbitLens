#!/usr/bin/env python3
"""
SatQuery AI - Vision-Language Specialist Model Training Script (Step 3)
Fine-tunes Qwen2-VL-2B on multi-task remote sensing data using LoRA / QLoRA.

Usage:
    # Run QLoRA training on prepared dataset in Colab/Kaggle or GPU instance:
    python train.py --dataset-dir ./satquery_data/processed --output-dir ./satquery_checkpoints --epochs 3
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import DATA_CONFIG, MODEL_CONFIG, RUNTIME_ENV
from src.data.formatters import SatQueryDatasetFormatter

def print_banner():
    print("=" * 70)
    print("[SatQuery AI] Vision-Language Model LoRA Fine-Tuning (Step 3)")
    print(f"Base Model : {MODEL_CONFIG.base_model_id}")
    print(f"Runtime    : {RUNTIME_ENV.upper()}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="SatQuery AI LoRA Fine-Tuning")
    parser.add_argument("--dataset-dir", type=str, default=str(DATA_CONFIG.processed_dir), help="Path to formatted HuggingFace dataset.")
    parser.add_argument("--output-dir", type=str, default="./satquery_checkpoints", help="Output directory for checkpoints and adapters.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--no-qlora", action="store_true", help="Disable 4-bit QLoRA and use full 16-bit LoRA.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Truncate train dataset for rapid debugging.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Truncate eval dataset for rapid debugging.")

    args = parser.parse_args()
    print_banner()

    # 1. Load Preprocessed Dataset
    print(f"\n[1/3] Loading formatted dataset from: {args.dataset_dir}...")
    try:
        dataset_dict = SatQueryDatasetFormatter.load(args.dataset_dir)
        train_ds = dataset_dict["train"]
        val_ds = dataset_dict["validation"] if "validation" in dataset_dict else None
        
        if args.max_train_samples and hasattr(train_ds, "select"):
            train_ds = train_ds.select(range(min(len(train_ds), args.max_train_samples)))
        if val_ds and args.max_eval_samples and hasattr(val_ds, "select"):
            val_ds = val_ds.select(range(min(len(val_ds), args.max_eval_samples)))
            
        print(f"  • Training samples   : {len(train_ds)}")
        print(f"  • Validation samples : {len(val_ds) if val_ds else 0}")
    except Exception as e:
        print(f"❌ Failed to load dataset from {args.dataset_dir}: {e}")
        print("💡 Hint: Run 'python data_pipeline.py --synthetic' or prepare your dataset first.")
        sys.exit(1)

    # 2. Initialize Trainer & LoRA Adapters
    print(f"\n[2/3] Initializing SatQueryTrainer (QLoRA={not args.no_qlora})...")
    from src.training.trainer import SatQueryTrainer

    trainer = SatQueryTrainer(
        base_model_id=MODEL_CONFIG.base_model_id,
        use_qlora=not args.no_qlora,
        output_dir=args.output_dir
    )

    # 3. Execute Training Loop
    print("\n[3/3] Starting Training Loop...")
    train_results = trainer.train(
        train_dataset=train_ds,
        eval_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr
    )

    print("\n" + "=" * 70)
    print(f"✅ Training Complete! LoRA specialist adapter saved to: {Path(args.output_dir) / 'final_adapter'}")
    print("=" * 70)

if __name__ == "__main__":
    main()
