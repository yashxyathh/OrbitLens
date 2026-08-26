#!/usr/bin/env python3
"""
SatQuery AI - Remote Sensing Dataset Ingestion & Preprocessing Pipeline
Optimized for Google Colab, Kaggle, and Cloud GPU instances.

Usage:
    # Quick smoke-test / dry-run with synthetic data:
    python data_pipeline.py --synthetic --samples-per-task 25

    # Full data acquisition from remote sources:
    python data_pipeline.py --download-all --max-samples 2000
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List

# Ensure UTF-8 output across Windows PowerShell and Unix terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import DATA_CONFIG, MODEL_CONFIG, RUNTIME_ENV
from src.data.schemas import SatQuerySample, TaskType
from src.data.downloaders import (
    download_bigearthnet,
    download_rsvqa,
    download_vrsbench,
    download_cdvqa,
    generate_synthetic_satellite_dataset
)
from src.data.formatters import SatQueryDatasetFormatter

def print_banner():
    print("=" * 70)
    print("[SatQuery AI] Remote Sensing Dataset Ingestion & Preprocessing")
    print(f"Runtime Environment : {RUNTIME_ENV.upper()}")
    print(f"Root Data Directory  : {DATA_CONFIG.root_dir.absolute()}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="SatQuery AI Dataset Acquisition & Formatting")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic dataset for fast dry-run testing.")
    parser.add_argument("--samples-per-task", type=int, default=50, help="Number of synthetic samples per task.")
    parser.add_argument("--download-all", action="store_true", help="Attempt download across BigEarthNet, RSVQA, VRSBench, CDVQA.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples per downloaded dataset.")
    parser.add_argument("--output-dir", type=str, default=str(DATA_CONFIG.processed_dir), help="Output directory for HuggingFace dataset.")
    
    args = parser.parse_args()
    print_banner()
    
    all_samples: List[SatQuerySample] = []
    
    if args.synthetic or (not args.download_all and not args.synthetic):
        print(f"\n[1/3] Generating synthetic satellite dataset ({args.samples_per_task} samples/task)...")
        synthetic_samples = generate_synthetic_satellite_dataset(
            num_samples_per_task=args.samples_per_task,
            image_size=(512, 512)
        )
        all_samples.extend(synthetic_samples)
        
    if args.download_all:
        print(f"\n[1/3] Downloading & processing remote sensing datasets (max {args.max_samples}/dataset)...")
        
        # 1. BigEarthNet (Land-cover / domain adaptation)
        try:
            ben_samples = download_bigearthnet(max_samples=args.max_samples)
            all_samples.extend(ben_samples)
        except Exception as e:
            print(f"[!] BigEarthNet fetch skipped: {e}")
            
        # 2. RSVQA (Single-image VQA)
        try:
            rsvqa_samples = download_rsvqa(max_samples=args.max_samples)
            all_samples.extend(rsvqa_samples)
        except Exception as e:
            print(f"[!] RSVQA fetch skipped: {e}")
            
        # 3. VRSBench (Dense captioning & visual grounding)
        try:
            vrs_samples = download_vrsbench(max_samples=args.max_samples)
            all_samples.extend(vrs_samples)
        except Exception as e:
            print(f"[!] VRSBench fetch skipped: {e}")
            
        # 4. CDVQA (Bi-temporal change detection)
        try:
            cd_samples = download_cdvqa(max_samples=args.max_samples)
            all_samples.extend(cd_samples)
        except Exception as e:
            print(f"[!] CDVQA fetch skipped: {e}")

    if not all_samples:
        print("[X] Error: No samples were loaded or generated.")
        sys.exit(1)

    # Compute Task Distribution Statistics
    task_counts = {}
    for s in all_samples:
        t = s.task_type.value
        task_counts[t] = task_counts.get(t, 0) + 1

    print("\n[2/3] Dataset Summary & Task Distribution:")
    print(f"  * Total Aggregated Samples : {len(all_samples)}")
    for task_name, count in task_counts.items():
        print(f"    - {task_name.ljust(18)} : {count} samples ({count/len(all_samples)*100:.1f}%)")

    # Format and Save to HuggingFace DatasetDict
    print(f"\n[3/3] Formatting into HuggingFace DatasetDict (80% Train / 10% Val / 10% Test)...")
    formatter = SatQueryDatasetFormatter()
    dataset_dict = formatter.build_hf_dataset_dict(all_samples)
    saved_path = formatter.save(dataset_dict, output_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print(f"[+] Success! Dataset prepared and saved to: {saved_path}")
    if hasattr(dataset_dict, "__getitem__"):
        print(f"  Train samples : {len(dataset_dict['train']['id'] if isinstance(dataset_dict['train'], dict) else dataset_dict['train'])}")
        print(f"  Val samples   : {len(dataset_dict['validation']['id'] if isinstance(dataset_dict['validation'], dict) else dataset_dict['validation'])}")
        print(f"  Test samples  : {len(dataset_dict['test']['id'] if isinstance(dataset_dict['test'], dict) else dataset_dict['test'])}")
    print("=" * 70)

if __name__ == "__main__":
    main()
