import os
import json
import logging
import random
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from datasets import Dataset, DatasetDict
    HAS_HF_DATASETS = True
except ImportError:
    Dataset = None
    DatasetDict = None
    HAS_HF_DATASETS = False

from src.data.schemas import SatQuerySample, TaskType
from src.config import DATA_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SatQuery-Formatter")

class SatQueryDatasetFormatter:
    """
    Formats remote sensing samples into standard Hugging Face Datasets
    specifically optimized for Qwen2-VL and Florence-2 multi-task training.
    """

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def build_hf_dataset_dict(
        self,
        samples: List[SatQuerySample],
        shuffle: bool = True
    ) -> Any:
        """
        Converts a list of SatQuerySample objects into a split DatasetDict (train, validation, test).
        """
        if not samples:
            raise ValueError("Cannot format an empty sample list!")

        if shuffle:
            random.seed(self.seed)
            samples = samples.copy()
            random.shuffle(samples)

        total = len(samples)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]

        logger.info(
            f"Dataset Split -> Total: {total} | Train: {len(train_samples)} | "
            f"Val: {len(val_samples)} | Test: {len(test_samples)}"
        )

        def _to_raw_dict(sample_list: List[SatQuerySample]) -> Dict[str, List[Any]]:
            data: Dict[str, List[Any]] = {
                "id": [],
                "task_type": [],
                "query": [],
                "response": [],
                "images": [],
                "conversations": [],
                "num_images": []
            }
            for s in sample_list:
                data["id"].append(s.id)
                data["task_type"].append(s.task_type.value)
                data["query"].append(s.query)
                data["response"].append(s.response)
                data["images"].append(s.images)
                data["conversations"].append(s.to_qwen_vl_conversation())
                data["num_images"].append(len(s.images))
            return data

        raw_train = _to_raw_dict(train_samples)
        raw_val = _to_raw_dict(val_samples)
        raw_test = _to_raw_dict(test_samples)

        if HAS_HF_DATASETS:
            train_ds = Dataset.from_dict(raw_train)
            val_ds = Dataset.from_dict(raw_val)
            test_ds = Dataset.from_dict(raw_test)

            dataset_dict = DatasetDict({
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds
            })
            return dataset_dict
        else:
            return {
                "train": raw_train,
                "validation": raw_val,
                "test": raw_test
            }

    def save(self, dataset_dict: Any, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Save formatted DatasetDict to disk."""
        out_path = Path(output_dir) if output_dir else DATA_CONFIG.processed_dir
        out_path.mkdir(parents=True, exist_ok=True)
        
        if HAS_HF_DATASETS and hasattr(dataset_dict, "save_to_disk"):
            dataset_dict.save_to_disk(str(out_path))
        else:
            # Fallback JSON / manifest serialization
            manifest_path = out_path / "dataset_manifest.json"
            meta = {
                split: {
                    "count": len(data["id"]),
                    "sample_ids": data["id"][:10]
                }
                for split, data in dataset_dict.items()
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
                
        logger.info(f"Saved formatted dataset to {out_path}")
        return out_path

    @staticmethod
    def load(input_dir: Optional[Union[str, Path]] = None) -> Any:
        """Load formatted DatasetDict from disk."""
        in_path = Path(input_dir) if input_dir else DATA_CONFIG.processed_dir
        logger.info(f"Loading dataset from {in_path}...")
        if HAS_HF_DATASETS:
            return DatasetDict.load_from_disk(str(in_path))
        else:
            with open(in_path / "dataset_manifest.json", "r", encoding="utf-8") as f:
                return json.load(f)


class Qwen2VLDataCollator:
    """
    PyTorch Data Collator for Qwen2-VL training.
    Encodes single and multi-image satellite scenes and masks user prompt tokens
    with -100 so cross-entropy loss is computed exclusively on the assistant response.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        conversations = [item["conversations"] for item in batch]
        
        # 1. Format text using the tokenizer chat template
        texts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in conversations
        ]
        
        # 2. Extract image inputs
        image_inputs, video_inputs = process_vision_info(conversations)
        
        # 3. Process with Qwen2-VL processor
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        # Mask out image token positions from loss calculation
        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_token_id is not None:
            labels[labels == image_token_id] = -100
            
        inputs["labels"] = labels
        return inputs
