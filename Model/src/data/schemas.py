from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import json

class TaskType(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"

@dataclass
class BoundingBox:
    """Normalized bounding box coordinates in [0, 1000] integer scale or [0.0, 1.0] float scale."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    label: str

    def to_qwen_format(self) -> Dict[str, Any]:
        """Convert to Qwen2-VL standard grounding representation."""
        return {
            "box_2d": [int(self.ymin * 1000), int(self.xmin * 1000), int(self.ymax * 1000), int(self.xmax * 1000)],
            "label": self.label
        }

@dataclass
class SatQuerySample:
    """Unified Remote Sensing Vision-Language Sample representation."""
    id: str
    images: List[Image.Image]  # 1 image for VQA/Caption/Grounding; 2 images for Change Detection
    query: str
    task_type: TaskType
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.images:
            raise ValueError("SatQuerySample must contain at least 1 image.")
        if self.task_type == TaskType.CHANGE_DETECTION and len(self.images) < 2:
            raise ValueError("Change Detection samples must contain exactly 2 bi-temporal images.")

    def to_qwen_vl_conversation(self) -> List[Dict[str, Any]]:
        """
        Format sample into Qwen2-VL chat format with support for single and multi-image inputs.
        Qwen2-VL natively processes multiple images sequentially in the conversation history.
        """
        user_content: List[Dict[str, Any]] = []
        
        # Add all images
        for img in self.images:
            user_content.append({"type": "image", "image": img})
            
        # Add task instruction prompt
        user_content.append({"type": "text", "text": self.query})
        
        conversation = [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": self.response}
                ]
            }
        ]
        return conversation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HuggingFace Dataset storage."""
        return {
            "id": self.id,
            "images": self.images,
            "query": self.query,
            "task_type": self.task_type.value,
            "response": self.response,
            "metadata": json.dumps(self.metadata)
        }
