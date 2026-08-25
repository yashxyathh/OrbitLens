"""
SatQuery AI - Agentic Task Router
==================================
This module serves as the primary routing brain of SatQuery AI.
It analyzes incoming natural language queries and image inputs to determine 
which specialized remote sensing vision-language task pipeline should be invoked.

Routing Taxonomy:
-----------------
1. `single_image_vqa`: Single-image Visual Question Answering (RSVQA).
   - Used for specific analytical questions on a single satellite image (e.g. counting objects, identifying road widths, surface types).
2. `captioning`: Satellite Scene Description & Land-Use Summary (VRSBench).
   - Used for broad descriptive queries asking for overall landscape, land use/land cover (LULC), or scene summaries.
3. `grounding`: Spatial Localization & Visual Grounding.
   - Used when users ask where an object is located, requesting bounding boxes, quadrant positions, or pinpointing facilities.
4. `change_detection`: Bi-Temporal Change Analysis (CDVQA).
   - Used when two images representing different timestamps (T0 before vs T1 after) are provided to detect flooding, damage, or urban sprawl.
5. `optical_sar_fusion`: Multi-Modal Optical + SAR (Radar) Fusion.
   - Used when paired optical and radar/SAR imagery are analyzed together for moisture, roughness, penetration, or cloud-covered terrain.
"""

import re
from typing import Tuple
try:
    from schemas import TaskType
except ImportError:
    from backend.schemas import TaskType


# Keyword dictionaries mapped to specialized RS tasks
KEYWORDS_CAPTIONING = [
    "caption", "describe", "description", "overview", "summary", "summarize",
    "what is in this scene", "tell me about this", "scene overview",
    "land use", "land cover", "lulc", "what kind of area", "terrain type",
    "general view", "explain this image", "what does this satellite image show",
    "what do you see"
]

KEYWORDS_GROUNDING = [
    "where is", "where are", "locate", "location", "find the", "pinpoint",
    "coordinates", "bounding box", "bbox", "grounding", "spatial position",
    "which quadrant", "top-left", "top-right", "bottom-left", "bottom-right",
    "detect and locate", "highlight", "mark the", "spot the", "which part"
]

KEYWORDS_SAR_FUSION = [
    "sar", "radar", "backscatter", "microwave", "polarimetric", "sentinel-1",
    "sentinel 1", "speckle", "roughness", "dielectric", "soil moisture",
    "penetration", "optical+sar", "optical and sar", "fusion", "synthetic aperture",
    "polarization", "vv", "vh", "hh", "hv", "all-weather", "cloud-penetrating"
]

KEYWORDS_CHANGE_DETECTION = [
    "change", "changed", "changes", "difference", "differences", "before and after",
    "before vs after", "temporal", "time series", "flood", "flooded", "inundation",
    "damage", "destruction", "destroyed", "expansion", "deforestation", "growth",
    "disaster assessment", "post-disaster", "pre-disaster", "what changed",
    "compare", "comparison", "evolution", "loss", "increase", "disappeared", "new"
]


def route_query(query: str, image_count: int) -> Tuple[TaskType, str, float]:
    """
    Evaluates the user query text and image count to route to the appropriate RS specialist model.
    
    Parameters:
        query (str): The natural language question or instruction from the user.
        image_count (int): Number of images supplied (must be 1 or 2).
        
    Returns:
        Tuple[TaskType, str, float]:
            - TaskType: The resolved task pipeline.
            - str: A detailed rationale explaining the routing decision for auditability.
            - float: Routing confidence heuristic score (0.0 to 1.0).
    """
    clean_query = query.lower().strip()
    
    # -------------------------------------------------------------
    # 1. DUAL-IMAGE ROUTING LOGIC (image_count == 2)
    # -------------------------------------------------------------
    if image_count == 2:
        # Check if the query specifically references SAR / Radar fusion
        has_sar_keywords = any(kw in clean_query for kw in KEYWORDS_SAR_FUSION)
        if has_sar_keywords:
            return (
                "optical_sar_fusion",
                "Dual-image input detected with multi-modal radar/SAR signatures. "
                "Routing to Multi-Sensor Optical & SAR Fusion Specialist.",
                0.96
            )
        
        # Check if query references temporal change, disaster, or comparison
        has_change_keywords = any(kw in clean_query for kw in KEYWORDS_CHANGE_DETECTION)
        if has_change_keywords:
            return (
                "change_detection",
                "Dual-image input detected with bi-temporal change keywords (e.g. before/after, flood, damage). "
                "Routing to Bi-Temporal Change Detection Specialist (CDVQA).",
                0.97
            )
            
        # Fallback default for 2 images: Bi-temporal change detection
        return (
            "change_detection",
            "Dual-image input supplied. Defaulting to Bi-Temporal Change Detection Specialist.",
            0.91
        )
        
    # -------------------------------------------------------------
    # 2. SINGLE-IMAGE ROUTING LOGIC (image_count == 1)
    # -------------------------------------------------------------
    elif image_count == 1:
        # Priority 1: Visual Grounding & Spatial Localization
        for kw in KEYWORDS_GROUNDING:
            if re.search(r'\b' + re.escape(kw) + r'\b', clean_query) or kw in clean_query:
                return (
                    "grounding",
                    f"Single image with spatial localization query matching pattern '{kw}'. "
                    "Routing to Remote Sensing Visual Grounding Specialist.",
                    0.94
                )
                
        # Priority 2: Scene Description & Landscape Captioning
        for kw in KEYWORDS_CAPTIONING:
            if re.search(r'\b' + re.escape(kw) + r'\b', clean_query) or kw in clean_query:
                return (
                    "captioning",
                    f"Single image with broad scene descriptive query matching pattern '{kw}'. "
                    "Routing to Satellite Scene Captioning Specialist (VRSBench).",
                    0.95
                )
                
        # Priority 3: Default Single Image VQA (Object counting, specific questions)
        return (
            "single_image_vqa",
            "Single image with specific feature inquiry. "
            "Routing to Remote Sensing Visual Question Answering (RSVQA) Specialist.",
            0.92
        )

    # -------------------------------------------------------------
    # 3. INVALID IMAGE COUNT
    # -------------------------------------------------------------
    else:
        raise ValueError(
            f"Unsupported image count: {image_count}. SatQuery AI accepts 1 or 2 images."
        )
