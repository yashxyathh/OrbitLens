import os
from celery import Celery
from dotenv import load_dotenv
import time

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "satquery_worker",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# We will import the actual inference function inside the task to avoid circular imports
@celery_app.task(bind=True)
def process_imagery_task(self, query: str, image_urls: list, task_type: str):
    """
    Celery task for processing satellite imagery.
    """
    try:
        from llm_client import analyze_imagery_production
        
        # In a real scenario, analyze_imagery_production would download the image_urls
        # or pass them directly to the multimodal model.
        answer_text, model_confidence, model_invoked = analyze_imagery_production(image_urls, query, task_type)
        
        return {
            "status": "completed",
            "answer": answer_text,
            "confidence": model_confidence,
            "model_invoked": model_invoked
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
