from dotenv import load_dotenv
# import httpx
import logging
import os
import ollama
from src.utils.sapbert import sapbert_embed
from typing import List, Sequence

logger = logging.getLogger('shape.sources.llm')
logger.setLevel(logging.DEBUG)


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SAPBERT_URL = os.getenv("SAPBERT_URL", "http://localhost:8000/embed")

ollama_source = ollama.Client(
    host=OLLAMA_HOST
)

llm_models = {
    "mistral": "mistral",
    "mixtral_8x7": "mixtral:8x7b",
    "mixtral_8x22": "mixtral:8x22b",
    "llama": "llama3.1:70b",
    "phi4": "phi4"
}

def embed(model: str, labels: list[str]) -> List[Sequence[float]]:
    """
    Generate embeddings for the given labels using the selected model.
    """
    if model == "sapbert":
        # payload = {
        #     "labels": labels
        # }
        # with httpx.Client() as client:
        #     response = client.post(
        #         SAPBERT_URL,
        #         json=payload
        #     )
        #     return response.json()["embeddings"]
        return sapbert_embed(labels)
    elif model == "nomic-embed-text":
        ollama_response = ollama_source.embed(
            model=model,
            input=labels
        )
        return list(ollama_response.embeddings)
    else:
        logger.warning("Model %s not supported", model)
        return []