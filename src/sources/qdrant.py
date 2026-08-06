from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

qdrant_source : QdrantClient = QdrantClient(
    url=f"http://{QDRANT_HOST}",
    api_key=QDRANT_API_KEY
)

def create_collection_if_not_exists(collection_name: str, vector_size: int):
    if not qdrant_source.collection_exists(collection_name):
        qdrant_source.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )