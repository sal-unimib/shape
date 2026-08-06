
from typing import Any, Dict

class RetrievalConfiguration:
    collection : str
    embedding : str
    topk : int

    def __init__(self, collection: str, embedding: str, topk: int):
        self.collection = collection
        self.embedding = embedding
        self.topk = topk

class ReasoningConfiguration:
    hyperparameters : Dict[str, Any]
    prompt : str

    def __init__(self, hyperparameters: Dict[str, Any], prompt: str):
        self.hyperparameters = hyperparameters
        self.prompt = prompt

class StageConfiguration:
    """
    Configuration for a stage of the pipeline.
    """
    def __init__(self, retrieval: Dict[str, RetrievalConfiguration], reasoning: Dict[str, ReasoningConfiguration]):
        self.retrieval = retrieval
        self.reasoning = reasoning