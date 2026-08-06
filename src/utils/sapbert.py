import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Sequence

SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
sapbert_model = AutoModel.from_pretrained(SAPBERT_MODEL)
sapbert_model.to(DEVICE)
sapbert_model.eval()

def sapbert_embed(labels: list[str]) -> List[Sequence[float]]:
    """
    Generate embeddings for the given labels using the SapBERT model.
    """
    tokenized_inputs = tokenizer(
        labels,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    tokenized_inputs = {
        name: tensor.to(DEVICE)
        for name, tensor in tokenized_inputs.items()
    }

    with torch.no_grad():
        model_outputs = sapbert_model(**tokenized_inputs)

    sentence_embeddings = model_outputs.last_hidden_state.mean(dim=1)

    normalized_embeddings = torch.nn.functional.normalize(
        sentence_embeddings,
        p=2,
        dim=1
    )

    return normalized_embeddings.cpu().tolist()

