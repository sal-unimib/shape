from datetime import datetime
from dotenv import load_dotenv
import json
import logging
import numpy as np
import os
import pandas as pd
from qdrant_client.models import PointStruct
import re
from src.sources.llm import embed
from src.sources.qdrant import qdrant_source
import sys
from time import time
from typing import List, Sequence
from tqdm import tqdm
import traceback

# ================================
# CONFIG
# ================================

load_dotenv()
SNOMED_BASE_PATH = os.getenv("SNOMED_BASE_PATH", "dataset/SnomedCT_International_RF2_20260201T120000Z/Snapshot")

FSN = "900000000000003001"
SYNONYM = "900000000000013009"
ACCEPTABLE = "900000000000549004"
PREFERRED = "900000000000548007"

DESCRIPTION_TYPE_MAP = {
    FSN: "Fully specified name (FSN)",
    SYNONYM: "Synonym",
    "900000000000550004": "Text definition"
}

DESCRIPTION_CASE_SIGNIFICANCE_MAP = {
    "900000000000448009": "Entire term case sensitive",
    "900000000000017005": "Case insensitive",
    "900000000000020002": "Initial character case insensitive"
}

LANGUAGE_ACCEPTABILITY_MAP = {
    ACCEPTABLE: "Acceptable",
    PREFERRED: "Preferred"
}

# ================================
# LOGGING
# ================================

logger = logging.getLogger("shape.routines.create_snomed_collection")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

# Logging - Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Logging - File handler
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(log_dir, f"snomed_pipeline_{timestamp}.log")

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.propagate = False

def create_qdrant_collections():
    """
    Create the Qdrant collections for SNOMED CT.
    """
    from qdrant_client.models import VectorParams, Distance
    if not qdrant_source.collection_exists("snomed_ct_concepts"):
        qdrant_source.create_collection(
            collection_name="snomed_ct_concepts",
            vectors_config={
                "concept_nomic_vector": VectorParams(size=768, distance=Distance.COSINE),
                "concept_sapbert_vector": VectorParams(size=768, distance=Distance.COSINE),
                "terms_avg_nomic_vector": VectorParams(size=768, distance=Distance.COSINE),
                "terms_avg_sapbert_vector": VectorParams(size=768, distance=Distance.COSINE),
            }
        )
    
    if not qdrant_source.collection_exists("snomed_ct_terms"):
        qdrant_source.create_collection(
            collection_name="snomed_ct_terms",
            vectors_config={
                "nomic_vector": VectorParams(size=768, distance=Distance.COSINE),
                "sapbert_vector": VectorParams(size=768, distance=Distance.COSINE),
            }
        )

def avg_embedding(vectors: List[Sequence[float]]) -> Sequence[float]:
    """
    Compute the average of a list of vectors and normalize it.
    
    Args:
        vectors: A list of vectors.
        
    Returns:
        The average vector.
    """
    # Ensure vectors is a numpy array
    vec_array = np.array(vectors, dtype=np.float32)
    
    # Compute the mean along the first axis (axis=0)
    mean_vec = np.mean(vec_array, axis=0)
    
    # Normalize the mean vector (L2 normalization)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        normalized_vec = mean_vec / norm
    else:
        normalized_vec = mean_vec
    
    return normalized_vec.tolist()

def create_snomed_ct_nodes():
    """
    Load SNOMED CT files, filter active ones, join them, and create SNOMED CT nodes.
    """
    # 1. Load SNOMED CT files
    try:
        concepts_df = pd.read_csv(f"{SNOMED_BASE_PATH}/Terminology/sct2_Concept_Snapshot_INT_20260201.txt", sep="\t", dtype=str)
        descriptions_df = pd.read_csv(f"{SNOMED_BASE_PATH}/Terminology/sct2_Description_Snapshot-en_INT_20260201.txt", sep="\t", dtype=str)
        language_df = pd.read_csv(f"{SNOMED_BASE_PATH}/Refset/Language/der2_cRefset_LanguageSnapshot-en_INT_20260201.txt", sep="\t", dtype=str)
    except Exception as e:
        logging.error(f"Errore caricamento file: {e}")
        raise

    # 2. Filter SNOMED CT rows to keep only active ones
    active_concepts_df = concepts_df[concepts_df["active"] == "1"]
    active_descriptions_df = descriptions_df[descriptions_df["active"] == "1"]
    active_language_df = language_df[language_df["active"] == "1"]

    # 3. Join
    description_language_df = active_descriptions_df.merge(
        active_language_df,
        left_on="id",
        right_on="referencedComponentId",
        how="left"
    )

    grouped_concepts = description_language_df.groupby("conceptId")
    total_concepts = len(grouped_concepts)
    snomed_nodes = []

    for i, (concept_id, description_group) in enumerate(tqdm(grouped_concepts, total=total_concepts, desc="Creating SNOMED CT Nodes")):
        try:
            # FSN
            fsn_rows = description_group[description_group["typeId"] == FSN]
            if fsn_rows.empty:
                continue

            fsn_term = fsn_rows.iloc[0]["term"]

            match = re.match(r"(.+?)\s\((.+)\)", fsn_term)
            if match:
                label = match.group(1)
                semantic_tag = match.group(2)
            else:
                label = fsn_term
                semantic_tag = None

            # DESCRIPTIONS
            description_list = []
            preferred_terms = []
            acceptable_terms = []

            for _, row in description_group.iterrows():
                term = row["term"]
                type_id = row["typeId"]
                acc_id = row["acceptabilityId"]
                case_id = row["caseSignificanceId"]

                desc_type = DESCRIPTION_TYPE_MAP.get(type_id, type_id)
                acceptability = LANGUAGE_ACCEPTABILITY_MAP.get(acc_id)
                case = DESCRIPTION_CASE_SIGNIFICANCE_MAP.get(case_id)

                description_list.append({
                    "term": term,
                    "type": desc_type,
                    "acceptability": acceptability,
                    "caseSignificance": case
                })

                if acceptability == "Acceptable":
                    acceptable_terms.append(term)
                elif acceptability == "Preferred":
                    preferred_terms.append(term)

            snomed_nodes.append({
                "concept_id": concept_id,
                "label": label,
                "semantic_tag": semantic_tag,
                "fsn": fsn_term,
                "preferred_terms": list(set(preferred_terms)),
                "acceptable_terms": list(set(acceptable_terms)),
                # "descriptions": description_list
            })
        except Exception as e:
            logger.error(f"Errore processing concept {concept_id}: {e}\n{traceback.format_exc()}")

    return snomed_nodes

def build_embedding_text(node: dict) -> str:
    """
    Build the embedding text for a given concept node.
    
    Args:
        node: The node to build the embedding text for.
        
    Returns:
        The embedding text.
    """
    parts = []

    if node.get("label"):
        parts.append(node["label"])

    # if node.get("semantic_tag"):
    #     parts.append(f"({node['semantic_tag']})")

    fsn = node.get("fsn", "")
    if fsn and fsn != node.get("label"):
        parts.append(fsn)

    preferred = list(set(node.get("preferred_terms", [])))
    already = {node.get("label", "").lower(), fsn.lower()}
    preferred = [t for t in preferred if t.lower() not in already][:5]
    if preferred:
        parts.extend(preferred)

    acceptable = list(set(node.get("acceptable_terms", [])))
    already.update(t.lower() for t in preferred)
    acceptable = [t for t in acceptable if t.lower() not in already][:3]
    if acceptable:
        parts.extend(acceptable)
    
    return ". ".join(parts)

def save_concept_embedding(concept_id: str, embedding: Sequence[float], model: str, mode: str):
    """
    Save the embedding for a given concept.
    
    Args:
        concept_id: The ID of the concept.
        embedding: The embedding of the concept.
        model: The model used to generate the embedding.
        mode: The mode used to generate the embedding.
    """
    path = f"dataset/snomed_concepts_{model}_{mode}_embeddings.jsonl"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "concept_id": concept_id,
            "embedding": embedding
        }) + "\n")

def save_term_embedding(term_id: int, concept_id: str, term: str, term_type: str, embedding: Sequence[float], model: str):
    """
    Save the embedding for a given term.
    
    Args:
        term_id: The ID of the term.
        concept_id: The ID of the concept.
        term: The term.
        term_type: The type of the term.
        embedding: The embedding of the term.
        model: The model used to generate the embedding.
    """
    path = f"dataset/snomed_terms_{model}_embeddings.jsonl"

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "term_id": term_id,
            "concept_id": concept_id,
            "term": term,
            "term_type": term_type,
            "embedding": embedding
        }) + "\n")

def create_snomed_ct_concepts_and_collections():
    """
    Create the Qdrant collections for SNOMED CT concepts and terms, and populate them with embeddings.
    """

    start_time = time()

    all_snomed_nodes = create_snomed_ct_nodes()
    total_snomed_nodes = len(all_snomed_nodes)
    logger.info(f"Created {total_snomed_nodes} SNOMED CT nodes")

    os.makedirs("dataset/snomed_embeddings", exist_ok=True)

    terms = []
    term_types = []
    term_payload = []
    points = []
    sapbert_vectors_dict = {}
    nomic_vectors_dict = {}
    i = 1
    try:
        for node in tqdm(all_snomed_nodes, total=total_snomed_nodes, desc="Embedding SNOMED CT Terms"):
            # BATCH EMBEDDING
            sapbert_vectors_dict[node["concept_id"]] = []
            nomic_vectors_dict[node["concept_id"]] = []

            terms.append(node["label"])
            term_types.append("label")
            term_payload.append(node)
            
            for t in node["preferred_terms"]:
                terms.append(t)
                term_types.append("preferred")
                term_payload.append({
                    "concept_id": node["concept_id"],
                    "label": t,
                    "label_type": "preferred",
                    "semantic_tag": node["semantic_tag"],
                    "fsn": node["fsn"],
                    "preferred_terms": node["preferred_terms"],
                    "acceptable_terms": node["acceptable_terms"],
                })
            
            for t in node["acceptable_terms"]:
                terms.append(t)
                term_types.append("acceptable")
                term_payload.append({
                    "concept_id": node["concept_id"],
                    "label": t,
                    "label_type": "acceptable",
                    "semantic_tag": node["semantic_tag"],
                    "fsn": node["fsn"],
                    "preferred_terms": node["preferred_terms"],
                    "acceptable_terms": node["acceptable_terms"],
                })

            # BATCH INSERT
            if len(terms) >= 100:
                sapbert_vectors = embed("sapbert", terms)
                nomic_vectors = embed("nomic-embed-text", terms)

                for idx, (sapbert_vector, nomic_vector) in enumerate(zip(sapbert_vectors, nomic_vectors)):
                    points.append(
                        PointStruct(
                            id=i,
                            vector={
                                "sapbert_vector": sapbert_vector,
                                "nomic_vector": nomic_vector
                            },
                            payload=term_payload[idx]
                        )
                    )
                    i += 1
                    sapbert_vectors_dict[term_payload[idx]["concept_id"]].append(sapbert_vector)
                    nomic_vectors_dict[term_payload[idx]["concept_id"]].append(nomic_vector)
                    save_term_embedding(
                        term_id=i, 
                        concept_id=term_payload[idx]["concept_id"],
                        term=term_payload[idx]["label"],
                        term_type=term_payload[idx]["label_type"],
                        embedding=sapbert_vector,
                        model="sapbert"
                    )
                    save_term_embedding(
                        term_id=i, 
                        concept_id=term_payload[idx]["concept_id"],
                        term=term_payload[idx]["label"],
                        term_type=term_payload[idx]["label_type"],
                        embedding=nomic_vector,
                        model="nomic-embed-text"
                    )
                qdrant_source.upsert(collection_name="snomed_ct_terms", points=points)

                terms = []
                term_types = []
                term_payload = []
                points = []

        # FINAL BATCH
        if terms:
            sapbert_vectors = embed("sapbert", terms)
            nomic_vectors = embed("nomic-embed-text", terms)

            for idx, (sapbert_vector, nomic_vector) in enumerate(zip(sapbert_vectors, nomic_vectors)):
                points.append(
                    PointStruct(
                        id=i,
                        vector={
                            "sapbert_vector": sapbert_vector,
                            "nomic_vector": nomic_vector
                        },
                        payload=term_payload[idx]
                    )
                )
                i += 1
                sapbert_vectors_dict[term_payload[idx]["concept_id"]].append(sapbert_vector)
                nomic_vectors_dict[term_payload[idx]["concept_id"]].append(nomic_vector)
                save_term_embedding(
                    term_id=i, 
                    concept_id=term_payload[idx]["concept_id"],
                    term=term_payload[idx]["label"],
                    term_type=term_payload[idx]["label_type"],
                    embedding=sapbert_vector,
                    model="sapbert"
                )
                save_term_embedding(
                    term_id=i, 
                    concept_id=term_payload[idx]["concept_id"],
                    term=term_payload[idx]["label"],
                    term_type=term_payload[idx]["label_type"],
                    embedding=nomic_vector,
                    model="nomic-embed-text"
                )
            qdrant_source.upsert(collection_name="snomed_ct_terms", points=points)

            terms = []
            term_types = []
            term_payload = []
            points = []

    except Exception as e:
        logging.error(f"Error while creating SNOMED CT Terms Points: {e}\n{traceback.format_exc()}")
    

    points = []
    batch_texts = []
    batch_nodes = []
    try:
        for node in tqdm(all_snomed_nodes, total=total_snomed_nodes, desc="Embedding SNOMED CT Concepts"):
            
            # BATCH EMBEDDING
            concept_text = f"search_document: {build_embedding_text(node)}"
            batch_texts.append(concept_text)
            batch_nodes.append({
                **node,
                "label_type": "label"
            })

            # BATCH INSERT
            if len(batch_texts) == 100:
                sapbert_vectors = embed("sapbert", batch_texts)
                nomic_vectors = embed("nomic-embed-text", batch_texts)

                for idx, (sapbert_vector, nomic_vector) in enumerate(zip(sapbert_vectors, nomic_vectors)):
                    avg_nomic_vector = avg_embedding(nomic_vectors_dict[batch_nodes[idx]["concept_id"]])
                    avg_sapbert_vector = avg_embedding(sapbert_vectors_dict[batch_nodes[idx]["concept_id"]])
                    points.append(
                        PointStruct(
                            id=int(batch_nodes[idx]["concept_id"]),
                            vector={
                                "concept_nomic_vector": nomic_vector,
                                "concept_sapbert_vector": sapbert_vector,
                                "terms_avg_nomic_vector": avg_nomic_vector,
                                "terms_avg_sapbert_vector": avg_sapbert_vector,
                            },
                            payload=batch_nodes[idx]
                        )
                    )
                    save_concept_embedding(batch_nodes[idx]["concept_id"], nomic_vector, "nomic-embed-text", "concept")
                    save_concept_embedding(batch_nodes[idx]["concept_id"], sapbert_vector, "sapbert", "concept")
                    save_concept_embedding(batch_nodes[idx]["concept_id"], avg_nomic_vector, "nomic-embed-text", "avg")
                    save_concept_embedding(batch_nodes[idx]["concept_id"], avg_sapbert_vector, "sapbert", "avg")
                qdrant_source.upsert(collection_name="snomed_ct_concepts", points=points)
                points = []
                batch_texts = []
                batch_nodes = []

        # FINAL BATCH
        if batch_texts:
            sapbert_vectors = embed("sapbert", batch_texts)
            nomic_vectors = embed("nomic-embed-text", batch_texts)

            for idx, (sapbert_vector, nomic_vector) in enumerate(zip(sapbert_vectors, nomic_vectors)):
                avg_nomic_vector = avg_embedding(nomic_vectors_dict[batch_nodes[idx]["concept_id"]])
                avg_sapbert_vector = avg_embedding(sapbert_vectors_dict[batch_nodes[idx]["concept_id"]])
                points.append(
                    PointStruct(
                        id=int(batch_nodes[idx]["concept_id"]),
                        vector={
                            "concept_nomic_vector": nomic_vector,
                            "concept_sapbert_vector": sapbert_vector,
                            "terms_avg_nomic_vector": avg_nomic_vector,
                            "terms_avg_sapbert_vector": avg_sapbert_vector,
                        },
                        payload=batch_nodes[idx]
                    )
                )
                save_concept_embedding(batch_nodes[idx]["concept_id"], nomic_vector, "nomic-embed-text", "concept")
                save_concept_embedding(batch_nodes[idx]["concept_id"], sapbert_vector, "sapbert", "concept")
                save_concept_embedding(batch_nodes[idx]["concept_id"], avg_nomic_vector, "nomic-embed-text", "avg")
                save_concept_embedding(batch_nodes[idx]["concept_id"], avg_sapbert_vector, "sapbert", "avg")
            qdrant_source.upsert(collection_name="snomed_ct_concepts", points=points)
            points = []
            batch_texts = []
            batch_nodes = []

    except Exception as e:
        logging.error(f"Error while creating SNOMED CT concepts Points: {e}\n{traceback.format_exc()}")

    elapsed = time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    logging.info(f"Pipeline completata in {hours}h {minutes}m {seconds}s")
