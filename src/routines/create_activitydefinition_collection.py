from fhir.resources.activitydefinition import ActivityDefinition
import json
import logging
import os
from qdrant_client.models import PointStruct
from src.sources.llm import embed
from src.sources.qdrant import qdrant_source, create_collection_if_not_exists

logger = logging.getLogger('shape.routines.create_activitydefinition_collection')
logger.setLevel(logging.DEBUG)

async def create_activitydefinition_collection(embed_model : str, collection_name : str):
    logger.info("Starting ActivityDefinition fetch...")
    
    base_path = "'dataset/activitydefinition'"

    bundle = []

    for filename in os.listdir(base_path):
        with open(os.path.join(base_path, filename), 'r') as f:
            data = json.load(f)
        bundle.append(ActivityDefinition.model_validate(data))
    
    logger.info("Retrieved %s ActivityDefinition resources", len(bundle))

    # 2. Generate ActivityDefinition point for QDrant VectorDB
    ad_points_dict = [] 
    for activity in bundle:
        ad_points_dict.append({
            "id" : activity.id,
            "title": activity.title,
            "subtitle": activity.subtitle,
            "description": activity.description,
            "resource": activity.model_dump_json(),
            "doc": f"title: {activity.title}\nsubtitle: {activity.subtitle}\ndescription: {activity.description}"
        })
    
    # 3. Generate and append Embedding
    logger.info("Generating embeddings for %s ActivityDefinition resources using %s", len(ad_points_dict), embed_model)
    embeddings = embed([action['doc'] for action in ad_points_dict], embed_model)
    for i in range(len(embeddings)):
        ad_points_dict[i]['embedding'] = embeddings[i]

    logger.info("Embeddings generated for %s ActivityDefinition resources", len(ad_points_dict))

    # 4. Store points in QDrant VectorDB
    create_collection_if_not_exists(collection_name, len(embeddings[0]))
    logger.info("Storing %s ActivityDefinition points in QDrant", len(ad_points_dict))
    points = []
    for point in ad_points_dict:
        points.append(PointStruct(
            id=point['id'],
            vector=point['embedding'],
            payload={
                "id" : point['id'],
                "title": point['title'],
                "subtitle": point['subtitle'],
                "description": point['description'],
                "resource": point['resource']
            }
        ))
    qdrant_source.upsert(
        collection_name=collection_name,
        points=points
    )
    logger.info("Stored %s ActivityDefinition points in QDrant", len(ad_points_dict))
