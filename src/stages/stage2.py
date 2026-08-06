from fhir.resources.activitydefinition import ActivityDefinition
import logging
import json

from src.sources.configuration import get_configuration
from src.model.core import Action
from src.model.enums import ShapeStage, LLMConfidence
from src.model.grounded_workflow import ActivityCandidate, GroundedAction, GroundedWorkflow
from src.model.procedural_workflow import ProceduralWorkflow
from src.sources.llm import embed, ollama_source, llm_models
from src.sources.qdrant import qdrant_source

import time
from typing import List, Sequence

logger = logging.getLogger('shape.stages.stage2')
logger.setLevel(logging.DEBUG)

stage_configuration = get_configuration(ShapeStage.STAGE2)
retrieval_configuration = stage_configuration.retrieval['clinical_activity_grounding']
reasoning_configuration = stage_configuration.reasoning['clinical_activity_grounding']

def retrieve_candidates(embedding: Sequence[float]) -> List[ActivityCandidate]:
    """
    Retrieves candidate ActivityDefinitions for a given embedding.
    This function queries the Qdrant vector database to find candidate ActivityDefinitions.
    
    Args:
        embedding (Sequence[float]): The embedding of the action label.
    
    Returns:
        List[ActivityCandidate]: A list of candidate ActivityDefinitions.

    Raises:
        Exception: If there are persistent errors while querying Qdrant after all retry attempts.
    """
    max_retries = 3
    delay_seconds = 2
    query_res = []
    for attempt in range(max_retries):
        try:
            query_res = qdrant_source.query_points(
                collection_name=retrieval_configuration.collection,
                query=embedding,
                limit=retrieval_configuration.topk
            ).points
            break
        except Exception as e:
            logger.warning("Error while attempting to query QDrant")
            logger.warning("Attempt %d failed: %s", attempt + 1, str(e))
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay_seconds)
    
    candidates_list = []
    for point in query_res:
        if point.payload:
            candidates_list.append({
                "activity_definition": ActivityDefinition.model_validate(json.loads(point.payload['resource'])),
                "similarity_score": point.score,
            })
        else:
            logger.warning("Point has no payload:\n%s", point)
    
    sorted_candidates = sorted(
        candidates_list,
        key=lambda x: x["similarity_score"],
        reverse=True
    )
    candidates = []
    for rank, (cand) in enumerate(sorted_candidates, start=1):
        candidates.append(ActivityCandidate(
            activity_definition=cand['activity_definition'],
            similarity_score=cand['similarity_score'],
            retrieval_rank=rank
        ))

    return candidates

def clinical_activity_grounding(procedural_workflow: ProceduralWorkflow, llm : str) -> GroundedWorkflow:
    """
    Stage 2: Clinical activity grounding
    This function grounds the actions in a procedural workflow to FHIR ActivityDefinitions.

    Args:
        procedural_workflow (ProceduralWorkflow): The procedural workflow to ground
        llm (str): The identifier of the LLM model to use for reasoning

    Returns:
        GroundedWorkflow: The grounded workflow with grounded actions
    """

    # 1. Generate embeddings for actions' labels
    embeddings = embed(
        model=retrieval_configuration.embedding,
        labels=[f"search_query: {action.label}" for action in procedural_workflow.actions]
    )

    results = []
    for action, embedding in zip(procedural_workflow.actions, embeddings):
        # 2. Retrieve top-ranking candidate from ActivityDefinition vector database
        activity_candidates = retrieve_candidates(embedding)

        candidates_list_str = [cand.prompt_string() for cand in activity_candidates]
        selection_prompt = reasoning_configuration.prompt.replace("{{action}}", action.label).replace("{{candidates}}", "\n\n".join(candidates_list_str))
        logger.debug("%s - %s - %s - Candidates for action:\n%s", llm, procedural_workflow.name, action.id, candidates_list_str)
        llm_response = ollama_source.generate(
            model = llm_models[llm],
            format='json',
            stream=False,
            prompt=selection_prompt,
            options=reasoning_configuration.hyperparameters
        )
        logger.debug("%s - %s - %s - LLM response for Clinical Activity Grounding selection:\n%s", llm, procedural_workflow.name, action.id, llm_response.response)
        llm_resp_dict = json.loads(llm_response.response)

        try:
            results.append({
                "action" : action,
                "candidates" : activity_candidates,
                "selected_id": llm_resp_dict.get('selected_id'),
                "confidence": LLMConfidence(llm_resp_dict.get('confidence')),
                "reason": llm_resp_dict.get('reason')
            })
        except:
            logger.error("%s - %s - %s - Error creating GroundedAction from LLM response:\n%s", llm, procedural_workflow.name, action.id, llm_resp_dict)

    grounded_actions = []
    for result in results:
        ad = next((c for c in result['candidates'] if c.activity_definition.id == result['selected_id']), None)
        grounded_actions.append(GroundedAction(
            action=result['action'],
            activity_definition=ad.activity_definition if ad else None,
            candidates=result['candidates'],
            confidence=result['confidence'],
            reason=result['reason']
        ))

    return GroundedWorkflow(
        id=procedural_workflow.id,
        name=procedural_workflow.name,
        title=procedural_workflow.title,
        description=procedural_workflow.description,
        grounded_actions=grounded_actions,
        relationships=procedural_workflow.relationships
    )

