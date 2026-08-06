from src.sources.configuration import get_configuration
from src.model.core import Action, ActionCondition, Relationship, TemporalConstraint
from src.model.enums import ShapeStage
from src.model.narrative_pathway import NarrativePathway
from src.model.procedural_workflow import ProceduralWorkflow
from src.sources.llm import ollama_source, llm_models

import logging
import json

logger = logging.getLogger('shape.stages.stage1')
logger.setLevel(logging.DEBUG)

stage_configuration = get_configuration(ShapeStage.STAGE1)
task_configuration = stage_configuration.reasoning['action_identification_and_relationship_modeling']

def action_Identification_and_relationship_modeling(pathway : NarrativePathway, llm: str) -> ProceduralWorkflow:
    """
    Step 1: Action identification and relationship modeling
    This function identifies the actions in a narrative pathway and models the relationships between them.

    Args:
        pathway (NarrativePathway): The narrative pathway to process
        llm (str): The identifier of the LLM model to use for reasoning

    Returns:
        ProceduralWorkflow: The procedural workflow with identified actions and modeled relationships
    """

    gen_prompt = f"{task_configuration.prompt}\n{pathway.description}"
    llm_response = ollama_source.generate(
		model = llm_models[llm],
		format='json',
		stream=False,
		prompt=gen_prompt,
		options=task_configuration.hyperparameters
	)
    logger.debug("%s - %s - LLM response for Action identification & Relationship modeling: \n%s", llm, pathway.name, llm_response.response)
    llm_resp_dict = json.loads(llm_response.response)

    actions_list = []
    for action in llm_resp_dict.get('actions', []):
        try:
            act = Action(id=action.get('id'), label=action.get('label'))

            try:
                temporal_constraint_dict = action.get('temporal_constraint')
                if temporal_constraint_dict:
                    temporal_constraint = TemporalConstraint(**temporal_constraint_dict)
                else:
                    temporal_constraint = None
            except:
                logger.error("%s - %s - %s - Error creating TemporalConstraint from LLM response:\n%s", llm, pathway.name, act.id, action.get('temporal_constraint'))
                temporal_constraint = None
    
            try:
                condition_dict = action.get('condition')
                if condition_dict:
                    condition = ActionCondition(**condition_dict)
                else:
                    condition = None
            except:
                logger.error("%s - %s - %s - Error creating ActionCondition from LLM response:\n%s", llm, pathway.name, act.id, action.get('condition'))
                condition = None

            act.temporal_constraint = temporal_constraint
            act.condition = condition
            actions_list.append(act)
        except:
            logger.error("%s - %s - Error creating base Action from LLM response:\n%s", llm, pathway.name, action)

        
    relationships_list = []
    for rel in llm_resp_dict.get('relationships', []):
        try:
            relationship = Relationship(**rel)
            if relationship.from_action and relationship.to_action:
                relationships_list.append(relationship)
            else:
                logger.warning("%s - %s - Relationship from LLM response does not have both from_action and to_action:\n%s", llm, pathway.name, rel)
        except:
            logger.error("%s - %s - Error creating Relationship from LLM response: %s", llm, pathway.name, rel)
        
    return ProceduralWorkflow(
        id=pathway.id,
        name=pathway.name,
        title=pathway.title,
        description=pathway.description,
        actions=actions_list,
        relationships=relationships_list
    )
	