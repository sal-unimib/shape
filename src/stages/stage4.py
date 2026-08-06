from src.model.core import Relationship
from fhir.resources.activitydefinition import ActivityDefinition
import logging
import json

from src.sources.configuration import get_configuration
from src.model.core import TemporalConstraint
from src.model.enums import LLMConfidence, ParameterCategory, ShapeStage, SnomedLabelType, TemporalType
from src.model.parametrized_workflow import ActionParameter, ParametrizedAction, ParametrizedWorkflow
from src.model.fhir_workflow import (
    FHIRAction,
    FHIRParameter,
    FHIRTemporalConstraint,
    FHIRWorkflow,
    OntologyCandidate,
    OntologyParameter,
    SnomedConcept
)
from src.sources.llm import ollama_source, embed, llm_models
from src.sources.qdrant import qdrant_source

import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger('shape.stages.stage4')
logger.setLevel(logging.DEBUG)

stage_configuration = get_configuration(ShapeStage.STAGE4)
ontology_grounding_retrieval_configuration = stage_configuration.retrieval['ontology_grounding']
ontology_grounding_reasoning_configuration = stage_configuration.reasoning['ontology_grounding']
fhir_valueset_grounding_reasoning_configuration = stage_configuration.reasoning['fhir_valueset_grounding']
fhir_parameter_generation_reasoning_configuration = stage_configuration.reasoning['fhir_parameter_generation']
fhir_temporal_constraint_generation_reasoning_configuration = stage_configuration.reasoning['fhir_temporal_constraint_generation']

with open("src/resources/parameters/parameters.json", "r") as f:
    parameters_category = json.load(f)

def get_parameter_category(activity_definition: ActivityDefinition, parameter: ActionParameter) -> ParameterCategory:
    """
    Retrieves the category of a parameter based on the activity definition and parameter path.

    Args:
        activity_definition (ActivityDefinition): The FHIR ActivityDefinition associated with the action.
        parameter (ActionParameter): The action parameter whose category needs to be retrieved.

    Returns:
        ParameterCategory: The category of the parameter (e.g., ONTOLOGY, VALUE_SET, REFERENCE, etc.).
    """
    return ParameterCategory(parameters_category[activity_definition.id][parameter.path]["category"])

def ontology_parameter_resolution(action: ParametrizedAction, parameter: ActionParameter, llm: str) -> Optional[Tuple[Dict[str, Any], List[OntologyCandidate]]]:
    """
    Resolves an ontology-based parameter by retrieving candidate SNOMED CT concepts and using an LLM to select the most appropriate one.

    This function performs the following steps:
    1. Retrieves information about the parameter from the configuration based on the activity definition and parameter path.
    2. Generates an embedding for the parameter's label using the configured embedding model.
    3. Queries the QDrant vector database to find candidate SNOMED CT concepts similar to the parameter label.
    4. Constructs a prompt for the LLM containing the parameter information and candidate concepts.
    5. Generates the LLM response to select the appropriate candidate concept.

    Args:
        action (ParametrizedAction): The parametrized action containing the parameter to resolve.
        parameter (ActionParameter): The action parameter to resolve.
        llm (str): The identifier of the LLM model to use for reasoning.

    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]: A tuple containing:
            - The LLM response with the selected ontology concept.
            - The list of candidate concepts with their similarity scores and retrieval ranks.

    Raises:
        Exception: If there are persistent errors while querying QDrant after all retry attempts.
    """
    param_info = parameters_category[action.activity_definition.id][parameter.path]
    label_embedding = embed(ontology_grounding_retrieval_configuration.embedding, [parameter.label])[0]

    max_retries = 3
    delay_seconds = 2
    candidate_concepts_points = []
    for attempt in range(max_retries):
        try:
            candidate_concepts_points = qdrant_source.query_points(
                collection_name=ontology_grounding_retrieval_configuration.collection,
                query=label_embedding,
                limit=ontology_grounding_retrieval_configuration.topk
            ).points
            break
        except Exception as e:
            logger.warning("Error while attempting to query QDrant")
            logger.warning("Attempt %d failed: %s", attempt + 1, str(e))
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay_seconds)

    candidate_concepts = []
    for i, point in enumerate(sorted(candidate_concepts_points, key=lambda x: x.score, reverse=True)):
        if point.payload:
            candidate_concepts.append(OntologyCandidate(
                ontology_concept=SnomedConcept(
                    concept_id=point.payload["concept_id"],
                    label=point.payload["term"],
                    label_type=SnomedLabelType(point.payload["term_type"]),
                    semantic_tag=point.payload["semantic_tag"],
                    fsn=point.payload["fsn"],
                    preferred_terms=point.payload["preferred_terms"],
                    acceptable_terms=point.payload["acceptable_terms"]
                ),
                similarity_score=point.score,
                retrieval_rank=i+1
            ))
        else:
            logger.warning("Point has no payload:\n%s", point)

    candidate_concepts_str = ""
    for cand_conc in candidate_concepts:
        candidate_concepts_str += f"ID: {cand_conc.ontology_concept.concept_id} - label: {cand_conc.ontology_concept.label}:\n"
        candidate_concepts_str += f"\t- semantic_tag: {cand_conc.ontology_concept.semantic_tag}\n"
        pt_str = ", ".join([pt for pt in cand_conc.ontology_concept.preferred_terms[:3] or []])
        candidate_concepts_str += f"\t- preferred_terms: {pt_str}\n"
    
    logger.debug("Candidate SNOMED concepts retrieved: %s", candidate_concepts)
    prompt = (ontology_grounding_reasoning_configuration.prompt.replace("{{label}}", parameter.label)
                .replace("{{parameter_role}}", param_info["description"])
                .replace("{{parameter_description}}", action.label)
                .replace("{{snomed_concepts}}", candidate_concepts_str))
    llm_response = ollama_source.generate(
        model = llm_models[llm],
        format='json',
        stream=False,
        prompt=prompt,
        options=ontology_grounding_reasoning_configuration.hyperparameters
    )
    logger.debug("LLM Snomed Selection response: %s", llm_response.response)
    try:
        return json.loads(llm_response.response), candidate_concepts
    except:
        return

def fhir_parameter_resolution(activity_definition: ActivityDefinition, parameter: ActionParameter, llm: str) -> Optional[Dict[str, Any]]:
    """
    Resolves a FHIR parameter using an LLM.

    This function performs the following steps:
    1. Retrieves information about the parameter from the configuration based on the activity definition and parameter path.
    2. Generates an embedding for the parameter's label using the configured embedding model.
    3. Queries the QDrant vector database to find candidate SNOMED CT concepts similar to the parameter label.
    4. Constructs a prompt for the LLM containing the parameter information and candidate concepts.
    5. Generates the LLM response to select the appropriate candidate concept.

    Args:
        activity_definition (ActivityDefinition): The activity definition containing the parameter to resolve.
        parameter (ActionParameter): The action parameter to resolve.
        llm (str): The identifier of the LLM model to use for reasoning.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the resolved FHIR parameter, or None if the parameter could not be resolved.
    """
    param_info = parameters_category[activity_definition.id][parameter.path]
    
    prompt = (fhir_parameter_generation_reasoning_configuration.prompt.replace("{{label}}", parameter.label)
        .replace("{{fhir_schema_text}}", param_info["value"])
        .replace("{{parameter_description}}", param_info['description']))
    llm_response = ollama_source.generate(
        model = llm_models[llm],
        format='json',
        stream=False,
        prompt=prompt,
        options=fhir_parameter_generation_reasoning_configuration.hyperparameters
    )
    logger.debug("FHIR Primitive Generation - LLM Response: %s", llm_response.response)
    try:
        return json.loads(llm_response.response)
    except:
        return

def fhir_valueset_resolution(activity_definition: ActivityDefinition, parameter: ActionParameter, llm: str) -> Optional[Dict[str, Any]]:
    """
    Resolves a FHIR ValueSet using an LLM.

    This function performs the following steps:
    1. Retrieves information about the parameter from the configuration based on the activity definition and parameter path.
    2. Generates an embedding for the parameter's label using the configured embedding model.
    3. Queries the QDrant vector database to find candidate SNOMED CT concepts similar to the parameter label.
    4. Constructs a prompt for the LLM containing the parameter information and candidate concepts.
    5. Generates the LLM response to select the appropriate candidate concept.

    Args:
        activity_definition (ActivityDefinition): The activity definition containing the parameter to resolve.
        parameter (ActionParameter): The action parameter to resolve.
        llm (str): The identifier of the LLM model to use for reasoning.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the resolved FHIR parameter, or None if the parameter could not be resolved.
    """
    param_info = parameters_category[activity_definition.id][parameter.path]

    valueset_entries = ""
    for vs in param_info["value"]:
        valueset_entries += f"* Code: {vs['value']} - Title: {vs['title']} - Description: {vs['description']}\n"

    prompt = (fhir_valueset_grounding_reasoning_configuration.prompt.replace("{{label}}", parameter.label)
            .replace("{{parameter_description}}", param_info["description"])
            .replace("{{valueset_entries}}", valueset_entries))
    llm_response = ollama_source.generate(
        model = llm_models[llm],
        format='json',
        stream=False,
        prompt=prompt,
        options=fhir_valueset_grounding_reasoning_configuration.hyperparameters
    )
    logger.debug("FHIR Primitive Generation - LLM Response: %s", llm_response.response)
    try:
        return json.loads(llm_response.response)
    except:
        return

def temporal_constraint_resolution(temporal_constraint : TemporalConstraint, llm: str) -> Optional[Dict[str, Any]]:
    """
    Resolves a temporal constraint using an LLM.

    Args:
        temporal_constraint (TemporalConstraint): The temporal constraint to resolve.
        llm (str): The identifier of the LLM model to use for reasoning.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the resolved temporal constraint, or None if the temporal constraint could not be resolved.
    """
    if temporal_constraint.type == TemporalType.OFFSET:
        with open("src/resources/parameters/offset_duration.txt", "r") as f:
            temporal_struct = f.read()
    else:
        with open("src/resources/parameters/timing.txt", "r") as f:
            temporal_struct = f.read()
    prompt = fhir_temporal_constraint_generation_reasoning_configuration.prompt.replace("{{fhir_schema_text}}", temporal_struct).replace("{{label}}", temporal_constraint.label)
    llm_response = ollama_source.generate(
        model = llm_models[llm],
        format='json',
        stream=False,
        prompt=prompt,
        options=fhir_temporal_constraint_generation_reasoning_configuration.hyperparameters
    )
    logger.debug("Temporal Constraint FHIR Generation - LLM Response: %s", llm_response.response)
    try:
        return json.loads(llm_response.response) if temporal_constraint.type == TemporalType.OFFSET else {"repeat" : json.loads(llm_response.response)}
    except:
        return

def fhir_representation_generation(parametrized_workflow: ParametrizedWorkflow, llm: str) -> FHIRWorkflow:
    """
    Step 4: FHIR representation generation
    This function converts a parametrized workflow into a FHIR workflow.

    Args:
        parametrized_workflow (ParametrizedWorkflow): The parametrized workflow to convert.
        llm (str): The identifier of the LLM model to use for reasoning.

    Returns:
        FHIRWorkflow: The FHIR workflow corresponding to the parametrized workflow.
    """

    fhir_actions = []

    for action in parametrized_workflow.parametrized_actions:
        fhir_parameters = []
        for parameter in action.parameters:
            param_category = get_parameter_category(action.activity_definition, parameter)
            if param_category == ParameterCategory.ONTOLOGY:
                ontology_resolution = ontology_parameter_resolution(action, parameter, llm)
                if ontology_resolution:
                    selected_concept, candidate_concepts = ontology_resolution
                    if selected_concept and selected_concept["selected_concept"] and selected_concept["selected_concept"] != {}:
                        selected_value = {
                            "coding" : [{
                                "system" : "http://snomed.info/sct",
                                "code" : selected_concept["selected_concept"]["id"],
                                "display" : selected_concept["selected_concept"]["label"]
                            }],
                            "text" : parameter.label
                        }
                        confidence = selected_concept.get("confidence")
                        reason = None
                    else:
                        selected_value = None
                        confidence = selected_concept.get("confidence")
                        reason = selected_concept.get("reason")
                    ontology_parameter = OntologyParameter(
                        path=parameter.path,
                        label=parameter.label,
                        value=selected_value,
                        candidates=candidate_concepts,
                        confidence=LLMConfidence(confidence) if confidence else LLMConfidence.NONE,
                        reason=reason
                    )
                    fhir_parameters.append(ontology_parameter)
                else:
                    logger.error("%s - %s - %s - Error creating OntologyParameter for parameter: %s : %s", llm, parametrized_workflow.name, action.id, parameter.path, parameter.label)
            elif param_category == ParameterCategory.VALUE_SET:
                valueset_resolution = fhir_valueset_resolution(action.activity_definition, parameter, llm)
                if valueset_resolution and "code" in valueset_resolution.keys():
                    fhir_parameters.append(FHIRParameter(
                        path=parameter.path,
                        category=param_category,
                        label=parameter.label,
                        value=valueset_resolution
                    ))
                else:
                    logger.error("%s - %s - %s - Error creating FHIRParameter (ValueSet) for parameter: %s : %s", llm, parametrized_workflow.name, action.id, parameter.path, parameter.label)
            elif param_category == ParameterCategory.RESOURCE:
                resource_resolution = fhir_parameter_resolution(action.activity_definition, parameter, llm)
                if resource_resolution:
                    fhir_parameters.append(FHIRParameter(
                        path=parameter.path,
                        category=param_category,
                        label=parameter.label,
                        value=resource_resolution
                    ))
                else:
                    logger.error("%s - %s - %s - Error creating FHIRParameter (Resource) for parameter: %s : %s", llm, parametrized_workflow.name, action.id, parameter.path, parameter.label)
            elif param_category in [ParameterCategory.STRING, ParameterCategory.REFERENCE]:
                fhir_parameters.append(FHIRParameter(
                    path=parameter.path,
                    category=param_category,
                    label=parameter.label,
                    value=parameter.label
                ))
            else:
                logger.error("%s - %s - %s - %s - Unknown parameter category: %s", llm, parametrized_workflow.name, action.id, parameter.path, param_category)
        fhir_tc = None
        if action.temporal_constraint:
            tc_encoded = temporal_constraint_resolution(action.temporal_constraint, llm)
            fhir_tc = FHIRTemporalConstraint(type=action.temporal_constraint.type, value=tc_encoded)
        fhir_actions.append(FHIRAction(
            action=action,
            fhir_parameters=fhir_parameters,
            fhir_temporal_constraint=fhir_tc
        ))
    
    return FHIRWorkflow(
        id = parametrized_workflow.id,
        name = parametrized_workflow.name,
        title = parametrized_workflow.title,
        description = parametrized_workflow.description,
        fhir_actions = fhir_actions,
        relationships = parametrized_workflow.relationships,
    )
