from fhir.resources.plandefinition import PlanDefinition, PlanDefinitionActionDynamicValue
from fhir.resources.timing import Timing
from fhir.resources.duration import Duration
import logging
import json
from typing import Any, Dict
from src.model.enums import TemporalType
from src.model.fhir_workflow import FHIRWorkflow

logger = logging.getLogger('shape.stages.stage5')
logger.setLevel(logging.DEBUG)

def deterministic_plandefinition_construction(fhir_workflow : FHIRWorkflow) -> PlanDefinition:
    """
    Step 5: Deterministic PlanDefinition construction

    Args:
        fhir_workflow (FHIRWorkflow): The FHIR workflow to convert to PlanDefinition

    Returns:
        PlanDefinition: The PlanDefinition corresponding to the FHIR workflow
    """
    plandefinition_dict = {
        "id" : fhir_workflow.id,
        'name' : fhir_workflow.name,
        'title' : fhir_workflow.title,
        'description' : fhir_workflow.description,
        'status' : 'draft',
        'action': []
    }

    for action in fhir_workflow.fhir_actions:
        pd_action : Dict[str, Any] = {
            'linkId' : action.id,
            'title' : action.label
        }
        if action.condition:
            pd_action['condition'] = [{
                "kind": "applicability",
                "expression": {
                    "description": action.condition.label,
                    "language": "text/fhirpath",
                    "expression": "TODO"
                }
            }]

        related_actions = []
        for relationship in fhir_workflow.relationships:
            if relationship.to_action == action.id and relationship.from_action is not None:
                related_actions.append({
                    "targetId": relationship.from_action,
                    "relationship": "after-end",
                })
        
        if action.fhir_temporal_constraint:
            if action.fhir_temporal_constraint.type in [TemporalType.DURATION, TemporalType.REPEAT]:
                try:
                    Timing.model_validate(action.fhir_temporal_constraint.value)
                    pd_action['timingTiming'] = action.fhir_temporal_constraint.value
                except:
                    logger.warning("%s - %s - %s - Error creating Timing from FHIRTemporalConstraint:\n%s", fhir_workflow.id, fhir_workflow.name, action.id, action.fhir_temporal_constraint.value)
            if action.fhir_temporal_constraint.type == TemporalType.OFFSET:
                for rel_act in related_actions:
                    try:
                        Duration.model_validate(action.fhir_temporal_constraint.value)
                        rel_act['offsetDuration'] = action.fhir_temporal_constraint.value
                    except:
                        logger.warning("%s - %s - %s - Error creating Duration from FHIRTemporalConstraint: %s", fhir_workflow.id, fhir_workflow.name, action.id, action.fhir_temporal_constraint.value)

        if len(related_actions) > 0:
            pd_action['relatedAction'] = related_actions

        dynamic_values = []
        for parameter in action.fhir_parameters:
            try:
                dv = {
                    "path" : parameter.path,
                    "expression": {
                        "language" : "text/fhirpath",
                        "description": parameter.label,
                        "expression": json.dumps(parameter.value) if type(parameter.value) == dict else str(parameter.value)
                    }
                }
                PlanDefinitionActionDynamicValue.model_validate(dv)
                dynamic_values.append(dv)
            except:
                logger.warning("%s - %s - %s - Error creating DynamicValue from FHIRParameter: %s", fhir_workflow.id, fhir_workflow.name, action.id, parameter.value)

        if len(dynamic_values) > 0:
            pd_action['dynamicValue'] = dynamic_values

        plandefinition_dict["action"].append(pd_action)

    return PlanDefinition.model_validate(plandefinition_dict)


