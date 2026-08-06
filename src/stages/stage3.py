import logging
import json

from src.sources.configuration import get_configuration
from src.model.grounded_workflow import GroundedWorkflow
from src.model.parametrized_workflow import ActionParameter, ParametrizedAction, ParametrizedWorkflow
from src.model.enums import ShapeStage
from src.sources.llm import ollama_source, llm_models

logger = logging.getLogger('shape.stages.stage3')
logger.setLevel(logging.DEBUG)

stage_configuration = get_configuration(ShapeStage.STAGE3)
task_configuration = stage_configuration.reasoning['clinical_parameter_resolution']

def clinical_parameter_resolution(grounded_workflow: GroundedWorkflow, llm: str) -> ParametrizedWorkflow:
    """
    Step 3: Clinical parameter resolution
    This function resolves the dynamic values of the actions in a grounded workflow
    by identifying the parameters that need to be filled and populating them with
    the appropriate values.

    Args:
        grounded_workflow (GroundedWorkflow): The grounded workflow to resolve
        llm (str): The identifier of the LLM model to use for reasoning

    Returns:
        ParametrizedWorkflow: The parametrized workflow with resolved parameters
    """
    parametrized_actions = []

    for action in grounded_workflow.grounded_actions:
        prompt_dvs = ""
        action_dvs = {}
        if action.activity_definition:
            for dv in action.activity_definition.dynamicValue or []:
                dv_dict = {
                    'id' : dv.expression.expression.replace('%', '') if dv.expression.expression else 'N/A',
                    'path' : dv.path,
                    'description' : dv.expression.description
                }
                action_dvs[dv_dict['id']] = dv_dict
                prompt_dvs += f"\t- ID: {dv_dict['id']} - description: {dv_dict['description']}\n"
            
            prompt = task_configuration.prompt.replace("{{action_text}}", action.label).replace("{{formatted_parameters}}", prompt_dvs)
            llm_response = ollama_source.generate(
                model = llm_models[llm],
                format='json',
                stream=False,
                prompt=prompt,
                options=task_configuration.hyperparameters
            )
            logger.debug("%s - %s - %s - LLM response for Clinical Parameter Resolution:\n%s", llm, grounded_workflow.name, action.id, llm_response.response)
            llm_resp_dict = json.loads(llm_response.response)
            identified_parameters = llm_resp_dict.get("parameters", [])
            action_param_list = []
            for parameter in identified_parameters:
                try:
                    param_path = action_dvs[parameter['id']]['path']
                    action_param = ActionParameter(
                        path=param_path,
                        label=parameter.get("label")
                    )
                    action_param_list.append(action_param)
                except:
                    logger.error("%s - %s - %s - Error creating ActionParameter from LLM response:\n%s", llm, grounded_workflow.name, action.id, parameter)
            parametrized_actions.append(ParametrizedAction(action, action_param_list))
        else:
            parametrized_actions.append(ParametrizedAction(action, []))


    return ParametrizedWorkflow(
        id=grounded_workflow.id,
        name=grounded_workflow.name,
        title=grounded_workflow.title,
        description=grounded_workflow.description,
        parametrized_actions=parametrized_actions,
        relationships=grounded_workflow.relationships
    )