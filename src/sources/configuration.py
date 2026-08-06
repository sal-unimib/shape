import json
from src.model.enums import ShapeStage
from src.model.configuration import ReasoningConfiguration, RetrievalConfiguration, StageConfiguration
from typing import Dict

shape_configuration : Dict[ShapeStage, StageConfiguration] = {}

def get_configuration(stage: ShapeStage) -> StageConfiguration:
    """
    Returns the configuration for the specified stage of the pipeline.
    """
    global shape_configuration
    if not shape_configuration:
        with open("src/resources/hyperparameters.json", "r") as f:
            hyperparameters = json.load(f)

        with open("src/resources/retrieval.json", "r") as f:
            retrieval = json.load(f)
            
        for s in ShapeStage:
            stage_hyperparameters_dict = hyperparameters.get(s.value, {})
            stage_reasoning = {}
            stage_retrieval_dict = retrieval.get(s.value, {})
            stage_retrieval = {}

            for task, value in stage_hyperparameters_dict.items():
                with open(f"src/resources/prompts/{task}.txt", "r") as f:
                    prompt = f.read()
                stage_reasoning[task] = ReasoningConfiguration(
                    hyperparameters=value,
                    prompt=prompt
                )

            for task, value in stage_retrieval_dict.items():
                stage_retrieval[task] = RetrievalConfiguration(
                    collection=value['collection'],
                    embedding=value['embedding'],
                    topk=value['topk']
                )
            
            shape_configuration[s] = StageConfiguration(
                retrieval=stage_retrieval,
                reasoning=stage_reasoning,
            )
    return shape_configuration[stage]
