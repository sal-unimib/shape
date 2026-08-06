import json
import logging
import os
from datetime import datetime
from src.model.narrative_pathway import NarrativePathway
from src.sources.llm import llm_models
from src.stages.stage1 import action_Identification_and_relationship_modeling
from src.stages.stage2 import clinical_activity_grounding
from src.stages.stage3 import clinical_parameter_resolution
from src.stages.stage4 import fhir_representation_generation
from src.stages.stage5 import deterministic_plandefinition_construction
import sys
from time import time

# Logging
logger = logging.getLogger("shape")
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
log_file = os.path.join(log_dir, f"shape_{timestamp}.log")

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.propagate = False

def load_narrative_pathways():
    with open("dataset/clinical_pathways.json", "r") as f:
        narrative_pathways = json.load(f)
    return [NarrativePathway.from_dict(pathway) for pathway in narrative_pathways]

def shape_pipeline(narrative_pathway : NarrativePathway, llm : str):
    logger.info("%s - %s - SHAPE Pipeline started", llm, narrative_pathway.name)

    experiment_info = {}
    
    # Stage 1
    start_time = time()
    procedural_workflow = action_Identification_and_relationship_modeling(narrative_pathway, llm)
    logger.debug("%s - %s - SHAPE Pipeline stage 1 completed", llm, narrative_pathway.name)
    end_time_stage1 = time()
    logger.debug("%s - %s - SHAPE Pipeline stage 1 completed in %s seconds", llm, narrative_pathway.name, end_time_stage1 - start_time)
    with open(f"output/{llm}/{narrative_pathway.name}_output.json", 'w') as f:
        json.dump(procedural_workflow.to_dict(), f)
    experiment_info['stage1_execution_time'] = end_time_stage1 - start_time
    with open(f"output/{llm}/{narrative_pathway.name}_info.json", 'w') as f:
        json.dump(experiment_info, f)

    # Stage 2
    start_time_stage2 = time()
    grounded_workflow = clinical_activity_grounding(procedural_workflow, llm)
    logger.debug("%s - %s - SHAPE Pipeline stage 2 completed", llm, narrative_pathway.name)
    end_time_stage2 = time()
    logger.debug("%s - %s - SHAPE Pipeline stage 2 completed in %s seconds", llm, narrative_pathway.name, end_time_stage2 - start_time_stage2)
    with open(f"output/{llm}/{narrative_pathway.name}_output.json", 'w') as f:
        json.dump(grounded_workflow.to_dict(), f)
    experiment_info['stage2_execution_time'] = end_time_stage2 - start_time_stage2
    with open(f"output/{llm}/{narrative_pathway.name}_info.json", 'w') as f:
        json.dump(experiment_info, f)

    # Stage 3
    start_time_stage3 = time()
    parametrized_workflow = clinical_parameter_resolution(grounded_workflow, llm)
    logger.debug("%s - %s - SHAPE Pipeline stage 3 completed", llm, narrative_pathway.name)
    end_time_stage3 = time()
    logger.debug("%s - %s - SHAPE Pipeline stage 3 completed in %s seconds", llm, narrative_pathway.name, end_time_stage3 - start_time_stage3)
    with open(f"output/{llm}/{narrative_pathway.name}_output.json", 'w') as f:
        json.dump(parametrized_workflow.to_dict(), f)
    experiment_info['stage3_execution_time'] = end_time_stage3 - start_time_stage3
    with open(f"output/{llm}/{narrative_pathway.name}_info.json", 'w') as f:
        json.dump(experiment_info, f)

    # Stage 4
    start_time_stage4 = time()
    fhir_workflow = fhir_representation_generation(parametrized_workflow, llm)
    logger.debug("%s - %s - SHAPE Pipeline stage 4 completed", llm, narrative_pathway.name)
    end_time_stage4 = time()
    logger.debug("%s - %s - SHAPE Pipeline stage 4 completed in %s seconds", llm, narrative_pathway.name, end_time_stage4 - start_time_stage4)
    with open(f"output/{llm}/{narrative_pathway.name}_output.json", 'w') as f:
        json.dump(fhir_workflow.to_dict(), f)
    experiment_info['stage4_execution_time'] = end_time_stage4 - start_time_stage4
    with open(f"output/{llm}/{narrative_pathway.name}_info.json", 'w') as f:
        json.dump(experiment_info, f)

    # Stage 5
    start_time_stage5 = time()
    plandefinition = deterministic_plandefinition_construction(fhir_workflow)
    logger.debug("%s - %s - SHAPE Pipeline stage 5 completed", llm, narrative_pathway.name) 
    end_time_stage5 = time()
    logger.debug("%s - %s - SHAPE Pipeline stage 5 completed in %s seconds", llm, narrative_pathway.name, end_time_stage5 - start_time_stage5)
    with open(f"output/{llm}/{narrative_pathway.name}_plandefinition.json", 'w') as f:
        json.dump(plandefinition.model_dump(), f)
    experiment_info['stage5_execution_time'] = end_time_stage5 - start_time_stage5
    experiment_info['total_execution_time'] = end_time_stage5 - start_time
    with open(f"output/{llm}/{narrative_pathway.name}_info.json", 'w') as f:
        json.dump(experiment_info, f)
    logger.info("%s - %s - SHAPE Pipeline completed in %s seconds", llm, narrative_pathway.name, end_time_stage5 - start_time)


def main():
    narrative_pathways = load_narrative_pathways()
    for llm in llm_models.keys():
        os.makedirs(f"output/{llm}", exist_ok=True)
        for narrative_pathway in narrative_pathways:
            shape_pipeline(narrative_pathway, llm)

if __name__ == "__main__":
    main()
