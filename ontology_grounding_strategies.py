from src.model.fhir_workflow import FHIRWorkflow
from src.model.fhir_workflow import FHIRAction
from src.model.enums import ShapeStage
from src.sources.configuration import get_configuration
from typing import Dict
from src.model.enums import SnomedLabelType
from src.model.fhir_workflow import SnomedConcept
from src.model.fhir_workflow import OntologyCandidate
import json
from tqdm import tqdm
from src.sources.llm import embed, llm_models, ollama_source
from src.model.fhir_workflow import OntologyParameter
from src.sources.qdrant import qdrant_source

embed_models = ['sapbert', 'nomic-embed-text']
embed_levels = ['term', 'avg', 'full']

stage_configuration = get_configuration(ShapeStage.STAGE4)
ontology_grounding_reasoning_configuration = stage_configuration.reasoning['ontology_grounding']

def get_embeddings():
    input_file = 'evaluation/rq2/ontology_grounding_strategies/onto_parameters.json'
    with open(input_file) as f:
        parameters_dict = json.load(f)

    total_parameters = sum(
        len(parameters_dict[llm][plan_id][action_id])
        for llm in parameters_dict
        for plan_id in parameters_dict[llm]
        for action_id in parameters_dict[llm][plan_id]
    )

    parameters = {}
    sapbert_embeddings = {}
    nomic_embeddings = {}

    with tqdm(total=total_parameters, desc="Generating embeddings", unit="item") as pbar:
        for llm in parameters_dict:
            parameters[llm] = {}
            sapbert_embeddings[llm] = {}
            nomic_embeddings[llm] = {}
            for plan_id in parameters_dict[llm]:
                parameters[llm][plan_id] = {}
                sapbert_embeddings[llm][plan_id] = {}
                nomic_embeddings[llm][plan_id] = {}
                for action_id in parameters_dict[llm][plan_id]:
                    parameters[llm][plan_id][action_id] = {}
                    sapbert_embeddings[llm][plan_id][action_id] = {}
                    nomic_embeddings[llm][plan_id][action_id] = {}
                    for path in parameters_dict[llm][plan_id][action_id]:
                        pbar.set_postfix_str(f"{llm} | {plan_id}")
                        parameter = OntologyParameter.from_dict(parameters_dict[llm][plan_id][action_id][path])
                        parameters[llm][plan_id][action_id][path] = parameter
                        sapbert_embeddings[llm][plan_id][action_id][path] = {
                            "param_label": parameter.candidates[0].ontology_concept.label,
                            "embedding" : embed("sapbert", [parameter.candidates[0].ontology_concept.label])[0]
                        }
                        nomic_embeddings[llm][plan_id][action_id][path] = {
                            "param_label": parameter.candidates[0].ontology_concept.label,
                            "embedding" : embed("nomic-embed-text", [parameter.candidates[0].ontology_concept.label])[0]
                        }
                        pbar.update(1)

    sapbert_output_file = "evaluation/rq2/ontology_grounding_strategies/sapbert_embeddings.json"
    nomic_output_file = "evaluation/rq2/ontology_grounding_strategies/nomic_embeddings.json"

    with open(sapbert_output_file, 'w') as f:
        json.dump(sapbert_embeddings, f)

    with open(nomic_output_file, 'w') as f:
        json.dump(nomic_embeddings, f)

def get_candidates():
    sapbert_embedding_file = "evaluation/rq2/ontology_grounding_strategies/sapbert_embeddings.json"
    nomic_embedding_file = "evaluation/rq2/ontology_grounding_strategies/nomic_embeddings.json"

    with open(sapbert_embedding_file) as f:
        sapbert_embeddings = json.load(f)

    with open(nomic_embedding_file) as f:
        nomic_embeddings = json.load(f)

    total_sapbert = sum(
        len(sapbert_embeddings[llm][plan_id][action_id])
        for llm in sapbert_embeddings
        for plan_id in sapbert_embeddings[llm]
        for action_id in sapbert_embeddings[llm][plan_id]
    )

    total_nomic = sum(
        len(nomic_embeddings[llm][plan_id][action_id])
        for llm in nomic_embeddings
        for plan_id in nomic_embeddings[llm]
        for action_id in nomic_embeddings[llm][plan_id]
    )

    sapbert_candidates = {}
    with tqdm(total=total_sapbert, desc="SapBERT candidates (Qdrant)", unit="param") as pbar:
        for llm in sapbert_embeddings:
            sapbert_candidates[llm] = {}
            for plan_id in sapbert_embeddings[llm]:
                sapbert_candidates[llm][plan_id] = {}
                for action_id in sapbert_embeddings[llm][plan_id]:
                    sapbert_candidates[llm][plan_id][action_id] = {}
                    for path in sapbert_embeddings[llm][plan_id][action_id]:
                        pbar.set_postfix_str(f"{llm} | {plan_id}")
                        raw_emb = sapbert_embeddings[llm][plan_id][action_id][path]
                        sapbert_emb = raw_emb["embedding"] if isinstance(raw_emb, dict) and "embedding" in raw_emb else raw_emb

                        sapbert_candidates[llm][plan_id][action_id][path] = {
                            'term' : [],
                            'avg' : [],
                            'full' : []
                        }
                        term_candidates = qdrant_source.query_points(
                            collection_name="snomed_sapbert",
                            query=sapbert_emb,
                            limit=50
                        ).points
                        for i, point in enumerate(sorted(term_candidates, key=lambda x: x.score, reverse=True)):
                            sapbert_candidates[llm][plan_id][action_id][path]['term'].append(OntologyCandidate(
                                ontology_concept=SnomedConcept(
                                    concept_id=point.payload['concept_id'],
                                    label=point.payload['term'],
                                    label_type=SnomedLabelType(point.payload['term_type']),
                                    semantic_tag=point.payload['semantic_tag'],
                                    fsn=point.payload['fsn'],
                                    preferred_terms=point.payload['preferred_terms'],
                                    acceptable_terms=point.payload['acceptable_terms']
                                ),
                                similarity_score=point.score,
                                retrieval_rank=i+1
                            ))
                        avg_candidates = qdrant_source.query_points(
                            collection_name="snomed_sapbert_avg",
                            query=sapbert_emb,
                            limit=50
                        ).points
                        for i, point in enumerate(sorted(avg_candidates, key=lambda x: x.score, reverse=True)):
                            sapbert_candidates[llm][plan_id][action_id][path]['avg'].append(OntologyCandidate(
                                ontology_concept=SnomedConcept(
                                    concept_id=point.payload['conceptId'],
                                    label=point.payload['label'],
                                    label_type=SnomedLabelType.MAIN_LABEL,
                                    semantic_tag=point.payload['semantic_tag'],
                                    fsn=point.payload['fsn'],
                                    preferred_terms=point.payload['preferred_terms'],
                                    acceptable_terms=point.payload['acceptable_terms']
                                ),
                                similarity_score=point.score,
                                retrieval_rank=i+1
                            ))
                        full_candidates = qdrant_source.query_points(
                            collection_name="snomed_sapbert_full",
                            query=sapbert_emb,
                            limit=50
                        ).points
                        for i, point in enumerate(sorted(full_candidates, key=lambda x: x.score, reverse=True)):
                            sapbert_candidates[llm][plan_id][action_id][path]['full'].append(OntologyCandidate(
                                ontology_concept=SnomedConcept(
                                    concept_id=point.payload['conceptId'],
                                    label=point.payload['label'],
                                    label_type=SnomedLabelType.MAIN_LABEL,
                                    semantic_tag=point.payload['semantic_tag'],
                                    fsn=point.payload['fsn'],
                                    preferred_terms=point.payload['preferred_terms'],
                                    acceptable_terms=point.payload['acceptable_terms']
                                ),
                                similarity_score=point.score,
                                retrieval_rank=i+1
                            ))
                        pbar.update(1)
        
    candidates_output_file = "evaluation/rq2/ontology_grounding_strategies/sapbert_candidates.json"
    with open(candidates_output_file, 'w') as f:
        json.dump(sapbert_candidates, f, default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o))

    nomic_candidates = {}
    with tqdm(total=total_nomic, desc="Nomic candidates (Qdrant)", unit="param") as pbar:
        for llm in nomic_embeddings:
            nomic_candidates[llm] = {}
            for plan_id in nomic_embeddings[llm]:
                nomic_candidates[llm][plan_id] = {}
                for action_id in nomic_embeddings[llm][plan_id]:
                    nomic_candidates[llm][plan_id][action_id] = {}
                    for path in nomic_embeddings[llm][plan_id][action_id]:
                        pbar.set_postfix_str(f"{llm} | {plan_id}")
                        raw_emb = nomic_embeddings[llm][plan_id][action_id][path]
                        nomic_emb = raw_emb["embedding"] if isinstance(raw_emb, dict) and "embedding" in raw_emb else raw_emb

                        nomic_candidates[llm][plan_id][action_id][path] = {
                            'term' : [],
                            'avg' : [],
                            'full' : []
                        }
                        term_candidates = qdrant_source.query_points(
                            collection_name="snomed_nomic",
                            query=nomic_emb,
                            limit=50
                        ).points
                        for i, point in enumerate(sorted(term_candidates, key=lambda x: x.score, reverse=True)):
                            full_concept = qdrant_source.retrieve("snomed_ct_new", [int(point.payload['concept_id'])])[0].payload
                            nomic_candidates[llm][plan_id][action_id][path]['term'].append(OntologyCandidate(
                                ontology_concept=SnomedConcept(
                                    concept_id=point.payload['concept_id'],
                                    label=point.payload['term'],
                                    label_type=SnomedLabelType(point.payload['term_type']),
                                    semantic_tag=point.payload['semantic_tag'],
                                    fsn=full_concept['fsn'],
                                    preferred_terms=full_concept['preferred_terms'],
                                    acceptable_terms=full_concept['acceptable_terms']
                                ),
                                similarity_score=point.score,
                                retrieval_rank=i+1
                            ))
                        avg_candidates = qdrant_source.query_points(
                            collection_name="snomed_avg",
                            query=nomic_emb,
                            limit=50
                        ).points
                        for i, point in enumerate(sorted(avg_candidates, key=lambda x: x.score, reverse=True)):
                            nomic_candidates[llm][plan_id][action_id][path]['avg'].append(OntologyCandidate(
                                ontology_concept=SnomedConcept(
                                    concept_id=point.payload['conceptId'],
                                    label=point.payload['label'],
                                    label_type=SnomedLabelType.MAIN_LABEL,
                                    semantic_tag=point.payload['semantic_tag'],
                                    fsn=point.payload['fsn'],
                                    preferred_terms=point.payload['preferred_terms'],
                                    acceptable_terms=point.payload['acceptable_terms']
                                ),
                                similarity_score=point.score,
                                retrieval_rank=i+1
                            ))
                        full_candidates = qdrant_source.query_points(
                            collection_name="snomed_ct_new",
                            query=nomic_emb,
                            limit=50
                        ).points
                        for i, point in enumerate(sorted(full_candidates, key=lambda x: x.score, reverse=True)):
                            nomic_candidates[llm][plan_id][action_id][path]['full'].append(OntologyCandidate(
                                ontology_concept=SnomedConcept(
                                    concept_id=point.payload['conceptId'],
                                    label=point.payload['label'],
                                    label_type=SnomedLabelType.MAIN_LABEL,
                                    semantic_tag=point.payload['semantic_tag'],
                                    fsn=point.payload['fsn'],
                                    preferred_terms=point.payload['preferred_terms'],
                                    acceptable_terms=point.payload['acceptable_terms']
                                ),
                                similarity_score=point.score,
                                retrieval_rank=i+1
                            ))
                        pbar.update(1)
        
    candidates_output_file = "evaluation/rq2/ontology_grounding_strategies/nomic_candidates.json"
    with open(candidates_output_file, 'w') as f:
        json.dump(nomic_candidates, f, default=lambda o: o.to_dict() if hasattr(o, 'to_dict') else str(o))
                    
def get_parametrized_workflow(llm: str, plan_id: str) -> Dict[str, FHIRAction]:
    path = f"output/{llm}/{plan_id}_output.json"
    with open(path) as f:
        plan = json.load(f)
    pw = FHIRWorkflow.from_dict(plan)

    fhir_actions = {action.id: action for action in pw.fhir_actions}
    return fhir_actions

def parameter_selection():
    sapbert_candidates_file = "evaluation/rq2/ontology_grounding_strategies/sapbert_candidates.json"
    with open(sapbert_candidates_file) as f:
        sapbert_candidates = json.load(f)
    
    total_sapbert_evals = sum(
        len(sapbert_candidates[llm][plan_id][action_id][path])
        for llm in sapbert_candidates
        for plan_id in sapbert_candidates[llm]
        for action_id in sapbert_candidates[llm][plan_id]
        for path in sapbert_candidates[llm][plan_id][action_id]
    )

    sapbert_parameter_resolution = {}
    with tqdm(total=total_sapbert_evals, desc="SapBERT parameter resolution (LLM)", unit="cfg") as pbar:
        for llm in sapbert_candidates:
            sapbert_parameter_resolution[llm] = {}
            for plan_id in sapbert_candidates[llm]:
                sapbert_parameter_resolution[llm][plan_id] = {}
                pw = get_parametrized_workflow(llm, plan_id)
                for action_id in sapbert_candidates[llm][plan_id]:
                    pw_action = pw[action_id]
                    sapbert_parameter_resolution[llm][plan_id][action_id] = {}
                    for path in sapbert_candidates[llm][plan_id][action_id]:
                        pw_parameter = next(p for p in pw_action.parameters if p.path == path)
                        param_role = next((v.expression.description for v in pw_action.activity_definition.dynamicValue if v.path == path), None)
                        sapbert_parameter_resolution[llm][plan_id][action_id][path] = {}
                        for emb_level in sapbert_candidates[llm][plan_id][action_id][path]:
                            pbar.set_postfix_str(f"{llm} | {plan_id} | {emb_level}")
                            sapbert_parameter_resolution[llm][plan_id][action_id][path][emb_level] = {}
                            candidates_concepts = []
                            for c in sapbert_candidates[llm][plan_id][action_id][path][emb_level]:
                                candidates_concepts.append(OntologyCandidate.from_dict(c) if isinstance(c, dict) else c)
                            
                            k50_candidate_concepts_str = ""
                            k20_candidate_concepts_str = ""
                            k10_candidate_concepts_str = ""
                            for cand_conc in candidates_concepts:
                                k50_candidate_concepts_str += f"ID: {cand_conc.ontology_concept.concept_id} - label: {cand_conc.ontology_concept.label}:\n"
                                k50_candidate_concepts_str += f"\t- semantic_tag: {cand_conc.ontology_concept.semantic_tag}\n"
                                pt_str = ", ".join([pt for pt in cand_conc.ontology_concept.preferred_terms[:3] or []])
                                k50_candidate_concepts_str += f"\t- preferred_terms: {pt_str}\n"
                            for cand_conc in candidates_concepts[:20]:
                                k20_candidate_concepts_str += f"ID: {cand_conc.ontology_concept.concept_id} - label: {cand_conc.ontology_concept.label}:\n"
                                k20_candidate_concepts_str += f"\t- semantic_tag: {cand_conc.ontology_concept.semantic_tag}\n"
                                pt_str = ", ".join([pt for pt in cand_conc.ontology_concept.preferred_terms[:3] or []])
                                k20_candidate_concepts_str += f"\t- preferred_terms: {pt_str}\n"
                            for cand_conc in candidates_concepts[:10]:
                                k10_candidate_concepts_str += f"ID: {cand_conc.ontology_concept.concept_id} - label: {cand_conc.ontology_concept.label}:\n"
                                k10_candidate_concepts_str += f"\t- semantic_tag: {cand_conc.ontology_concept.semantic_tag}\n"
                                pt_str = ", ".join([pt for pt in cand_conc.ontology_concept.preferred_terms[:3] or []])
                                k10_candidate_concepts_str += f"\t- preferred_terms: {pt_str}\n"
                            
                            prompt = (ontology_grounding_reasoning_configuration.prompt.replace("{{label}}", pw_parameter.label)
                                .replace("{{parameter_role}}", param_role or "")
                                .replace("{{parameter_description}}", pw_action.label)
                                .replace("{{snomed_concepts}}", k50_candidate_concepts_str))
                            llm_response = ollama_source.generate(
                                model = llm_models[llm],
                                format='json',
                                stream=False,
                                prompt=prompt,
                                options=ontology_grounding_reasoning_configuration.hyperparameters
                            )
                            sapbert_parameter_resolution[llm][plan_id][action_id][path][emb_level]['k50'] = json.loads(llm_response.response)
                            
                            prompt = (ontology_grounding_reasoning_configuration.prompt.replace("{{label}}", pw_parameter.label)
                                .replace("{{parameter_role}}", param_role or "")
                                .replace("{{parameter_description}}", pw_action.label)
                                .replace("{{snomed_concepts}}", k20_candidate_concepts_str))
                            llm_response = ollama_source.generate(
                                model = llm_models[llm],
                                format='json',
                                stream=False,
                                prompt=prompt,
                                options=ontology_grounding_reasoning_configuration.hyperparameters
                            )
                            sapbert_parameter_resolution[llm][plan_id][action_id][path][emb_level]['k20'] = json.loads(llm_response.response)

                            prompt = (ontology_grounding_reasoning_configuration.prompt.replace("{{label}}", pw_parameter.label)
                                .replace("{{parameter_role}}", param_role or "")
                                .replace("{{parameter_description}}", pw_action.label)
                                .replace("{{snomed_concepts}}", k10_candidate_concepts_str))
                            llm_response = ollama_source.generate(
                                model = llm_models[llm],
                                format='json',
                                stream=False,
                                prompt=prompt,
                                options=ontology_grounding_reasoning_configuration.hyperparameters
                            )
                            sapbert_parameter_resolution[llm][plan_id][action_id][path][emb_level]['k10'] = json.loads(llm_response.response)
                            pbar.update(1)

    sapbert_resolution_file = "evaluation/rq2/ontology_grounding_strategies/sapbert_parameter_resolution.json"
    with open(sapbert_resolution_file, 'w') as f:
        json.dump(sapbert_parameter_resolution, f)
    
    nomic_candidates_file = "evaluation/rq2/ontology_grounding_strategies/nomic_candidates.json"
    with open(nomic_candidates_file) as f:
        nomic_candidates = json.load(f)
    
    total_nomic_evals = sum(
        len(nomic_candidates[llm][plan_id][action_id][path])
        for llm in nomic_candidates
        for plan_id in nomic_candidates[llm]
        for action_id in nomic_candidates[llm][plan_id]
        for path in nomic_candidates[llm][plan_id][action_id]
    )

    nomic_parameter_resolution = {}
    with tqdm(total=total_nomic_evals, desc="Nomic parameter resolution (LLM)", unit="cfg") as pbar:
        for llm in nomic_candidates:
            nomic_parameter_resolution[llm] = {}
            for plan_id in nomic_candidates[llm]:
                nomic_parameter_resolution[llm][plan_id] = {}
                pw = get_parametrized_workflow(llm, plan_id)
                for action_id in nomic_candidates[llm][plan_id]:
                    pw_action = pw[action_id]
                    nomic_parameter_resolution[llm][plan_id][action_id] = {}
                    for path in nomic_candidates[llm][plan_id][action_id]:
                        pw_parameter = next(p for p in pw_action.parameters if p.path == path)
                        param_role = next((v.expression.description for v in pw_action.activity_definition.dynamicValue if v.path == path), None)
                        nomic_parameter_resolution[llm][plan_id][action_id][path] = {}
                        for emb_level in nomic_candidates[llm][plan_id][action_id][path]:
                            pbar.set_postfix_str(f"{llm} | {plan_id} | {emb_level}")
                            nomic_parameter_resolution[llm][plan_id][action_id][path][emb_level] = {}
                            candidates_concepts = []
                            for c in nomic_candidates[llm][plan_id][action_id][path][emb_level]:
                                candidates_concepts.append(OntologyCandidate.from_dict(c) if isinstance(c, dict) else c)
                            
                            k50_candidate_concepts_str = ""
                            k20_candidate_concepts_str = ""
                            k10_candidate_concepts_str = ""
                            for cand_conc in candidates_concepts:
                                k50_candidate_concepts_str += f"ID: {cand_conc.ontology_concept.concept_id} - label: {cand_conc.ontology_concept.label}:\n"
                                k50_candidate_concepts_str += f"\t- semantic_tag: {cand_conc.ontology_concept.semantic_tag}\n"
                                pt_str = ", ".join([pt for pt in cand_conc.ontology_concept.preferred_terms[:3] or []])
                                k50_candidate_concepts_str += f"\t- preferred_terms: {pt_str}\n"
                            for cand_conc in candidates_concepts[:20]:
                                k20_candidate_concepts_str += f"ID: {cand_conc.ontology_concept.concept_id} - label: {cand_conc.ontology_concept.label}:\n"
                                k20_candidate_concepts_str += f"\t- semantic_tag: {cand_conc.ontology_concept.semantic_tag}\n"
                                pt_str = ", ".join([pt for pt in cand_conc.ontology_concept.preferred_terms[:3] or []])
                                k20_candidate_concepts_str += f"\t- preferred_terms: {pt_str}\n"
                            for cand_conc in candidates_concepts[:10]:
                                k10_candidate_concepts_str += f"ID: {cand_conc.ontology_concept.concept_id} - label: {cand_conc.ontology_concept.label}:\n"
                                k10_candidate_concepts_str += f"\t- semantic_tag: {cand_conc.ontology_concept.semantic_tag}\n"
                                pt_str = ", ".join([pt for pt in cand_conc.ontology_concept.preferred_terms[:3] or []])
                                k10_candidate_concepts_str += f"\t- preferred_terms: {pt_str}\n"
                            
                            prompt = (ontology_grounding_reasoning_configuration.prompt.replace("{{label}}", pw_parameter.label)
                                .replace("{{parameter_role}}", param_role or "")
                                .replace("{{parameter_description}}", pw_action.label)
                                .replace("{{snomed_concepts}}", k50_candidate_concepts_str))
                            llm_response = ollama_source.generate(
                                model = llm_models[llm],
                                format='json',
                                stream=False,
                                prompt=prompt,
                                options=ontology_grounding_reasoning_configuration.hyperparameters
                            )
                            nomic_parameter_resolution[llm][plan_id][action_id][path][emb_level]['k50'] = json.loads(llm_response.response)
                            
                            prompt = (ontology_grounding_reasoning_configuration.prompt.replace("{{label}}", pw_parameter.label)
                                .replace("{{parameter_role}}", param_role or "")
                                .replace("{{parameter_description}}", pw_action.label)
                                .replace("{{snomed_concepts}}", k20_candidate_concepts_str))
                            llm_response = ollama_source.generate(
                                model = llm_models[llm],
                                format='json',
                                stream=False,
                                prompt=prompt,
                                options=ontology_grounding_reasoning_configuration.hyperparameters
                            )
                            nomic_parameter_resolution[llm][plan_id][action_id][path][emb_level]['k20'] = json.loads(llm_response.response)

                            prompt = (ontology_grounding_reasoning_configuration.prompt.replace("{{label}}", pw_parameter.label)
                                .replace("{{parameter_role}}", param_role or "")
                                .replace("{{parameter_description}}", pw_action.label)
                                .replace("{{snomed_concepts}}", k10_candidate_concepts_str))
                            llm_response = ollama_source.generate(
                                model = llm_models[llm],
                                format='json',
                                stream=False,
                                prompt=prompt,
                                options=ontology_grounding_reasoning_configuration.hyperparameters
                            )
                            nomic_parameter_resolution[llm][plan_id][action_id][path][emb_level]['k10'] = json.loads(llm_response.response)
                            pbar.update(1)

    nomic_resolution_file = "evaluation/rq2/ontology_grounding_strategies/nomic_parameter_resolution.json"
    with open(nomic_resolution_file, 'w') as f:
        json.dump(nomic_parameter_resolution, f)

def main():
    # get_embeddings()
    # get_candidates()
    parameter_selection()

if __name__ == "__main__":
    main()
