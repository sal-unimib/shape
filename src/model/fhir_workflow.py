from src.model.core import Relationship
from src.model.enums import LLMConfidence, ParameterCategory, SnomedLabelType, TemporalType
from src.model.parametrized_workflow import ParametrizedAction
from typing import Any, Dict, List, Optional

class FHIRParameter:
    path : str
    category : ParameterCategory
    label: str
    value: Any

    def __init__(self, path : str, category : ParameterCategory, label: str, value: Any):
        self.path = path
        self.category = category
        self.label = label
        self.value = value
    
    def to_dict(self):
        return {
            'path' : self.path,
            'category' : self.category.value,
            'label' : self.label,
            'value' : self.value
        }
    
    @staticmethod
    def from_dict(param: Dict) -> "FHIRParameter":
        return FHIRParameter(
            path=param['path'],
            category=ParameterCategory(param['category']),
            label=param['label'],
            value=param['value']
        )
    
    def __repr__(self):
        return str(self.to_dict())

class SnomedConcept:
    concept_id : str
    label : str
    label_type : SnomedLabelType
    semantic_tag : str
    fsn : str
    preferred_terms : List[str]
    acceptable_terms : List[str]

    def __init__(self, concept_id : str, label : str, label_type : SnomedLabelType, semantic_tag : str, fsn : str, preferred_terms : List[str] = [], acceptable_terms : List[str] = []):
        self.concept_id = concept_id
        self.label = label
        self.label_type = label_type
        self.semantic_tag = semantic_tag
        self.fsn = fsn
        self.preferred_terms = preferred_terms
        self.acceptable_terms = acceptable_terms

    def to_dict(self):
        return {
            'concept_id' : self.concept_id,
            'label' : self.label,
            'label_type' : self.label_type,
            'semantic_tag' : self.semantic_tag,
            'fsn' : self.fsn,
            'preferred_terms' : self.preferred_terms,
            'acceptable_terms' : self.acceptable_terms
        }

    @staticmethod
    def from_dict(concept: Dict) -> "SnomedConcept":
        return SnomedConcept(
            concept_id = concept["concept_id"],
            label = concept["label"],
            label_type = concept["label_type"],
            semantic_tag = concept["semantic_tag"],
            fsn = concept["fsn"],
            preferred_terms = concept.get("preferred_terms", []),
            acceptable_terms = concept.get("acceptable_terms", []),
        )

    def __repr__(self):
        return str(self.to_dict())

class OntologyCandidate:
    ontology_concept : SnomedConcept
    similarity_score : float
    retrieval_rank : int

    def __init__(self, ontology_concept : SnomedConcept, similarity_score : float, retrieval_rank : int):
        self.ontology_concept = ontology_concept
        self.similarity_score = similarity_score
        self.retrieval_rank = retrieval_rank

    def to_dict(self):
        return {
            'ontology_concept' : self.ontology_concept.to_dict(),
            'similarity_score' : self.similarity_score,
            'retrieval_rank' : self.retrieval_rank
        }

    @staticmethod
    def from_dict(candidate: Dict) -> "OntologyCandidate":
        return OntologyCandidate(
            ontology_concept=SnomedConcept.from_dict(candidate["ontology_concept"]),
            similarity_score=candidate["similarity_score"],
            retrieval_rank=candidate["retrieval_rank"]
        )

    def __repr__(self):
        return str(self.to_dict())

class OntologyParameter(FHIRParameter):
    candidates : List[OntologyCandidate]
    confidence : LLMConfidence
    reason : Optional[str]

    def __init__(self, path : str, label: str, value: Any, candidates : List[OntologyCandidate], confidence : LLMConfidence, reason : Optional[str] = None):
        super().__init__(path, ParameterCategory.ONTOLOGY, label, value)
        self.candidates = candidates
        self.confidence = confidence
        self.reason = reason

    def to_dict(self):
        return {
            'path' : self.path,
            'category' : self.category.value,
            'label' : self.label,
            'value' : self.value,
            'candidates' : [candidate.to_dict() for candidate in self.candidates],
            'confidence' : self.confidence.value,
            'reason' : self.reason
        }
    
    @staticmethod
    def from_dict(param: Dict) -> "OntologyParameter":
        return OntologyParameter(
            path=param['path'],
            label=param['label'],
            value=param['value'],
            candidates=[OntologyCandidate.from_dict(candidate) for candidate in param.get('candidates', [])],
            confidence=LLMConfidence(param['confidence']),
            reason=param.get('reason')
        )

    def __repr__(self):
        return str(self.to_dict())

class FHIRTemporalConstraint:
    type : TemporalType
    value: Optional[Dict]

    def __init__(self, type: TemporalType, value: Optional[Dict] = None):
        self.type = type
        self.value = value

    def to_dict(self):
        return {
            'type' : self.type.value,
            'value' : self.value
        }

    @staticmethod
    def from_dict(tc: Dict) -> "FHIRTemporalConstraint":
        return FHIRTemporalConstraint(
            type = TemporalType(tc["type"]),
            value = tc.get("value")
        )

    def __repr__(self):
        return str(self.to_dict())

class FHIRAction(ParametrizedAction):
    fhir_parameters : List[FHIRParameter]
    fhir_temporal_constraint : Optional[FHIRTemporalConstraint]

    def __init__(self, action: ParametrizedAction, fhir_parameters: List[FHIRParameter], fhir_temporal_constraint: Optional[FHIRTemporalConstraint] = None):
        self.id = action.id
        self.label = action.label
        self.temporal_constraint = action.temporal_constraint
        self.condition = action.condition
        self.activity_definition = action.activity_definition
        self.candidates = action.candidates
        self.confidence = action.confidence
        self.reason = action.reason
        self.parameters = action.parameters
        self.fhir_parameters = fhir_parameters
        self.fhir_temporal_constraint = fhir_temporal_constraint
    
    def to_dict(self):
        base_dict = super().to_dict()
        base_dict['fhir_parameters'] = [param.to_dict() for param in self.fhir_parameters]
        base_dict['fhir_temporal_constraint'] = self.fhir_temporal_constraint.to_dict() if self.fhir_temporal_constraint else None
        return base_dict
    
    @staticmethod
    def from_dict(action: Dict) -> "FHIRAction":
        fhir_params = []
        for param in action.get('fhir_parameters', []):
            if param.get('category') == ParameterCategory.ONTOLOGY.value:
                fhir_param = OntologyParameter.from_dict(param)
            else:
                fhir_param = FHIRParameter.from_dict(param)
            fhir_params.append(fhir_param)
        return FHIRAction(
            action = ParametrizedAction.from_dict(action),
            fhir_parameters = fhir_params,
            fhir_temporal_constraint = FHIRTemporalConstraint.from_dict(action.get('fhir_temporal_constraint')) if action.get('fhir_temporal_constraint') else None
        )

    def __repr__(self):
        return str(self.to_dict())

class FHIRWorkflow:
    id : str
    name : str
    title : str
    description : str
    fhir_actions : List[FHIRAction]
    relationships : List[Relationship]

    def __init__(self, id : str, name : str, title : str, description : str, fhir_actions : List[FHIRAction], relationships : List[Relationship]):
        self.id = id
        self.name = name
        self.title = title
        self.description = description
        self.fhir_actions = fhir_actions
        self.relationships = relationships

    def to_dict(self):
        return {
            'id' : self.id,
            'name' : self.name,
            'title' : self.title,
            'description' : self.description,
            'fhir_actions' : [action.to_dict() for action in self.fhir_actions],
            'relationships' : [relationship.to_dict() for relationship in self.relationships]
        }

    @staticmethod
    def from_dict(plan: Dict) -> "FHIRWorkflow":
        return FHIRWorkflow(
            id=plan['id'],
            name=plan['name'],
            title=plan['title'],
            description=plan['description'],
            fhir_actions=[FHIRAction.from_dict(action) for action in plan['fhir_actions']],
            relationships=[Relationship.from_dict(rel) for rel in plan['relationships']]
        )

    def __repr__(self):
        return str(self.to_dict())
