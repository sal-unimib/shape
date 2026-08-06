from fhir.resources.activitydefinition import ActivityDefinition
from src.model.core import Action, Relationship
from src.model.enums import LLMConfidence
from typing import Dict, List, Optional

class ActivityCandidate:
    activity_definition : ActivityDefinition
    similarity_score : float
    retrieval_rank : int

    def __init__(self, activity_definition : ActivityDefinition, similarity_score : float, retrieval_rank : int):
        self.activity_definition = activity_definition
        self.similarity_score = similarity_score
        self.retrieval_rank = retrieval_rank

    def to_dict(self):
        return {
            'activity_definition' : self.activity_definition.model_dump(),
            'similarity_score' : self.similarity_score,
            'retrieval_rank' : self.retrieval_rank
        }

    @staticmethod
    def from_dict(candidate: Dict) -> "ActivityCandidate":
        return ActivityCandidate(
            activity_definition=ActivityDefinition.model_validate(candidate['activity_definition']),
            similarity_score=candidate['similarity_score'],
            retrieval_rank=candidate['retrieval_rank']
        )

    def __repr__(self):
        return str(self.to_dict())

    def prompt_string(self) -> str:
        """
        Returns a string representation of the candidate suitable for inclusion in a prompt.
        """
        candidate_str = f"Candidate ID {self.activity_definition.id} : \"{self.activity_definition.title}\"\n"
        candidate_str += f"- Description: {self.activity_definition.description}\n"
        candidate_str += f"- Synonyms: {self.activity_definition.subtitle}\n"
        candidate_str += "- Possible parameters:\n"
        for dv in self.activity_definition.dynamicValue or []:
            candidate_str += f"\t- {dv.expression.description}\n"
        return candidate_str

class GroundedAction(Action):
    activity_definition : Optional[ActivityDefinition]
    candidates : List[ActivityCandidate]
    confidence : LLMConfidence
    reason : Optional[str]

    def __init__(self, action : Action,  candidates : List[ActivityCandidate], confidence : LLMConfidence, activity_definition : Optional[ActivityDefinition] = None, reason : Optional[str] = None):
        self.id = action.id
        self.label = action.label
        self.temporal_constraint = action.temporal_constraint
        self.condition = action.condition
        self.activity_definition = activity_definition
        self.candidates = candidates
        self.confidence = confidence
        self.reason = reason

    def to_dict(self) -> Dict:
        return {
            "id" : self.id,
            "label" : self.label,
            "temporal_constraint" : self.temporal_constraint.to_dict() if self.temporal_constraint else None,
            "condition" : self.condition.to_dict() if self.condition else None,
            'activity_definition' : self.activity_definition.model_dump() if self.activity_definition else None,
            'candidates' : [candidate.to_dict() for candidate in self.candidates],
            'confidence' : self.confidence.value,
            'reason' : self.reason
        }

    @staticmethod
    def from_dict(action: Dict) -> "GroundedAction":
        return GroundedAction(
            action=Action.from_dict(action),
            activity_definition=ActivityDefinition.model_validate(action.get('activity_definition')) if action.get('activity_definition') else None,
            candidates=[ActivityCandidate.from_dict(c) for c in action['candidates']],
            confidence=LLMConfidence(action['confidence']),
            reason=action.get('reason', None)
        )

    def __repr__(self):
        return str(self.to_dict())

class GroundedWorkflow:
    id : str
    name : str
    title : str
    description : str
    grounded_actions : List[GroundedAction]
    relationships : List[Relationship]

    def __init__(self, id : str, name : str, title : str, description : str, grounded_actions : List[GroundedAction], relationships : List[Relationship]):
        self.id = id
        self.name = name
        self.title = title
        self.description = description
        self.grounded_actions = grounded_actions
        self.relationships = relationships
    
    def to_dict(self):
        return {
            'id' : self.id,
            'name' : self.name,
            'title' : self.title,
            'description' : self.description,
            'grounded_actions' : [action.to_dict() for action in self.grounded_actions],
            'relationships' : [relationship.to_dict() for relationship in self.relationships]
        }

    @staticmethod
    def from_dict(plan: Dict) -> "GroundedWorkflow":
        return GroundedWorkflow(
            id=plan['id'],
            name=plan['name'],
            title=plan['title'],
            description=plan['description'],
            grounded_actions=[GroundedAction.from_dict(action) for action in plan['grounded_actions']],
            relationships=[Relationship.from_dict(rel) for rel in plan['relationships']]
        )

    def __repr__(self):
        return str(self.to_dict())