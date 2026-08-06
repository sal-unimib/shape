from src.model.core import Action
from src.model.core import Relationship
from src.model.grounded_workflow import GroundedAction
from typing import List, Dict

class ActionParameter:
    path : str
    label : str

    def __init__(self, path : str, label : str):
        self.path = path
        self.label = label

    def to_dict(self):
        return {
            'path' : self.path,
            'label' : self.label
        }

    @staticmethod
    def from_dict(param: Dict) -> "ActionParameter":
        return ActionParameter(
            path=param['path'],
            label=param['label']
        )

    def __repr__(self):
        return str(self.to_dict())

class ParametrizedAction(GroundedAction):
    parameters : List[ActionParameter]

    def __init__(self, action : GroundedAction, parameters : List[ActionParameter]):
        self.id = action.id
        self.label = action.label
        self.temporal_constraint = action.temporal_constraint
        self.condition = action.condition
        self.activity_definition = action.activity_definition
        self.candidates = action.candidates
        self.confidence = action.confidence
        self.reason = action.reason
        self.parameters = parameters

    def to_dict(self):
        base_dict = super().to_dict()
        base_dict['parameters'] = [param.to_dict() for param in self.parameters]
        return base_dict

    @staticmethod
    def from_dict(action: Dict) -> "ParametrizedAction":
        grounded_action = GroundedAction.from_dict(action)
        parameters = [ActionParameter.from_dict(param) for param in action.get('parameters', [])]
        return ParametrizedAction(grounded_action, parameters)

    def __repr__(self):
        return str(self.to_dict())

class ParametrizedWorkflow:
    id : str
    name : str
    title : str
    description : str
    parametrized_actions : List[ParametrizedAction]
    relationships : List[Relationship]

    def __init__(self, id : str, name : str, title : str, description : str, parametrized_actions : List[ParametrizedAction], relationships : List[Relationship]):
        self.id = id
        self.name = name
        self.title = title
        self.description = description
        self.parametrized_actions = parametrized_actions
        self.relationships = relationships
    
    def to_dict(self):
        return {
            'id' : self.id,
            'name' : self.name,
            'title' : self.title,
            'description' : self.description,
            'parametrized_actions' : [action.to_dict() for action in self.parametrized_actions],
            'relationships' : [relationship.to_dict() for relationship in self.relationships]
        }

    @staticmethod
    def from_dict(plan: Dict) -> "ParametrizedWorkflow":
        return ParametrizedWorkflow(
            id=plan['id'],
            name=plan['name'],
            title=plan['title'],
            description=plan['description'],
            parametrized_actions=[ParametrizedAction.from_dict(action) for action in plan['parametrized_actions']],
            relationships=[Relationship.from_dict(rel) for rel in plan['relationships']]
        )

    def __repr__(self):
        return str(self.to_dict())
