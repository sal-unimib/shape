from src.model.core import Action, Relationship
from typing import Dict, List

class ProceduralWorkflow:
    id : str
    name : str
    title : str
    description : str
    actions : List[Action]
    relationships : List[Relationship]

    def __init__(self, id : str, name : str, title : str, description : str, actions : List[Action], relationships : List[Relationship]):
        self.id = id
        self.name = name
        self.title = title
        self.description = description
        self.actions = actions
        self.relationships = relationships
    
    def to_dict(self) -> Dict:
        return {
            'id' : self.id,
            'name' : self.name,
            'title' : self.title,
            'description' : self.description,
            'actions' : [action.to_dict() for action in self.actions],
            'relationships' : [relationship.to_dict() for relationship in self.relationships]
        }

    @staticmethod
    def from_dict(plan: Dict) -> "ProceduralWorkflow":
        return ProceduralWorkflow(
            id=plan['id'],
            name=plan['name'],
            title=plan['title'],
            description=plan['description'],
            actions=[Action.from_dict(action) for action in plan['actions']],
            relationships=[Relationship.from_dict(rel) for rel in plan['relationships']]
        )

    def __repr__(self):
        return str(self.to_dict())

