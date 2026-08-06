from src.model.enums import TemporalType, ConditionComputability
from typing import Dict, Optional

class TemporalConstraint:
    type : TemporalType
    label : str

    def __init__(self, type : str, label : str):
        self.type = TemporalType(type)
        self.label = label
    
    def to_dict(self):
        return {
            "type" : self.type.value,
            "label" : self.label
        }
    
    @staticmethod
    def from_dict(tc: dict) -> "TemporalConstraint":
        return TemporalConstraint(
            type = TemporalType(tc['type']),
            label = tc['label']
        )

    def __repr__(self):
        return str(self.to_dict())

class ActionCondition:
    label: str
    computability : ConditionComputability

    def __init__(self, label: str, computability: str):
        self.label = label
        self.computability = ConditionComputability(computability)
    
    def to_dict(self):
        return {
            "label" : self.label,
            "computability" : self.computability.value
        }
    
    @staticmethod
    def from_dict(cond : Dict) -> "ActionCondition":
        return ActionCondition(
            label = cond['label'],
            computability = ConditionComputability(cond['computability'])
        )
    
    def __repr__(self):
        return str(self.to_dict())

class Action:
    id : str
    label : str
    temporal_constraint : Optional[TemporalConstraint]
    condition : Optional[ActionCondition]

    def __init__(self, id : str, label : str, temporal_constraint : Optional[TemporalConstraint] = None, condition : Optional[ActionCondition] = None):
        self.id = id
        self.label = label
        self.temporal_constraint = temporal_constraint
        self.condition = condition
    
    def to_dict(self):
        return {
            "id" : self.id,
            "label" : self.label,
            "temporal_constraint" : self.temporal_constraint.to_dict() if self.temporal_constraint else None,
            "condition" : self.condition.to_dict() if self.condition else None
        }

    @staticmethod
    def from_dict(action: Dict) -> "Action":
        tc_dict = action.get('temporal_constraint', None)
        cond_dict = action.get('condition', None)
        return Action(
            id = action['id'],
            label = action['label'],
            temporal_constraint = TemporalConstraint.from_dict(tc_dict) if tc_dict else None,
            condition = ActionCondition.from_dict(cond_dict) if cond_dict else None
        )

    def __repr__(self):
        return str(self.to_dict())

class Relationship:
    from_action : str
    to_action : str

    def __init__(self, from_action : str, to_action : str):
        self.from_action = from_action
        self.to_action = to_action
    
    def to_dict(self):
        return {
            "from_action" : self.from_action,
            "to_action" : self.to_action,
        }
    
    @staticmethod
    def from_dict(rel : Dict) -> "Relationship":
        return Relationship(
            from_action = rel['from_action'],
            to_action = rel['to_action']
        )

    def __repr__(self):
        return str(self.to_dict())
