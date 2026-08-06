from enum import Enum

class ShapeStage(str, Enum):
    STAGE1 = "stage1"
    STAGE2 = "stage2"
    STAGE3 = "stage3"
    STAGE4 = "stage4"

class RelationshipType(Enum):
    SEQUENCE = "SEQUENCE"
    PARALLEL = "PARALLEL"
    CONDITIONAL = "CONDITIONAL"

class ConditionComputability(str, Enum):
    COMPUTABLE = "COMPUTABLE"
    SEMICOMPUTABLE = "SEMICOMPUTABLE"
    NONCOMPUTABLE = "NONCOMPUTABLE"
    TODO = "TODO"

class TemporalType(str, Enum):
    REPEAT = "REPEAT"
    OFFSET = "OFFSET"
    DURATION = "DURATION"

class LLMConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"

class ParameterCategory(str, Enum):
    ONTOLOGY = "ontology"
    REFERENCE = "reference"
    VALUE_SET = "value_set"
    RESOURCE = "resource"
    STRING = "string"

class SnomedLabelType(str, Enum):
    MAIN_LABEL = "label"
    PREFERRED_TERM = "preferred"
    ACCEPTABLE_TERM = "acceptable"
