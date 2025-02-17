from enum import Enum

class ModelType(Enum):
    CLASSIFICATION = "PFM (Classification)"
    REGRESSION = "Observer (Regression)"
    NOT_AVAILABLE = "Not Available"
    
class ProblemType(Enum):
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"
    NOT_AVAILABLE = "Not Available"