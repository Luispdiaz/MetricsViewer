from enum import Enum

class FileNames(Enum):
    PARAMS = 'params.yaml'
    TOTAL_LOSS_HISTORY = 'total_loss_history_per_split.json'
    TOTAL_LOSS_HISTORY_PER_OUTPUT = 'total_loss_history_per_split_per_output.json'