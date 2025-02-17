import os
import json

def load_json_from_file(file_path):
    """Carga los datos de un archivo JSON almacenado en el servidor."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)  
    return {}