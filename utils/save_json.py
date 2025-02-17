import os
import json

def save_json_to_file(data, filename):
    UPLOAD_FOLDER = "data/uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, 'w') as f:
        json.dump(data, f) 
    return file_path