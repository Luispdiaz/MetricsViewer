import os

def clean_upload_folder():
    UPLOAD_FOLDER = "data/uploads"
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)