import base64
import os
import tempfile

def save_decoded_file(content, filename):
    """
    Recibe el contenido (string base64) y el nombre de archivo, y guarda el contenido decodificado en un archivo temporal.
    Devuelve la ruta del archivo temporal o None si el contenido no tiene el formato esperado.
    """
    if ',' not in content:
        print(f"El contenido del archivo {filename} no tiene el formato esperado.")
        return None
    # Dividir solo en la primera coma (para no romper el contenido si tiene comas)
    content_type, content_string = content.split(',', 1)
    try:
        decoded = base64.b64decode(content_string)
    except Exception as e:
        print(f"Error decodificando el archivo {filename}: {e}")
        return None

    temp_file_path = os.path.join(tempfile.gettempdir(), filename)
    try:
        with open(temp_file_path, 'wb') as f:
            f.write(decoded)
    except Exception as e:
        print(f"Error al escribir el archivo temporal {filename}: {e}")
        return None
    return temp_file_path
