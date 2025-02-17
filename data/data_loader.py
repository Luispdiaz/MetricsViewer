import base64
import json
import yaml
from dash import html
import os
import re
from enums.file_names import FileNames
from utils.save_json import save_json_to_file
from utils.cast import bool_to_string
from utils.clean_upload_folder import clean_upload_folder

def parse_files(contents, filenames):
    uploaded_files = []  
    valid_files = []   
    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        if filename.endswith('.json'):
            try:
                json.loads(decoded.decode('utf-8'))  
                uploaded_files.append(html.Li(f'File: {filename}'))
                valid_files.append(filename)
            except json.JSONDecodeError:
                uploaded_files.append(html.Li(f'File: {filename} (invalid JSON)'))
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            try:
                yaml.safe_load(decoded.decode('utf-8')) 
                uploaded_files.append(html.Li(f'File: {filename}'))
                valid_files.append(filename)
            except yaml.YAMLError:
                uploaded_files.append(html.Li(f'File: {filename} (invalid YAML)'))

    files_list = html.Ul(uploaded_files)
    return files_list, valid_files

def extract_model_info(contents, filenames):
    model_info = {}  

    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        if filename == FileNames.PARAMS.value:
            try:
                data = yaml.safe_load(decoded.decode('utf-8'))
                model_info = {
                    'model_name': data.get('model_config', {}).get('name', 'Not Available'),
                    'version': data.get('metadata', {}).get('model_identification', {}).get('version', 'Not Available'),
                    'model_type': data.get('model_config', {}).get('model_type', {}).get('optimization_task', 'Not Available'),
                    'epochs': data.get('training_config', {}).get('epochs', 'Not Available'),
                    'training_duration': data.get('metadata', {}).get('model_training_details', {}).get('training_duration', 'Not Available'),
                    'batch_size': data.get('data_loader_config', {}).get('batch_size',
                                data.get('metadata', {}).get('model_training_details', {}).get('hyperparameters', {}).get('batch_size', 'Not Available')),
                    'framework': data.get('metadata', {}).get('model_identification', {}).get('framework', 'Not Available'),
                    'num_encoder_layers': data.get('model_config', {}).get('num_encoder_layers', 'Not Available'),
                    'num_hidden_neurons': data.get('model_config', {}).get('num_hidden_neurons', 'Not Available'),
                    'num_linear_layers': data.get('model_config', {}).get('num_linear_layers', 'Not Available'),
                    'n_head': data.get('model_config', {}).get('n_head', 'Not Available'),
                    'learning_rate': data.get('training_config', {}).get('lr_after_warmup', 'Not Available'),
                    'num_workers': data.get('data_loader_config', {}).get('num_workers', 'Not Available'),
                    'shuffle': bool_to_string(data.get('data_loader_config', {}).get('shuffle', 'Not Available')),
                    'balance_dataset': bool_to_string(data.get('dataset_config', {}).get('balance_dataset', 'Not Available')),
                    'normalize': bool_to_string(data.get('dataset_config', {}).get('normalize', 'Not Available')),
                    'input_noise': data.get('dataset_config', {}).get('input_noise', 'Not Available'),
                    'gpu': (
                        data.get('metadata', {})
                        .get('model_training_details', {})
                        .get('training_environment', {})
                        .get('gpu_info', [{}])[0]
                        .get('name', 'Not Available')
                    ),
                    'cpus': data.get('metadata', {}).get('model_training_details', {}).get('training_environment', {}).get('cpu_count', 'Not Available'),
                    'cpu_percent': data.get('metadata', {}).get('model_training_details', {}).get('training_environment', {}).get('cpu_percent', 'Not Available'),
                    'ram_available': data.get('metadata', {}).get('model_training_details', {}).get('training_environment', {}).get('ram_available', 'Not Available'),
                    'ram_total': data.get('metadata', {}).get('model_training_details', {}).get('training_environment', {}).get('ram_total', 'Not Available'),
                    'outputs': data.get('model_config', {}).get('outputs', [])
                }
                break  
            except yaml.YAMLError as e:
                print(f"Error processing YAML file: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    return model_info

def extract_model_data_loss_total(contents, filenames):
    file_path = None  

    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        if filename == FileNames.TOTAL_LOSS_HISTORY.value:
            try:
                model_data_loss_total = json.loads(decoded.decode('utf-8'))
                file_path = save_json_to_file(model_data_loss_total, filename)
                break
            except json.JSONDecodeError as e:
                print(f"❌ Error al procesar el archivo JSON: {e}")

    return file_path  

def extract_model_data_loss_output(contents, filenames):
    file_path = None

    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        if filename == FileNames.TOTAL_LOSS_HISTORY_PER_OUTPUT.value:
            try:
                model_data_loss_output = json.loads(decoded.decode('utf-8'))
                file_path = save_json_to_file(model_data_loss_output, filename)
                break
            except json.JSONDecodeError as e:
                print(f"❌ Error al procesar el archivo JSON: {e}")

    return file_path

def extract_model_pet_train(contents, filenames):
    pet_files = {}
    pattern = re.compile(r"TRAIN_SPLIT_LEAK_(.+)_PET\.json", re.IGNORECASE)
    
    for content, filename in zip(contents, filenames):
        match = pattern.fullmatch(filename)
        if match:
            custom_name = match.group(1).upper()
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            try:
                pet_data = json.loads(decoded.decode('utf-8'))
                file_path = save_json_to_file(pet_data, filename)
                pet_files[custom_name] = file_path
            except json.JSONDecodeError as e:
                print(f"❌ Error al procesar el archivo JSON (TRAIN PET): {e}")
    return pet_files  

def extract_model_pet_valid(contents, filenames):
    pet_files = {}
    pattern = re.compile(r"VALID_SPLIT_LEAK_(.+)_PET\.json", re.IGNORECASE)
    
    for content, filename in zip(contents, filenames):
        match = pattern.fullmatch(filename)
        if match:
            custom_name = match.group(1).upper()
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            try:
                pet_data = json.loads(decoded.decode('utf-8'))
                file_path = save_json_to_file(pet_data, filename)
                pet_files[custom_name] = file_path
            except json.JSONDecodeError as e:
                print(f"❌ Error al procesar el archivo JSON (VALID PET): {e}")
    return pet_files

def parse_contents(contents, filenames):
    clean_upload_folder()
    files_list, valid_files = parse_files(contents, filenames)
    model_info = extract_model_info(contents, filenames)
    model_data_loss_total = extract_model_data_loss_total(contents, filenames)
    model_data_loss_output = extract_model_data_loss_output(contents, filenames)
    model_pet_train = extract_model_pet_train(contents, filenames)
    model_pet_valid = extract_model_pet_valid(contents, filenames)
    return (files_list, model_info, model_data_loss_total, 
            model_data_loss_output, model_pet_train, model_pet_valid)
