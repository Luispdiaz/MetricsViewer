from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
from enums.model_types import ModelType
from dash import html, Output, Input
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import json
from utils.format_bytes_gb import format_bytes_to_gb
from utils.load_json import load_json_from_file
from graphs.loss_graphs import generate_class_loss_graph, generate_class_total_loss_graph
from graphs.residual import create_residuals_graph
from graphs.metrics import compute_metric_evolution_classification, compute_metric_evolution_regression, create_metric_figure
from graphs.confusion import create_flat_3d_heatmap
from enums.model_types import ProblemType

with open("data/constants/all_cards.json", "r", encoding="utf-8") as file:
    all_cards = json.load(file)
 

default_cards = ['Model Name', 'Model Version', 'Model Type', 'Number of Epochs', 'Training Time'] 

def register_dashboard_callbacks(app):
    @app.callback(
        Output('dynamic-card-row', 'children'),
        [Input('stored-file-info', 'data'),
        Input('available-cards-checklist-1', 'value'),
        Input('available-cards-checklist-2', 'value')],
        prevent_initial_call=False
    )
    def update_dashboard(model_info, selected_cards_1, selected_cards_2):

        current_cards = []

        if model_info:
            model_data = {
                'Model Name': model_info.get('model_name', 'No model name available').replace('_', ' ').title(),
                'Model Version': model_info.get('version', 'Not Available'),
                'Model Type': (
                    ModelType.CLASSIFICATION.value if model_info.get('model_type', 'Not Available') == 'CLASSIFICATION' 
                    else ModelType.REGRESSION.value if model_info.get('model_type') == 'REGRESSION' 
                    else ModelType.NOT_AVAILABLE.value
                ),
                'Number of Epochs': model_info.get('epochs', 'Not Available'),
                'Training Time': model_info.get('training_duration', 'Not Available'),
                'Batch Size': model_info.get('batch_size', 'Not Available'),
                'Framework': model_info.get('framework', 'Not Available'),
                'Encoder Layers': model_info.get('num_encoder_layers', 'Not Available'),
                'Hidden Neurons': model_info.get('num_hidden_neurons', 'Not Available'),
                'Linear Layers': model_info.get('num_linear_layers', 'Not Available'),
                'Attention Heads': model_info.get('n_head', 'Not Available'),
                'Final Learning Rate': model_info.get('learning_rate', 'Not Available'),
                'Number of Workers': model_info.get('num_workers', 'Not Available'),
                'Shuffle': model_info.get('shuffle', 'Not Available'),
                'Dataset Balance': model_info.get('balance_dataset', 'Not Available'),
                'Normalization': model_info.get('normalize', 'Not Available'),
                'Input Noise': model_info.get('input_noise', 'Not Available'),
                'GPU': model_info.get('gpu', 'Not Available'),
                'CPUs': model_info.get('cpus', 'Not Available'),
                'RAM': f"{format_bytes_to_gb(model_info.get('ram_available', 'Not Available'))} available of {format_bytes_to_gb(model_info.get('ram_total', 'Not Available'))}",  # Corregido
            }

            selected_cards = (selected_cards_1 or []) + (selected_cards_2 or []) 

            for card_title in selected_cards:
                if card_title in model_data:
                    content = model_data[card_title] if model_data[card_title] != '' else 'Not Available'
                    
                    new_card = dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.H4(card_title, className="card-title"),
                            html.P(content, className="card-text"),
                        ])
                    ]), width=4, style={'marginBottom': '20px'}) 
                    current_cards.append(new_card)

        if not current_cards:
            current_cards.append(dbc.Col(html.Div("No data available."), width=12))

        return current_cards

    

    @app.callback(
        Output("modal", "is_open"),
        [Input("add-card-button", "n_clicks"), Input("ready-button", "n_clicks")],
        [State("modal", "is_open")]
    )
    def toggle_modal(n1, n2, is_open):
        if n1: 
            return not is_open
        elif n2:  
            return False
        return is_open
    
    @app.callback(
        Output('train-class-loss-graph', 'figure'),
        [Input('stored-loss-data-total', 'data')],
        [State('stored-pet-train', 'data')]
    )
    def update_total_loss_graph(loss_file_path, pet_train_data):
        
        if not loss_file_path:
            print("No hay archivo de pérdida total")
            return go.Figure()

        loss_data = load_json_from_file(loss_file_path)
        
        n_epochs = None
        if pet_train_data and isinstance(pet_train_data, dict) and pet_train_data:
            last_key = sorted(pet_train_data.keys())[-1]  
            pet_file_path = pet_train_data[last_key]
            pet_data = load_json_from_file(pet_file_path)
            if pet_data:
                history = pet_data.get("history", {})
                if history:
                    try:
                        epoch_keys = [int(k) for k in history.keys()]
                        n_epochs = max(epoch_keys) + 1 
                    except Exception as e:
                        print(f"Error al convertir claves de epoch: {e}")
            else:
                print(f"No hay datos en el PET de entrenamiento para la clave {last_key}")
        else:
            print("No hay PET de entrenamiento disponible")

        if n_epochs and n_epochs > 0:
            total_train = len(loss_data["TRAIN_SPLIT"])
            total_valid = len(loss_data["VALID_SPLIT"])
            block_size_train = total_train // n_epochs  
            block_size_valid = total_valid // n_epochs
            new_train = []
            new_valid = []
            for i in range(0, total_train, block_size_train):
                block_train = loss_data["TRAIN_SPLIT"][i: i + block_size_train]
                if block_train:
                    avg_train = sum(block_train) / len(block_train)
                    new_train.append(avg_train)
            for i in range(0, total_valid, block_size_valid):
                block_valid = loss_data["VALID_SPLIT"][i: i + block_size_valid]
                if block_valid:
                    avg_valid = sum(block_valid) / len(block_valid)
                    new_valid.append(avg_valid)
            processed_loss_data = {"TRAIN_SPLIT": new_train, "VALID_SPLIT": new_valid}
        else:
            processed_loss_data = loss_data

        
        return generate_class_total_loss_graph('Total Loss', processed_loss_data)

    @app.callback(
        [Output('output-class-dropdown', 'options'),
        Output('output-class-dropdown', 'value')],
        Input('stored-loss-data-output', 'data')
    )
    def update_output_dropdown(loss_file_path):
        if not loss_file_path:
            return [], None

        loss_data = load_json_from_file(loss_file_path)

        if "TRAIN_SPLIT" in loss_data:
            output_keys = list(loss_data["TRAIN_SPLIT"].keys())
            output_options = [{"label": key.replace("_", " ").title(), "value": key} for key in output_keys]
            default_value = output_keys[0] if output_keys else None
        else:
            output_options, default_value = [], None

        return output_options, default_value


    @app.callback(
        Output('class-loss-graph', 'figure'),
        [Input('output-class-dropdown', 'value')],
        [State('stored-loss-data-output', 'data'),
        State('stored-pet-train', 'data')]
    )
    def update_class_loss_graph(selected_output, loss_file_path, pet_train_data):
        if not loss_file_path or not selected_output:
            return go.Figure()

        loss_data = load_json_from_file(loss_file_path)

        if "TRAIN_SPLIT" not in loss_data or "VALID_SPLIT" not in loss_data:
            return go.Figure()

        train_loss = loss_data["TRAIN_SPLIT"].get(selected_output, [])
        valid_loss = loss_data["VALID_SPLIT"].get(selected_output, [])

        n_epochs = None
        if pet_train_data and isinstance(pet_train_data, dict):
            last_key = sorted(pet_train_data.keys())[-1]
            pet_file_path = pet_train_data[last_key]
            pet_data = load_json_from_file(pet_file_path)
            
            if pet_data:
                history = pet_data.get("history", {})
                if history:
                    try:
                        epoch_keys = [int(k) for k in history.keys()]
                        n_epochs = max(epoch_keys) + 1
                    except Exception as e:
                        print(f"Error al obtener epochs: {e}")

        if n_epochs and n_epochs > 0:
            total_train = len(train_loss)
            total_valid = len(valid_loss)
            block_size_train = total_train // n_epochs if total_train >= n_epochs else 1
            block_size_valid = total_valid // n_epochs if total_valid >= n_epochs else 1

            new_train = [
                sum(train_loss[i:i + block_size_train]) / len(train_loss[i:i + block_size_train])
                for i in range(0, total_train, block_size_train)
            ]
            new_valid = [
                sum(valid_loss[i:i + block_size_valid]) / len(valid_loss[i:i + block_size_valid])
                for i in range(0, total_valid, block_size_valid)
            ]
        else:
            new_train, new_valid = train_loss, valid_loss

        formatted_title = selected_output.replace("_", " ").title()
        return generate_class_loss_graph(f"Loss for {formatted_title}", new_train, new_valid)
   

    
    @app.callback(
        [
            Output('output-dropdown', 'options'),
            Output('output-dropdown', 'value')
        ],
        [Input('stored-pet-train', 'data')]
    )
    def update_output_dropdown(train_data):
        if not train_data:
            return [], None

        outputs = {}
        for key, file_path in train_data.items():
            train_json = load_json_from_file(file_path)
            if not train_json:
                continue
            if "output_to_track" in train_json:
                custom_name = train_json["output_to_track"].get("custom_name", "unknown")
                optimized_via = train_json["output_to_track"].get("optimized_via", "unknown")
                outputs[key] = {
                    'label': f"{custom_name.replace('_', ' ').title()} ({optimized_via.replace('_', ' ').title()})",
                    'value': key
                }
        output_options = list(outputs.values())
        if not output_options:
            return [], None
        default_output = output_options[0]['value']
        return output_options, default_output

    @app.callback(
        [
            Output('metric-dropdown', 'options'),
            Output('metric-dropdown', 'value')
        ],
        [Input('output-dropdown', 'value')],
        [State('stored-pet-train', 'data')]
    )
    def update_metric_dropdown(selected_output, train_data):
        if not train_data or not selected_output:
            return [], None

        train_json = load_json_from_file(train_data.get(selected_output, ''))
        if not train_json or "performance_evaluation" not in train_json:
            return [], None

        available_metrics = train_json["performance_evaluation"].get("evaluation_metrics", [])
        if not available_metrics:
            return [], None

        metric_options = [{'label': metric.replace('_', ' ').title(), 'value': metric} for metric in available_metrics]
        default_metric = metric_options[0]['value'] if metric_options else None
        return metric_options, default_metric

    @app.callback(
        [
            Output('train-metrics-graph', 'figure'),
            Output('valid-metrics-graph', 'figure')
        ],
        [
            Input('metric-dropdown', 'value')
        ],
        [
            State('output-dropdown', 'value'),
            State('stored-pet-train', 'data'),
            State('stored-pet-valid', 'data'),
            State('stored-file-info', 'data')
        ]
    )
    def update_metric_graphs(selected_metric, selected_output, pet_train_store, pet_valid_store, file_info_store):
        if not selected_metric or not selected_output or not pet_train_store or not pet_valid_store or not file_info_store:
            return go.Figure(), go.Figure()

        train_file_path = pet_train_store.get(selected_output, '')
        valid_file_path = pet_valid_store.get(selected_output, '')

        train_json = load_json_from_file(train_file_path)
        valid_json = load_json_from_file(valid_file_path)

        if not train_json or "history" not in train_json:
            return go.Figure(), go.Figure()
        if not valid_json or "history" not in valid_json:
            return go.Figure(), go.Figure()

        train_history = train_json["history"]
        valid_history = valid_json["history"]

        outputs = file_info_store.get("outputs", [])
        class_labels = None
        problem_type = None

        for output in outputs:
            if selected_output.lower() in output.get("custom_name", "").lower():
                problem_type = str(output.get("optimized_via.value", "")).upper()
                if problem_type == ProblemType.CLASSIFICATION.value:
                    class_labels = output.get("class_names", [])
                break

        if not problem_type:
            return go.Figure(), go.Figure()

        train_epochs, train_metric_values = [], []
        valid_epochs, valid_metric_values = [], []

        if problem_type == ProblemType.CLASSIFICATION.value:
            if not class_labels:
                return go.Figure(), go.Figure()

            train_epochs, train_metric_values = compute_metric_evolution_classification(train_history, selected_metric, class_labels)
            valid_epochs, valid_metric_values = compute_metric_evolution_classification(valid_history, selected_metric, class_labels)

        elif problem_type == ProblemType.REGRESSION.value:  
            train_epochs, train_metric_values = compute_metric_evolution_regression(train_history, selected_metric)
            valid_epochs, valid_metric_values = compute_metric_evolution_regression(valid_history, selected_metric)

        formatted_metric_name = selected_metric.title().replace('Mse', 'MSE').replace('Mae', 'MAE').replace('_', ' ')

        train_fig = create_metric_figure(train_epochs, train_metric_values, formatted_metric_name, selected_output, "Train", "blue")
        valid_fig = create_metric_figure(valid_epochs, valid_metric_values, formatted_metric_name, selected_output, "Valid", "red")

        return train_fig, valid_fig
    
    @app.callback(
        [
            Output('confusion-matrix-graph', 'figure'),
            Output('residuals-graph', 'figure')
        ],
        [
            Input('split-radio', 'value'),
            Input('output-dropdown', 'value')
        ],
        [
            State('stored-file-info', 'data'),
            State('stored-pet-train', 'data'),
            State('stored-pet-valid', 'data')
        ]
    )
    def update_graphs(split_type, selected_output, file_info_store, pet_train_store, pet_valid_store):
        if not selected_output or not file_info_store:
            return go.Figure(), go.Figure()

        outputs = file_info_store.get("outputs", [])
        selected_output_info = next(
            (output for output in outputs if selected_output.lower() in output.get("custom_name", "").lower()),
            None
        )
        train_file_path = pet_train_store.get(selected_output, '')
        valid_file_path = pet_valid_store.get(selected_output, '')
        train_json = load_json_from_file(train_file_path)
        valid_json = load_json_from_file(valid_file_path)
        
        if not selected_output_info:
            return go.Figure(), go.Figure()

        problem_type = str(selected_output_info.get("optimized_via.value", "")).upper()

        # CLASIFICACIÓN - Confusion Matrix using percentiles
        if problem_type == ProblemType.CLASSIFICATION.value:
            history_train = train_json.get("history", {})
            history_valid = valid_json.get("history", {})

            def compute_percentile_matrices(history):
                # Ordenar los epochs de forma numérica
                epoch_keys = sorted(history.keys(), key=int)
                # Obtener la lista de matrices en orden
                matrices = [np.array(history[ep]) for ep in epoch_keys]
                matrices_stack = np.array(matrices)  # Shape: (num_epochs, rows, cols)
                # Definir los percentiles deseados: 20, 40, 60, 80 y 100
                percentiles = [20, 40, 60, 80, 100]
                # Calcular los percentiles a lo largo del eje 0 (entre epochs)
                percentile_matrices = np.percentile(matrices_stack, q=percentiles, axis=0)
                # Crear un diccionario con 5 claves (1 a 5)
                result = {}
                for i in range(5):
                    result[i + 1] = percentile_matrices[i]
                return result

            percentile_matrices_train = compute_percentile_matrices(history_train)
            percentile_matrices_valid = compute_percentile_matrices(history_valid)
                    
            # Utilizar los nombres de clase reales si existen
            class_labels = selected_output_info.get("class_names", [])
            if not class_labels:
                class_labels = [f'Class {i+1}' for i in range(percentile_matrices_train[1].shape[0])]
            classes = class_labels

            # Seleccionar el conjunto de matrices según split
            percentile_matrices = percentile_matrices_train if split_type == 'train' else percentile_matrices_valid

            # Calcular las etiquetas de epoch basadas en percentiles
            history_used = history_train if split_type == 'train' else history_valid
            epoch_keys_sorted = sorted(history_used.keys(), key=int)
            M = len(epoch_keys_sorted)
            percentiles = [20, 40, 60, 80, 100]
            epoch_labels = [epoch_keys_sorted[int(round((q/100)*(M-1)))] for q in percentiles]
            
            fig = create_flat_3d_heatmap(split_type, classes, percentile_matrices, epoch_labels)
            return go.Figure(), fig

        elif problem_type == ProblemType.REGRESSION.value:  
            history_train = train_json.get("history", {})
            history_valid = valid_json.get("history", {})

            history = history_train if split_type == 'train' else history_valid
            residuals_fig = create_residuals_graph(history)
            return go.Figure(), residuals_fig

        return go.Figure(), go.Figure()

