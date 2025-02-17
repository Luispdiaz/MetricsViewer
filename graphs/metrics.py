import plotly.graph_objects as go
import numpy as np
from utils.confusion_matrix import ConfusionMatrix
from utils.residuals import Residuals
from enums.performance_evaluation_manager_enums import RegressionMetric

def compute_metric_evolution_classification(history: dict, metric: str, class_labels: list):
    """
    Calcula la evolución de una métrica a partir del history de clasificación.
    Retorna una tupla (epochs, metric_values).
    """
    epoch_keys = sorted(history.keys(), key=int)
    epochs = []
    metric_values = []
    metric_lower = metric.lower()
    
    for ep in epoch_keys:
        matrix = np.array(history[ep])
        cm = ConfusionMatrix(class_labels)
        cm.matrix = matrix

        if metric_lower == 'accuracy':
            value = cm.accuracy()
        elif metric_lower == 'precision':
            prec = cm.precision()
            value = np.mean(list(prec.values()))
        elif metric_lower == 'recall':
            rec = cm.recall()
            value = np.mean(list(rec.values()))
        elif metric_lower == 'f1 score':
            f1 = cm.f1_score() 
            value = np.mean(list(f1.values()))
        elif metric_lower == 'specificity':
            spec = cm.specificity()
            value = np.mean(list(spec.values()))
        elif metric_lower == 'fallout':
            fol = cm.fallout()
            value = np.mean(list(fol.values()))
        else:
            value = 0

        epochs.append(int(ep) + 1) 
        metric_values.append(value)

    return epochs, metric_values

def compute_metric_evolution_regression(history: dict, metric: str):
    """
    Calcula la evolución de una métrica a partir del history de regresión.
    Retorna una tupla (epochs, metric_values).
    """
    epoch_keys = sorted(history.keys(), key=int)
    epochs = []
    metric_values = []
    
    for ep in epoch_keys:
        residuals = np.array(history[ep])
        residuals_tool = Residuals()
        residuals_tool.update(residuals)

        evaluation_metric_enum = RegressionMetric[metric.upper().replace(' ', '')]
        metric_result = residuals_tool.get_metrics([evaluation_metric_enum])
        value = metric_result.get(metric.replace(' ', '_').lower(), 0)

        epochs.append(int(ep) + 1)
        metric_values.append(value)

    return epochs, metric_values

def create_metric_figure(epochs, metric_values, formatted_metric_name, selected_output, dataset_type, color):
    """
    Crea y retorna una figura de Plotly para la evolución de una métrica.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=metric_values,
        mode='lines',
        name=f"{formatted_metric_name} ({dataset_type})",
        line=dict(color=color, width=3)
    ))
    fig.update_layout(
        title=f"{formatted_metric_name} Evolution ({dataset_type}) - {selected_output}",
        xaxis_title="Epoch",
        yaxis_title=formatted_metric_name,
        hovermode="x unified"
    )
    return fig
