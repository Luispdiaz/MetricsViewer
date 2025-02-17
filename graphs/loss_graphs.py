import plotly.graph_objects as go
import numpy as np


def generate_loss_values_for_output(output):
    epochs = np.arange(1, 21)
    if output == 'output_1':
        train_loss = np.random.uniform(0.1, 0.3, size=20)
        valid_loss = np.random.uniform(0.2, 0.4, size=20)
    elif output == 'output_2':
        train_loss = np.random.uniform(0.2, 0.4, size=20)
        valid_loss = np.random.uniform(0.3, 0.5, size=20)
    elif output == 'output_3':
        train_loss = np.random.uniform(0.15, 0.35, size=20)
        valid_loss = np.random.uniform(0.25, 0.45, size=20)
    return train_loss, valid_loss, epochs

def generate_class_total_loss_graph(title, loss_data):
    if not loss_data:  
        epochs = []
        train_loss_values = []
        valid_loss_values = []
    else:
        epochs = np.arange(1, len(loss_data["TRAIN_SPLIT"]) + 1)
        train_loss_values = loss_data["TRAIN_SPLIT"]
        valid_loss_values = loss_data["VALID_SPLIT"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss_values,
        mode='lines',
        name='Train Loss',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=valid_loss_values,
        mode='lines',
        name='Validation Loss',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode="x unified"
    )

    return fig

def generate_class_loss_graph(title, train_loss, valid_loss):
    epochs = np.arange(1, len(train_loss) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_loss,
        mode='lines',
        name='Train Loss',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=valid_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode="x unified"
    )

    return fig