import plotly.graph_objects as go
import numpy as np

def create_residuals_graph(history):
    """
    Creates and returns a Plotly figure showing the average residual per epoch
    with a standard deviation band.
    
    Parameters:
      - history: dict, where each key (string) represents an epoch and its value is a list of residuals.
    
    Returns:
      - residuals_fig: Plotly Figure with the residuals graph.
    """
    # Convert epoch keys to integers for plotting
    epochs = list(history.keys())
    epochs = [int(epoch) for epoch in epochs]
    
    residuals = []
    std_devs = []
    
    for epoch in epochs:
        values = history[str(epoch)]
        residuals.append(np.mean(values))
        std_devs.append(np.std(values))
    
    residuals_fig = go.Figure()
    
    # Plot the residuals line (average) with thicker width
    residuals_fig.add_trace(go.Scatter(
        x=[epoch + 1 for epoch in epochs],
        y=residuals,
        mode='lines',
        name='Residuals',
        line=dict(color='blue', width=4)
    ))
    
    # Plot the upper bound of the standard deviation (without fill)
    residuals_fig.add_trace(go.Scatter(
        x=[epoch + 1 for epoch in epochs],
        y=np.array(residuals) + np.array(std_devs),
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    
    # Plot the lower bound and fill between for the standard deviation band
    residuals_fig.add_trace(go.Scatter(
        x=[epoch + 1 for epoch in epochs],
        y=np.array(residuals) - np.array(std_devs),
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(0,0,255,0.2)',
        showlegend=False
    ))
    
    # Update layout using similar styling as the confusion matrix graph
    residuals_fig.update_layout(
        title='Residuals with Standard Deviation',
        xaxis_title='Epoch',
        yaxis_title='Residuals',
        height=600,
        margin=dict(l=0, r=0, b=0, t=50),
        plot_bgcolor='white',       # White background for the plotting area
        paper_bgcolor='white',      # White background for the outer frame
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black'
        )
    )
    
    return residuals_fig
