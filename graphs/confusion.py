import plotly.graph_objects as go
import numpy as np

def create_flat_3d_heatmap(split_type, classes, averaged_matrices):
    fig = go.Figure()

    n_classes = averaged_matrices[1].shape[0]  

    z_offsets = list(averaged_matrices.keys())  

    block_size = 1

    z_offset_base = 0  

    for idx, (z_offset, matrix) in enumerate(averaged_matrices.items()):
        current_z_offset = z_offset_base + idx  

        if current_z_offset >= len(z_offsets):
            break

        for i in range(matrix.shape[0]):  
            for j in range(matrix.shape[1]):  
                fig.add_trace(go.Mesh3d(
                    x=[i, i + 1, i + 1, i, i, i + 1, i + 1, i],  
                    y=[j, j, j + 1, j + 1, j, j, j + 1, j + 1],  
                    z=[current_z_offset, current_z_offset, current_z_offset, current_z_offset, 
                       current_z_offset + 1, current_z_offset + 1, current_z_offset + 1, current_z_offset + 1],
                    intensity=[matrix[i, j]] * 8,
                    colorscale='Viridis',
                    cmin=0, cmax=np.max(matrix),
                    opacity=1,
                    showscale=(i == 0 and j == 0 and z_offset == 1),
                    hovertext=f'Valor: {matrix[i, j]}',
                    hoverinfo='text'
                ))

                contour_x = [i, i + 1, i + 1, i, i]
                contour_y = [j, j, j + 1, j + 1, j]
                contour_z = [current_z_offset] * 5

                fig.add_trace(go.Scatter3d(
                    x=contour_x,
                    y=contour_y,
                    z=contour_z,
                    mode='lines',
                    line=dict(color='white', width=2),
                    showlegend=False
                ))

                fig.add_trace(go.Scatter3d(
                    x=contour_x,
                    y=contour_y,
                    z=[current_z_offset + 1] * 5,
                    mode='lines',
                    line=dict(color='white', width=2),
                    showlegend=False
                ))

    fig.update_layout(
        title=f'Confusion Matrices',
        scene=dict(
            xaxis_title='Predicted Class',
            yaxis_title='True Class',
            zaxis_title='Epoch Segment',
            xaxis=dict(range=[0, n_classes], showgrid=False),  
            yaxis=dict(range=[0, n_classes], showgrid=False),  
            zaxis=dict(range=[0, len(z_offsets)-1], showgrid=False), 
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)), 
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig
