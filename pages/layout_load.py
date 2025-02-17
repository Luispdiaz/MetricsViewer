import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Load')  # Registrar la página de carga

layout = html.Div([
    html.H1("View Training Metrics", style={'textAlign': 'center', 'fontWeight': 'bold'}),
    html.H3("Choose the folder you want to analyze. Quick and easy!", style={'textAlign': 'center', 'fontWeight': 'normal'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        multiple=True,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    html.Div(id='output-data-upload'),
    html.Div(
        dbc.Button('Start', id='check-files-button', color='primary', style={'marginTop': '20px', 'marginLeft': '17px'}),
    ),
    html.Div(id='check-files-result'),
], style={
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'marginTop': '30px'
        })
