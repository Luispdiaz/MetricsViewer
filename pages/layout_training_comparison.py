import dash
from dash import html

dash.register_page(__name__, path='/comparison', name='Comparison')  # Registrar la página de comparación

layout = html.Div([
    html.H1('Training Comparison Page', style={'textAlign': 'center'}),
    html.Div('Content for comparing training metrics will go here.'),
])
