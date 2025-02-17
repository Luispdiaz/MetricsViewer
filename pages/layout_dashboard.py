import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import numpy as np
from callbacks.dashboard_callbacks import all_cards

dash.register_page(__name__, path='/dashboard', name='Dashboard')

def generate_metrics_evolution(num_classes=3):
    epochs = np.arange(1, 21)
    
    train_metrics = {
        'Accuracy': {f'Class {i+1}': np.random.uniform(0.7, 1.0, size=20) for i in range(num_classes)},
        'Precision': {f'Class {i+1}': np.random.uniform(0.6, 0.9, size=20) for i in range(num_classes)},
        'Recall': {f'Class {i+1}': np.random.uniform(0.5, 0.85, size=20) for i in range(num_classes)},
    }

    valid_metrics = {
        'Accuracy': {f'Class {i+1}': np.random.uniform(0.65, 0.95, size=20) for i in range(num_classes)},
        'Precision': {f'Class {i+1}': np.random.uniform(0.55, 0.88, size=20) for i in range(num_classes)},
        'Recall': {f'Class {i+1}': np.random.uniform(0.5, 0.80, size=20) for i in range(num_classes)},
    }

    return train_metrics, valid_metrics, epochs

def generate_loss_function(output_type):
    epochs = np.arange(1, 21)
    if output_type == 'total':
        loss_values = np.random.uniform(0.2, 0.5, size=20)
    else:
        loss_values = np.random.uniform(0.3, 0.7, size=20)  
    return loss_values, epochs

layout = html.Div([
    dbc.Row([
        dbc.Col(
            dbc.Nav(
                [
                    html.H5("Sections", className='text-left', style={'marginBottom': '10px', 'fontWeight': 'bold'}),
                    dbc.NavLink("General Description", href="#section-description", external_link=True),
                    dbc.NavLink("Evolution Metrics", href="#section-evolution-metrics", external_link=True),
                    dbc.NavLink("Loss Function", href="#section-class-loss", external_link=True),
                ],
                vertical=True,
                pills=True,
                className='bg-light p-3',
            ),
            width=2,
            style={'position': 'sticky', 'top': '20px', 'alignSelf': 'flex-start'} 
        ),

        dbc.Col(
            [
                html.H1("AI Model Dashboard", style={'textAlign': 'left', 'marginTop': '20px', 'marginBottom': '20px'}),

                html.Div(id='section-description', children=[
                    html.H2("General Description", style={'marginTop': '20px', 'marginBottom': '20px'}),
                    
                    dbc.Row(id='dynamic-card-row'),

                    dbc.Row([
                        dbc.Col(
                            dbc.Button("Add", id="add-card-button", n_clicks=0, className="btn btn-primary"),
                            width=2,
                        )
                    ]),

                    dbc.Modal(
                        [
                            dbc.ModalHeader("Add a card"),
                            dbc.ModalBody([
                                html.Div("Select the cards to add:"),
                                
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Checklist(
                                            id='available-cards-checklist-1', 
                                            options=[{'label': all_cards[card]['title'], 'value': card}
                                                     for card in list(all_cards.keys())[:len(all_cards)//2]],  
                                            value=['Model Name', 'Model Version', 'Model Type', 'Number of Epochs', 'Training Time'],
                                            className="p-2",
                                            labelStyle={'marginBottom': '5px'},
                                        ),
                                        width=6  
                                    ),
                                    dbc.Col(
                                        dcc.Checklist(
                                            id='available-cards-checklist-2',  
                                            options=[{'label': all_cards[card]['title'], 'value': card}
                                                     for card in list(all_cards.keys())[len(all_cards)//2:]],  
                                            value=[],
                                            className="p-2",
                                            labelStyle={'marginBottom': '5px'},
                                        ),
                                        width=6  
                                    ),
                                ]),
                            ]),
                            dbc.ModalFooter(
                                dbc.Button("Ready", id="ready-button", className="ms-auto", n_clicks=0)
                            ),
                        ],
                        id="modal",
                        is_open=False,
                    ),
                ]),

                html.Div(id='section-evolution-metrics', children=[
                    html.H2("Evolution Metrics", style={'marginTop': '50px'}),

                    dbc.Row([
                        dbc.Col(
                            dcc.Dropdown(
                                id='output-dropdown',
                                clearable=False,
                                placeholder='Select Output Type',
                                style={'marginBottom': '20px', 'width': '300px'},
                            ),
                            width=6,
                            style={'display': 'flex', 'justifyContent': 'flex-end'}
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id='metric-dropdown',
                                options=[], 
                                placeholder='Select Metric',
                                clearable=False,
                                style={'marginBottom': '20px', 'width': '220px'},
                            ),
                            width=6,
                            style={'display': 'flex', 'justifyContent': 'flex-start'}
                        ),
                    ]),

                    dbc.Row([
                        dbc.Col(
                            dcc.Loading(children=[dcc.Graph(id='train-metrics-graph')], type="default"),
                            width=6
                        ),
                        dbc.Col(
                            dcc.Loading(children=[dcc.Graph(id='valid-metrics-graph')], type="default"),
                            width=6
                        ),
                    ]),

                    dcc.RadioItems(
                        id='split-radio',
                        options=[
                            {'label': 'Train', 'value': 'train'},
                            {'label': 'Valid', 'value': 'valid'}
                        ],
                        value='train',  
                        labelStyle={
                            'display': 'inline-block',
                            'margin': '0 10px',
                            'fontSize': '19px',  
                        },
                        style={'textAlign': 'center', 'margin': '20px', 'fontSize': '16px', 'fontWeight': 'bold', 'color': '#333'}
                    ),

                    dcc.Graph(id='confusion-matrix-graph', style={'display': 'none'}),
                    dcc.Loading(children=[dcc.Graph(id='residuals-graph', style={'display': 'block'})], type="default")

                ]),

                html.Div(id='section-class-loss', children=[
                    html.H2("Loss Function", style={'marginTop': '50px'}),

                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(id='train-class-loss-graph'),
                            width=6
                        ),

                        dbc.Col(
                            html.Div([
                                html.Div([
                                    dcc.Dropdown(
                                        id='output-class-dropdown',
                                        clearable=False,
                                        style={'width': '200px', 'marginBottom': '20px'}
                                    ),
                                ], style={'display': 'flex', 'justifyContent': 'center'}),  

                                dcc.Graph(id='class-loss-graph', style={'width': '100%'})
                            ], style={'width': '100%'})  

                        , width=6) 

                    ]),
                ])
            ],
            width=10
        ),
    ], style={'padding': '20px'})
])
