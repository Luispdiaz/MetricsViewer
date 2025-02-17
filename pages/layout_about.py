import dash
from dash import html

dash.register_page(__name__, path='/about', name='About')

layout = html.Div(
    style={
        'backgroundColor': '#f7f7f7',
        'padding': '50px 20px',
        'fontFamily': 'Helvetica, Arial, sans-serif',
        'color': '#333',
    },
    children=[
        html.Div(
            style={
                'maxWidth': '900px',
                'margin': '0 auto',
                'backgroundColor': '#fff',
                'padding': '40px',
                'borderRadius': '8px',
                'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
            },
            children=[
                html.H1(
                    'About Metrics Viewer',
                    style={
                        'textAlign': 'center',
                        'fontSize': '36px',
                        'marginBottom': '20px',
                        'color': '#2c3e50'
                    }
                ),
                # Uncomment the following lines if you have a logo in assets folder
                # html.Img(
                #     src='/assets/intelcon_logo.png',
                #     style={
                #         'display': 'block',
                #         'margin': '0 auto 30px',
                #         'width': '150px'
                #     }
                # ),
                html.P(
                    "Welcome to Metrics Viewer, an intuitive and powerful tool designed for visualizing and analyzing the performance metrics of your AI models. With Metrics Viewer, you can upload files containing comprehensive model information and explore detailed data, including Evolution Metrics and Loss Functions.",
                    style={'fontSize': '18px', 'lineHeight': '1.6', 'marginBottom': '20px'}
                ),
                html.P(
                    "Developed by Intelcon System, our platform empowers users to monitor model performance over time and gain valuable insights for optimization and improvement. Metrics Viewer offers an interactive experience that simplifies decision-making by providing clear and actionable data visualizations.",
                    style={'fontSize': '18px', 'lineHeight': '1.6', 'marginBottom': '20px'}
                ),
                html.H2(
                    "Key Features",
                    style={'fontSize': '28px', 'color': '#2c3e50', 'marginBottom': '15px'}
                ),
                html.Ul(
                    children=[
                        html.Li("Interactive visualization of Evolution Metrics to track model progress over epochs."),
                        html.Li("Detailed analysis of Loss Functions to identify optimization opportunities."),
                        html.Li("Seamless file upload and processing for comprehensive model insights."),
                        html.Li("User-friendly interface designed for efficient and intuitive data exploration."),
                    ],
                    style={'fontSize': '18px', 'lineHeight': '1.8', 'marginBottom': '20px'}
                ),
                html.H2(
                    "Version & Contact",
                    style={'fontSize': '28px', 'color': '#2c3e50', 'marginBottom': '15px'}
                ),
                html.P(
                    "Version: 1.0",
                    style={'fontSize': '18px', 'lineHeight': '1.6', 'marginBottom': '5px'}
                ),
                html.P(
                    "Developed by Intelcon System",
                    style={'fontSize': '18px', 'lineHeight': '1.6', 'marginBottom': '5px'}
                ),
                html.P(
                    "For more information, please contact us at: info@intelconsystem.com",
                    style={'fontSize': '18px', 'lineHeight': '1.6'}
                ),
            ]
        )
    ]
)
