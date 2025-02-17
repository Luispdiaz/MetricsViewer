import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import yaml

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.LUMEN])

app.layout = html.Div([
    dcc.Location(id='url', refresh=True), 
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Metrics Viewer", href="/"),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Training Overview", href="/", id='link-load')),
                    #dbc.NavItem(dbc.NavLink("Comparison", href="/comparison", id='link-comparison')),
                    dbc.NavItem(dbc.NavLink("About", href="/about", id='link-about')),
                ],
                className="ml-auto",
            )
        ]),
        color="primary",
        dark=True,
    ),
    dash.page_container,
    dcc.Store(id='stored-file-info', storage_type='session'),
    dcc.Store(id='stored-loss-data-total', storage_type='session'),
    dcc.Store(id='stored-loss-data-output', storage_type='session'),
    dcc.Store(id='stored-pet-train', storage_type='session'),
    dcc.Store(id='stored-pet-valid', storage_type='session'),
])

import pages.layout_load
import pages.layout_training_comparison
import pages.layout_about
import pages.layout_dashboard

from callbacks.load_callbacks import register_load_callbacks
from callbacks.dashboard_callbacks import register_dashboard_callbacks

register_load_callbacks(app)
register_dashboard_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=config['app']['debug'], port=config['app']['port'])

