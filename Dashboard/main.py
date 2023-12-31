import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from modules.apple_dashboard import AppleDashboard
from modules.tesla_dashboard import TeslaDashboard
from modules.microsoft_dashboard import MicrosoftDashboard
from modules.google_dashboard import GoogleDashboard

# Create dash app and set external stylesheets
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define empty layout to be filled later depending on the page
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

# Define main layout with buttons to navigate to different dashboards when no other page is loaded
main_menu_layout = html.Div([
    html.H1('Stock Prices Visualization and Prediction', style={'textAlign': 'center'}),
    html.P(
        ["Data extracted from ", html.A('StocksData', href='https://www.stockdata.org/')],
        style={'textAlign': 'center', 'margin-bottom': '20px'}
    ),

    dbc.Row(
        [
            dbc.Col(
                html.Img(src="/assets/apple_logo.png", style={'height': '80px', 'width': '80px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
            dbc.Col(
                dbc.Button("Apple", id="apple_button", n_clicks=0, href='/apple_dashboard', style={'height': '50px', 'font-size': '18px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
            dbc.Col(
                html.Img(src="/assets/google_logo.png", style={'height': '80px', 'width': '80px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
            dbc.Col(
                dbc.Button("Google", id="google_button", n_clicks=0, href='/google_dashboard', style={'height': '50px', 'font-size': '18px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
        ],
        justify='center',
        className='mt-3',
    ),
    dbc.Row(
        [
            dbc.Col(
                html.Img(src="/assets/microsoft_logo.png", style={'height': '80px', 'width': '80px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
            dbc.Col(
                dbc.Button("Microsoft", id="microsoft_button", n_clicks=0, href='/microsoft_dashboard', style={'height': '50px', 'font-size': '18px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
            dbc.Col(
                html.Img(src="/assets/tesla_logo.png", style={'height': '80px', 'width': '80px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
            dbc.Col(
                dbc.Button("Tesla", id="tesla_button", n_clicks=0, href='/tesla_dashboard', style={'height': '50px', 'font-size': '18px'}),
                width=2,
                style={'textAlign': 'center', 'margin-bottom': '20px'}
            ),
        ],
        justify='center',
        className='mt-3',
    ),
])




# Initialize dashboard instances. Create their variables, setup their layouts and callbacks
apple_dashboard = AppleDashboard()
apple_dashboard.initialize_variables()
apple_dashboard.setup_layout()
apple_dashboard.callbacks(app)

google_dashboard = GoogleDashboard()
google_dashboard.initialize_variables()
google_dashboard.setup_layout()
google_dashboard.callbacks(app)

microsoft_dashboard = MicrosoftDashboard()
microsoft_dashboard.initialize_variables()
microsoft_dashboard.setup_layout()
microsoft_dashboard.callbacks(app)

tesla_dashboard = TeslaDashboard()
tesla_dashboard.initialize_variables()
tesla_dashboard.setup_layout()
tesla_dashboard.callbacks(app)
      
# Callback to render different dashboards based on button clicks
"""Retrieved from: https://community.plotly.com/t/trigger-callback-when-url-path-is-updated-within-dash-2-5-multi-page-app/75125/2"""
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/apple_dashboard':
        return apple_dashboard.layout
    elif pathname == '/google_dashboard':
        return google_dashboard.layout
    elif pathname == '/microsoft_dashboard':
        return microsoft_dashboard.layout
    if pathname == '/tesla_dashboard':
        return tesla_dashboard.layout
    else:
        return main_menu_layout
    
if __name__ == '__main__':
    app.run_server(debug=False)