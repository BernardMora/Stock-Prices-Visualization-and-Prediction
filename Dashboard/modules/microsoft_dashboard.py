import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import requests
from itertools import chain
import numpy as np
from modules.errors import ErrorAbs, ErrorRel
import os
import sys

# To import from models.py, we need to add the parent directory to the path
"""Retrieved from: 
https://docs.python.org/3/library/os.path.html
https://www.geeksforgeeks.org/python-os-path-dirname-method/
https://www.geeksforgeeks.org/python-os-path-join-method/
"""
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models import Size_Block

# We only needed it for the import, so we can remove it
sys.path.remove(parent_dir) 


microsoft_dashboard = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, '/assets/fontawesome-all.css'])

# Made class for better modularity and easier imports
class MicrosoftDashboard:
    def __init__(self):
        self.msft_data = None
        self.msft_data_untransformed = None
        self.norm_factor_msft = None
        self.start_date_msft = None
        self.end_date_msft = None
        self.model_endpoint_msft = None
        self.stored_predictions_msft = None
        self.selected_data_msft = None
        self.layout = None

    # Function to initialize variables for microsoft dashboard
    def initialize_variables(self):
        # Get the current directory of the script
        current_dir = os.path.dirname(__file__)

        # Navigate from Dashboard directory (current) to Data directory and read the CSV file
        file_path = os.path.abspath(os.path.join(current_dir, '../../Data/Transformed/microsoft_stocks_transformed.csv'))
        self.msft_data = pd.read_csv(file_path)

        file_path2 = os.path.abspath(os.path.join(current_dir, '../../Data/API Gathered/microsoft_stocks.csv'))
        self.msft_data_untransformed = pd.read_csv(file_path2)

        # Reversing the order of rows and changing the index to descending order
        self.msft_data_untransformed = self.msft_data_untransformed[::-1].reset_index(drop=True)

        self.norm_factor_msft = np.linalg.norm(self.msft_data_untransformed['close'])

        self.msft_data['date'] = pd.to_datetime(self.msft_data['date'])

        self.start_date_msft = self.msft_data['date'].iloc[0]
        self.end_date_msft = self.msft_data['date'].iloc[-1]

        # Load your trained model
        self.model_endpoint_msft = "http://127.0.0.1:5000/predict_microsoft"  # Replace with your model's URL

    """Retrieved from: 
    https://dash.plotly.com/layout
    https://realpython.com/python-dash/
    https://dash.plotly.com/dash-core-components/rangeslider
    https://dash.plotly.com/dash-core-components/datepickerrange"""
    # Define layout for microsoft dashboard
    def setup_layout(self):
        self.layout = html.Div([
                    dbc.Row([
                        dbc.Col(html.A(dbc.Button("Back", color="gray"), href='/'), width='auto'),
                        dbc.Col(html.H1("Stock Prices Microsoft", style={'text-align': 'center'}), width=True),
                    ]),
                    html.Img(src="/assets/microsoft_logo.png", style={'width': '80px', 'height': '80px', 'margin': '20px auto', 'display': 'block'}),
                    dcc.RangeSlider(
                        id='date-slider-microsoft',
                        min=0,
                        max=len(self.msft_data) - 1,
                        value=[0, len(self.msft_data) - 1],
                        marks=None,
                        step=1
                    ),
                    dcc.DatePickerRange(
                        id='date-picker-microsoft',
                        start_date=self.start_date_msft,
                        end_date=self.end_date_msft,
                        min_date_allowed=self.start_date_msft,
                        max_date_allowed=self.end_date_msft,
                        display_format='YYYY-MM-DD'
                    ),
                    html.Button('Predict', id='predict-button-microsoft', n_clicks=0, style={'font-size': '18px', 'background-color': 'black', 'margin': '20px auto', 'display': 'block', 'color': 'white'}),  # Adjust button appearance
                    dcc.Graph(id='stock-graph-microsoft'),
                    dcc.Graph(id='error-graph-microsoft')
                ])


    # Callbacks for microsoft dashboard
    def callbacks(self, app):
        @app.callback(
            [Output('date-picker-microsoft', 'start_date'),
            Output('date-picker-microsoft', 'end_date')],
            [Input('date-slider-microsoft', 'value')]
        )
        def update_picker(value):
            if dash.callback_context.triggered[0]['prop_id'].split('.')[0] == 'date-slider-microsoft':
                start_date = self.msft_data['date'].iloc[value[0]]
                end_date = self.msft_data['date'].iloc[value[1]]
                return start_date, end_date
            else:
                return dash.no_update, dash.no_update

        """Retrieved from:
        https://plotly.com/python/creating-and-updating-figures/
        """
        @app.callback(
            [Output('stock-graph-microsoft', 'figure'),
            Output('error-graph-microsoft', 'figure')],
            [Input('date-slider-microsoft', 'value'),
            Input('predict-button-microsoft', 'n_clicks')],
            [State('date-picker-microsoft', 'start_date'),
            State('date-picker-microsoft', 'end_date')]
        )
        def update_graphs(value, n_clicks, start_date, end_date):
            start_idx = value[0]
            end_idx = value[1]
            self.selected_data_msft = self.msft_data_untransformed['close'][start_idx:end_idx]

            fig = go.Figure()
            # Plot real prices
            fig.add_trace(go.Scatter(x=self.msft_data['date'][start_idx:end_idx], y=self.selected_data_msft.tolist(), mode='lines', name='Real Prices'))

            fig.update_layout(
                title="Stock Prices",
                xaxis_title="Date",
                yaxis_title="Close Prices [USD]",
                xaxis=dict(tickangle=45),
                showlegend=True,
                legend=dict(x=0, y=1.0),
                margin=dict(l=40, r=40, t=40, b=40),
                font=dict(size=15)  # Increase font size for the entire figure
            )

            # Make predictions only if the button has been clicked
            if n_clicks > 0:
                # Get the normalized data for predictions
                self.selected_data_msft = self.msft_data['close'][start_idx:end_idx]

                # Make a POST request to the model endpoint
                """Retrieved from: https://www.w3schools.com/python/ref_requests_post.asp"""
                response = requests.post(self.model_endpoint_msft, json=self.selected_data_msft.values.tolist()) # Has to be a list

                # Retrieve the predictions from the response
                predictions = response.json()['predictions']

                # The predictions are retrieved like lists inside lists so we use chain.from_iterable to flatten the list
                """Retrieved from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists"""
                self.stored_predictions_msft = list(chain.from_iterable(predictions))

                # Get the dates and real data for the predictions
                prediction_dates = self.msft_data['date'][Size_Block+start_idx:Size_Block+start_idx+len(predictions)]
                self.selected_data_msft = self.msft_data_untransformed['close'][Size_Block+start_idx:Size_Block+start_idx+len(predictions)]

                self.stored_predictions_msft = np.array(self.stored_predictions_msft) * self.norm_factor_msft  # Reverse the normalization

                fig2 = go.Figure()
                # Plot real prices
                fig2.add_trace(go.Scatter(x=prediction_dates, y=self.selected_data_msft.tolist(), mode='lines', name='Real Prices'))
                # Plot predicted prices
                fig2.add_trace(go.Scatter(x=prediction_dates, y=self.stored_predictions_msft, mode='lines', name='Predicted Prices'))

                fig2.update_layout(
                    title="Stock Prices",
                    xaxis_title="Date",
                    yaxis_title="Close Predicted Prices",
                    xaxis=dict(tickangle=45),
                    showlegend=True,
                    legend=dict(x=0, y=1.0),
                    margin=dict(l=40, r=40, t=40, b=40),
                    font=dict(size=15)  # Increase font size for the entire figure
                )

                error_abs = ErrorAbs(self.stored_predictions_msft, self.selected_data_msft)  # Calculate absolute error
                error_rel = ErrorRel(self.selected_data_msft, error_abs)  # Calculate relative error
                error_rel = error_rel*100  # Convert to percentage

                # Plotting relative error
                fig_error = go.Figure()
                fig_error.add_trace(go.Scatter(x=prediction_dates, y=error_rel, mode='lines', name='Relative Error', line=dict(color='yellow')))

                fig_error.update_layout(
                    title="Relative Error",
                    xaxis_title="Date",
                    yaxis_title="Relative Error [%]",
                    xaxis=dict(tickangle=45),
                    showlegend=True,
                    legend=dict(x=0, y=1.0),
                    margin=dict(l=40, r=40, t=40, b=40),
                    font=dict(size=15)  # Increase font size for the entire figure
                )

                return fig2, fig_error

            return fig, {}

if __name__ == '__main__':
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    microsoft_dashboard = MicrosoftDashboard()
    microsoft_dashboard.initialize_variables()
    microsoft_dashboard.setup_layout()
    app.layout = microsoft_dashboard.layout
    microsoft_dashboard.callbacks(app)
    app.run_server(debug=True)
