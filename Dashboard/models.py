from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

current_dir = os.path.dirname(__file__)

"""Retrieved from: 
https://docs.python.org/3/library/os.path.html
https://www.geeksforgeeks.org/python-os-path-join-method/
"""
file_path = os.path.abspath(os.path.join(current_dir, '../Data/Transformed/apple_stocks_transformed.csv'))
aapl_df = pd.read_csv(file_path)

file_path = os.path.abspath(os.path.join(current_dir, '../Data/Transformed/google_stocks_transformed.csv'))
googl_df = pd.read_csv(file_path)

file_path = os.path.abspath(os.path.join(current_dir, '../Data/Transformed/microsoft_stocks_transformed.csv'))
msft_df = pd.read_csv(file_path)

file_path = os.path.abspath(os.path.join(current_dir, '../Data/Transformed/tesla_stocks_transformed.csv'))
tsla_df = pd.read_csv(file_path)

# Data package definition
global Size_Block
Size_Block = 100

global n_features
n_features = 1  

# Data partition function
def data_generator(data, window, distance=0):
    x = []
    y = []
    for i in range(len(data) - window - distance):
        # Extract multiple columns for input features 
        # features = data.iloc[i:i+window][['close', 'volume', 'momentum_rsi', 'trend_macd']]
        features = data.iloc[i:i+window][['close']]

        x.append(features.values)  # Append multiple columns as input features
        y.append(data.iloc[i+window+distance]['close'])  # Target variable (close_prices)
    return np.array(x), np.array(y)

def post_processing_process(predictions, data_to_predict, df):
    # Calculate graphics midpoints and modify the predictions to the midpoint of real data
    midpoint1 = np.min(data_to_predict["close"]) + (np.max(data_to_predict["close"])-np.min(data_to_predict["close"]))/2   # Real data midpoint 
    midpoint2 = np.min(predictions) + (np.max(predictions)-np.min(predictions))/2    # Predicted data midpoint

    predictions = predictions + (midpoint1 - midpoint2)
    midpoint3 = np.min(predictions) + (np.max(predictions)-np.min(predictions))/2    # Data predicted midpoint after modification

    # Relationship between maximum and minimum
    alpha1 = (np.max(data_to_predict["close"])-np.min(data_to_predict["close"])) / (np.max(predictions)-np.min(predictions))
    
    #----Calculate beta------#
    beta1 = (midpoint3*alpha1)-midpoint3 # With midpoint of the predictions times alpha
    if(alpha1 < 1):
        sign = 1
    else:
        sign = -1

    predictions = predictions*alpha1+(sign*beta1)

    return predictions

"""Retrieved from: https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c"""

app = Flask(__name__)

# Load apple model
apple_path = os.path.abspath(os.path.join(current_dir, '../Models/apple_model.pkl'))
with open(apple_path, 'rb') as model_file:
    apple_model = pickle.load(model_file)


# Load tesla model
tesla_path = os.path.abspath(os.path.join(current_dir, '../Models/tesla_model.pkl'))
with open(tesla_path, 'rb') as model_file:
    tesla_model = pickle.load(model_file)

# Load microsoft model
microsoft_path = os.path.abspath(os.path.join(current_dir, '../Models/microsoft_model.pkl'))
with open(microsoft_path, 'rb') as model_file:
    microsoft_model = pickle.load(model_file)

# Load google model
google_path = os.path.abspath(os.path.join(current_dir, '../Models/google_model.pkl'))
with open(google_path, 'rb') as model_file:
    google_model = pickle.load(model_file)


@app.route('/predict_apple', methods=['POST'])
def predict_apple():
    # Get data from the request
    data = request.json

    # Convert list to DataFrame
    data = pd.DataFrame(data, columns=["close"])

    predic_x,actual_y=data_generator(data, Size_Block , 0)
    predic_x = np.reshape(predic_x, (predic_x.shape[0], predic_x.shape[1],n_features))
    print(predic_x.shape)

    # Make predictions using the loaded model
    predictions = apple_model.predict(predic_x)

    # Post-process your predictions if needed
    predictions = post_processing_process(predictions, data, aapl_df)

    # Return predictions as a JSON response
    return jsonify({'predictions': predictions.tolist()})


@app.route('/predict_tesla', methods=['POST'])
def predict_tesla():
    # Get data from the request
    data = request.json

    # Convert list to DataFrame
    data = pd.DataFrame(data, columns=["close"])

    predic_x,actual_y=data_generator(data, Size_Block , 0)
    predic_x = np.reshape(predic_x, (predic_x.shape[0], predic_x.shape[1],n_features))
    print(predic_x.shape)

    # Make predictions using the loaded model
    predictions = tesla_model.predict(predic_x)

    # Post-process your predictions if needed
    predictions = post_processing_process(predictions, data, tsla_df)

    # Return predictions as a JSON response
    return jsonify({'predictions': predictions.tolist()})


@app.route('/predict_microsoft', methods=['POST'])
def predict_microsoft():
    # Get data from the request
    data = request.json

    # Convert list to DataFrame
    data = pd.DataFrame(data, columns=["close"])

    predic_x,actual_y=data_generator(data, Size_Block , 0)
    predic_x = np.reshape(predic_x, (predic_x.shape[0], predic_x.shape[1],n_features))
    print(predic_x.shape)

    # Make predictions using the loaded model
    predictions = microsoft_model.predict(predic_x)

    # Post-process your predictions if needed
    predictions = post_processing_process(predictions, data, msft_df)

    # Return predictions as a JSON response
    return jsonify({'predictions': predictions.tolist()})


@app.route('/predict_google', methods=['POST'])
def predict_google():
    # Get data from the request
    data = request.json

    # Convert list to DataFrame
    data = pd.DataFrame(data, columns=["close"])

    predic_x,actual_y=data_generator(data, Size_Block , 0)
    predic_x = np.reshape(predic_x, (predic_x.shape[0], predic_x.shape[1],n_features))
    print(predic_x.shape)

    # Make predictions using the loaded model
    predictions = google_model.predict(predic_x)

    # Post-process your predictions if needed
    predictions = post_processing_process(predictions, data, googl_df)

    # Return predictions as a JSON response
    return jsonify({'predictions': predictions.tolist()})

    
if __name__ == '__main__':
    app.run(port=5000, debug=True)
