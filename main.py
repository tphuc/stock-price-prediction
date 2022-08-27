# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM, Flatten, TimeDistributed, ConvLSTM2D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import xgboost as xgb

from pathlib import Path

# importing libraries
from keras.layers import SimpleRNN

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import pickle


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

prediction_models = ('RNN', 'LSTM', 'XGB')
selected_model = st.selectbox('Select model for prediction', prediction_models)

prediction_types = ('Close Price', 'ROC')
selected_type = st.selectbox('Select feature for prediction', prediction_types)

PERIODS = 30




def make_future_frames(data, target='Close'):
    data['Predictions'] = np.nan
    data['Predictions'].iloc[-1] = data[target].iloc[-1]
    data['Date'] = pd.to_datetime(data['Date'])
    df_future = pd.DataFrame(columns=['Date', target, 'Predictions'])
    df_future['Date'] = pd.date_range(
        start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=PERIODS)
    df_future['Predictions'] = predictions.flatten()
    df_future[target] = np.nan
    results = data.append(df_future)
    return results



class RNNPredictionModel:
    def __init__(self, data, target='Close', test_length=PERIODS, window_size=60):
        self.target_column = target
        data = data.copy()
        self.model_path = 'rnn_model.h5'
        dataset = data.values
        # Get the number of rows to train the model on
        training_data_len = len(dataset) - test_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(dataset)

        train_data = scaled_data[0:int(training_data_len), :]
        # Split the data into x_train and y_train data sets
        self.x_train = []
        self.y_train = []

        for i in range(window_size, len(train_data)):
            self.x_train.append(train_data[i-window_size:i, 0])
            self.y_train.append(train_data[i, 0])

        # Convert the x_train and y_train to numpy arrays
        self.x_train, self.y_train = np.array(
            self.x_train), np.array(self.y_train)

        # Reshape the data
        self.x_train = np.reshape(
            self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        test_data = scaled_data[training_data_len - window_size:, :]
        # Create the data sets x_test and y_test
        self.x_test = []
        for i in range(window_size, len(test_data)):
            self.x_test.append(test_data[i-window_size:i, 0])

        # Convert the data to a numpy array
        self.x_test = np.array(self.x_test)

        # Reshape the data
        self.x_test = np.reshape(
            self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def run(self):

        saved_model = Path(self.model_path)
        if not saved_model.is_file():
            # initializing the RNN
            regressor = Sequential()

            # adding first RNN layer and dropout regulatization
            regressor.add(SimpleRNN(
                units=50,
                activation="tanh",
                return_sequences=True,
                input_shape=(self.x_train.shape[1], 1))
            )

            regressor.add(Dropout(0.2))
            # adding second RNN layer and dropout regulatization

            regressor.add(
                SimpleRNN(units=50, activation="tanh", return_sequences=True))

            regressor.add(Dropout(0.2))

            # adding third RNN layer and dropout regulatization

            regressor.add(
                SimpleRNN(units=50, activation="tanh", return_sequences=True))

            regressor.add(Dropout(0.2))
            # adding fourth RNN layer and dropout regulatization
            regressor.add(
                SimpleRNN(units=50)
            )

            regressor.add(
                Dropout(0.2)
            )

            # adding the output layer
            regressor.add(Dense(units=1))

            # compiling RNN
            regressor.compile(
                optimizer="adam",
                loss="mean_squared_error",
                metrics=["accuracy"])

            regressor.save('rnn_model.h5')
        else:
            regressor = load_model('rnn_model.h5')

        y_pred = regressor.predict(self.x_test)  # predictions
        y_pred = self.scaler.inverse_transform(
            y_pred)  # scaling back from 0-1 to original
        return y_pred


class LSTMPredictionModel:

    def __init__(self, data, target='Close', test_length=PERIODS, window_size=60):
        self.target_column = target
        data = data.copy()
        self.model_path = 'lstm_model.h5'
        dataset = data.values
        # Get the number of rows to train the model on
        training_data_len = len(dataset) - test_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(dataset)

        train_data = scaled_data[0:int(training_data_len), :]
        # Split the data into x_train and y_train data sets
        self.x_train = []
        self.y_train = []

        for i in range(window_size, len(train_data)):
            self.x_train.append(train_data[i-window_size:i, 0])
            self.y_train.append(train_data[i, 0])

        # Convert the x_train and y_train to numpy arrays
        self.x_train, self.y_train = np.array(
            self.x_train), np.array(self.y_train)

        # Reshape the data
        self.x_train = np.reshape(
            self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

        test_data = scaled_data[training_data_len - window_size:, :]
        # Create the data sets x_test and y_test
        self.x_test = []
        for i in range(window_size, len(test_data)):
            self.x_test.append(test_data[i-window_size:i, 0])

        # Convert the data to a numpy array
        self.x_test = np.array(self.x_test)

        # Reshape the data
        self.x_test = np.reshape(
            self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))

    def run(self):
        saved_model = Path(self.model_path)
        if not saved_model.is_file():
            model = Sequential()
            model.add(LSTM(128, return_sequences=True,
                           input_shape=(self.x_train.shape[1], 1)))
            model.add(LSTM(64, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(self.x_train, self.y_train, batch_size=1, epochs=1)
        else:
            model = load_model(self.model_path)

        # Get the models predicted price values
        predictions = model.predict(self.x_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions


class XGBPredictionModel:
    def __init__(self, data, target='Close',  window_size=PERIODS):
       

        self.target_column = target
        self.model_path = 'xgb_model.sav'


        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'])
        data.index = range(len(data))

        data = data.filter(['Date', self.target_column])
        # features 
        data['EMA_9'] = data[self.target_column].ewm(9).mean().shift()
        data['SMA_5'] = data[self.target_column].rolling(5).mean().shift()
        data['SMA_10'] = data[self.target_column].rolling(10).mean().shift()
        data['SMA_15'] = data[self.target_column].rolling(15).mean().shift()
        data['SMA_30'] = data[self.target_column].rolling(30).mean().shift()
        EMA_12 = pd.Series(data[self.target_column].ewm(span=12, min_periods=12).mean())
        EMA_26 = pd.Series(data[self.target_column].ewm(span=26, min_periods=26).mean())
        data['MACD'] = pd.Series(EMA_12 - EMA_26)
        data['MACD_signal'] = pd.Series(data.MACD.ewm(span=9, min_periods=9).mean())

        data[self.target_column] = data[self.target_column].shift(-1)
        data = data.iloc[33:] # Because of moving averages and MACD line
        data = data[:-1]      # Because of shifting close price

        data.index = range(len(data))

        data.drop(['Date'],inplace=True,axis=1)

        
        # train test split indexes
        test_size  = (window_size+1) / len(data)
        valid_size = 0.15

        test_split_idx  = int(data.shape[0] * (1-test_size))
        valid_split_idx = int(data.shape[0] * (1-(valid_size+test_size)))  

        
        #train test split tcs

        train= data.loc[:valid_split_idx]
        valid= data.loc[valid_split_idx+1:test_split_idx]
        test= data.loc[test_split_idx+1:]

        self.y_train = train[self.target_column]
        self.X_train = train.drop([self.target_column], 1)

        self.y_valid = valid[self.target_column]
        self.X_valid = valid.drop([self.target_column], 1)

        self.y_test = test[self.target_column]
        self.X_test = test.drop([self.target_column], 1)

    def run(self):
        # from numpy import asarray
        saved_model = Path(self.model_path)
        if not saved_model.is_file():

            parameters = {'gamma': 0.01, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 400, 'random_state': 42}
            eval_set = [(self.X_train, self.y_train), (self.X_valid, self.y_valid)]
            self.model = xgb.XGBRegressor(**parameters, objective='reg:squarederror')
            self.model.fit(self.X_train, self.y_train, eval_set=eval_set, verbose=False)
            pickle.dump(self.model, open(self.model_path, "wb"))
          
        else:
            self.model = pickle.load(open(self.model_path, "rb"))
        y_pred = self.model.predict(self.X_test)  # predictions
        return y_pred


@st.cache(allow_output_mutation=True)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)

    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data['ROC'] = ((data['Close'] - data['Close'].shift(PERIODS))/(data['Close'].shift(PERIODS)))*100
data = data.fillna(0)




if selected_type == 'Close Price':
    dataset = data.filter(['Close'])
elif selected_type == 'ROC':
    dataset = data.filter(['ROC'])

if selected_model == 'LSTM':
    model = LSTMPredictionModel(dataset)
    predictions = model.run()
elif selected_model == 'RNN':
    model = RNNPredictionModel(dataset)
    predictions = model.run()
elif selected_model == 'XGB':
    target = 'Close' if selected_type == 'Close Price' else 'ROC'
    model = XGBPredictionModel(data, target=target)
    predictions = model.run()
   



def plot_data():
    if selected_type == 'ROC':
        y_col = 'ROC'
    else:
        y_col = 'Close'
    
    results = make_future_frames(data, target=y_col)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=results['Date'],
                y=results[y_col],
                name='Actual',
                marker=dict(
                    color='#40ad87',
                    size=5,
                    line=dict(
                        color='#848ad9',
                        width=1
                    ),
                ),

            ),
            go.Scatter(
                # mode='markers',
                x=results['Date'],
                y=results['Predictions'],
                name='Predict',
                marker=dict(
                    color='#303cdb',
                    size=5,
                    line=dict(
                        color='#848ad9',
                        width=1
                    ),
                ),
            )
        ]
    )

    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    fig.update_xaxes(showline=True, linewidth=1,
                     linecolor='black', gridcolor='#333333')
    fig.update_yaxes(showline=True, linewidth=1,
                     linecolor='black', gridcolor='#333333')
    fig.update_layout(width=int(900),   margin=dict(l=5, r=5, t=50, b=5),)
    st.plotly_chart(fig)


plot_data()


