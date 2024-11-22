import numpy
import time
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from models.data_preprocessor import DataPreprocessorUtils
from tensorflow.keras.models import load_model




class Arima(object):
    def __init__(self, window, lag, p:int=None, d:int=None, q = None, p_value:float=.05, trend = 't'):
        '''

        Args:
            train_dat: nparray of shape (1,). The time series training data
            p: int Corresponds to the number of previous observations to be used in the repressors predictions.If none
            is provided a grid search will be used to determine the optimal value.
            q: int Window length used to calculate the moving averages and forecast residuals. If none is provided
            a grid search will be used to determine the optimal value.
            d: int Number of differencing periods. This is used to ensure stationary signals. If none is provided
            it will be incrementally increased until a test for a stationary signal is passed
            p_value: The degree of certainty required to determine if the training data is stationary. Lower values
            imply greater certainty. The default value of .05 corresponds to a 95% certainty threshold to determine the
            input is stationary.
        '''
        if None not in (p,q):
            self.p = p
            self.q = q
        else:
            # self.p, self.q = self.search_pq()
            p=1
            q=1
        self.window = window
        self.lag = lag
        self.d = d
        self.p_val = p_value
        self.train_dat = None
        self.diff_history = None
        self.trend = trend

        a=1
    @staticmethod
    def deserialize(json_string):
        try:
            freq_search_dict = json.load(json_string)
            arma = Arima(**freq_search_dict)
        except Exception as e:
            tb = e.__traceback__
            raise Exception("Error initializing Arima filter. Invalid config likely encountered").with_traceback(tb)
        return arma
    def train_fit(self, train_dat):
        if self.d is None:
            self.d, self.train_dat = self.find_d()
        self.arima_model = ARIMA(endog=self.train_dat[:,1], exog=train_dat[:, 0], trend=self.trend,
                                 order=(self.p, self.d, self.q))
        self.model_fit  = self.arima_model.fit()

    def predict(self, data_window):
        self.train_dat = data_window
        self.train_fit(data_window)
        predictions_x = np.array([])
        # predictions_y = np.array([])
        delta_t = np.mean(np.diff(data_window[:, 0]))
        time = []
        for i in range(0,self.lag):
            exog = data_window[len(data_window)-1, 0] + i*delta_t
            time.append(exog)
        prediction = self.model_fit.forecast(steps=self.lag, exog=np.array(time))
        predictions_x = np.concatenate((predictions_x, prediction))
        predict_data = np.stack((time, predictions_x)).transpose()
        return predict_data
    def find_d(self):
        d=0
        p=.9999
        dat = self.train_dat[:,1]
        while True:
            # Test if train time series is stationary. Difference the data until the test is passed
            test_results = adfuller(dat)
            p = test_results[1]
            if p <=self.p_val:
                break
            d+=1
            # We keep a history of the first elements in the intermediate differences to allow for the future
            # calculation of an inverse difference to get back the original data
            history = []
            for i in range(1,d+1):
                dat = np.diff(dat)
                history.append(dat[0])
            self.diff_history = history
        # fig, (original, dif) = plt.subplots(2)
        # original.set_title("Original")
        # dif.set_title("Diffrenced")
        # original.plot(self.train_dat[0], self.train_dat[1])
        # dif.plot(self.train_dat[0][d:], dat)
        # plt.show()
        return d, dat

    def search_pq(self):
        a=1

    # def predict(self, x):
    #     self.arima_model.forcast()
    #     Todo write equations

    @staticmethod
    def deserialize(config_dict):
        arma = Arima(**config_dict)
        return arma


class MovingAvarage(object):
    def __init__(self, window_len, destination_port):
        self.destination = destination_port
        self.len = window_len

    @staticmethod
    def deserialize(json_string):
        ma_params_dict = json.loads(json_string)
        ma = MovingAvarage(**ma_params_dict)
        return ma

    def transform(self, data):
        sma = np.sum(data)/len(data)
        return  sma


class Exponenetial_Moving_Average(object):
    def __init__(self, window, weighting):
        self.window = window
        self.weighting = weighting

    @staticmethod
    def deserialize(json_string):
        ema_params_dict = json.loads(json_string)
        ema = MovingAvarage(**ema_params_dict)
        return ema

    # def calculate_ema(self, data):
   #     Todo write equations

    @staticmethod
    def deserialize(config_dict):
        arma = Arima(**config_dict)
        return arma

class Lstm:
    def __init__(self, window):
        self.window = window
        self.model = load_model('/link-analyzer/models/inputs/lstm_model.keras')
        with open('/link-analyzer/models/inputs/scalers.pkl', 'rb') as file:
            self.scalers = pickle.load(file)
        self.minmax_data = np.load('/link-analyzer/models/inputs/xvalues.npz')
        self.minmax_label = np.load('/link-analyzer/models/inputs/xvalues_label.npz')
        self.processor = DataPreprocessorUtils()

    def predict(self, data_window):
        input_data = np.array(data_window).reshape((1, len(data_window), 1))  # Reshape for LSTM
        input_data = self.processor.min_max_scale2(input_data, self.minmax_data['x_min'], self.minmax_data['x_max'])
        input_data = self.processor.robust_scale2(input_data, self.scalers)  # Apply scaling
        prediction = self.model.predict(input_data)
        inversed_data = self.processor.inverse_min_max_scale2(
            prediction, 
            self.minmax_label['x_min_label'], 
            self.minmax_label['x_max_label']
        )
        return inversed_data[0, 0]

    # @staticmethod
    # def deserialize(json_string):
    #     print(json_string)
    #     lstm_params_dict = json.load(json_string)
    #     lstm = Lstm(**lstm_params_dict)
    #     return lstm
    
    @staticmethod
    def deserialize(config_dict):
        lstm = Lstm(**config_dict)
        return lstm



import numpy as np
import time
import random

def main():
    window_size = 200
    prediction_window = []
    window_dict = {}
    window_dict['window']=200
    window_json_str = json.dumps(window_dict)
    model = Lstm(window=window_json_str)  # Assuming lstm is your class

    while True:
        # Generate a random float between 30.0 and 50.0
        new_value = random.uniform(30.0, 50.0)
        print(new_value)
        
        # Append to the prediction window
        prediction_window.append(new_value)
        
        # Keep the window size fixed at 100 (sliding window)
        if len(prediction_window) > window_size:
            prediction_window.pop(0)
        
        # Once the window is full, make a prediction
        if len(prediction_window) == window_size:
            prediction = model.predict(prediction_window)
            print(prediction.shape)
            print(f"Prediction for current window: {prediction}")
        
        # Generate data at 8 Hz (125 ms per data point)
        time.sleep(1/8)

if __name__ == "__main__":
    main()






# if __name__ == "__main__":
#     main()




# if __name__ == "__main__":
#     data = np.load("../../../../test_ts.npy")
#     # n = 15
#     data = data.transpose()

#     predictions_x = np.array([])
#     predictions_y =np.array([])
#     window_len = 30
#     predict_lag= 1

#     #  #First order AR, filter noise show trend
#     # model = Arima(p=1, d=0, q=0, trend ='c' ) # 1st order AR model ok
#     # model = Arima(p=1, d=0, q=0, trend='c') # Random Walk ok
#     # model = Arima(p=1, d=1, q=0, trend='t') #Diffed 1st order AR meh
#     # model = Arima(p=0, d=1, q=1, trend='t') #Exp smoothing  good
#     model = Arima(p=1, d=0, q=0, trend='ct') # exp smothing with growth ok
#     # model = Arima(p=1, d=1, q=2, trend='t')


#     for i in range(0, 150 ):

#         if (i+window_len+predict_lag)>=len(data):
#             break
#         window = data[i:i+window_len]
#         start = time.time()

#         model.train_dat =window
#         exog = data[i:i+predict_lag, 0]

#         model.train_fit(window)
#         # if i>window_len:
#         prediction = model.model_fit.forecast(steps=predict_lag, exog = exog)
#         stop = time.time()
#         print(str(stop-start))
#         predictions_x = np.concatenate((predictions_x, prediction))
#         predictions_y = np.concatenate((predictions_y, exog))
#     figure, ax = plt.subplots()
#     ax.plot(data[:,0], data[:,1], 'b')
#     forcast_plot, = ax.plot(predictions_y, predictions_x, 'g')
#     plt.show()
