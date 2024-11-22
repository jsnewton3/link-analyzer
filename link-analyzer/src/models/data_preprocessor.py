import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle


class DataPreprocessorUtils(object):
    def __init__(self,
                 window_len: int = 200,
                 predict_lag=10):
        self.win_len = window_len,
        self.lag = predict_lag

    # Windows the data based on the specified window length, and creates labels based on specified predictive lag
    @staticmethod
    def make_windowed_data(input_arr, window_len=200, lag=10):
        windows = []
        labels = []
        for i in range(len(input_arr) - (window_len + lag)):
            window = input_arr[i:i + window_len]
            label = input_arr[i + window_len + lag]
            windows.append(window)
            labels.append(label)
        return np.array(windows), np.array(labels)

    def scale_data(self, train, test):
        # Min-max scale data. The min-max scaler is calculated only on the train set and applied to train and test sets.
        # The scaling is done across all features and window samples and so the data is reshaped to a 2D array, normalized 
        # and then reshaped to a 3D array of shape (sample, sequence order, features)

        # Store the original shape so that we can revert after computation. The sequence length and number of feature are
        # the same for the test and train sets
        num_samples_train = train.shape[0]
        num_samples_test = test.shape[0]
        sequence_len = train.shape[1]
        num_feature = train.shape[2]

        # Reshaped window datq
        train = np.reshape(train, newshape=(num_samples_train * sequence_len, num_feature))
        test = np.reshape(test, newshape=(num_samples_test * sequence_len, num_feature))

        # Calculate the feature axis norms on train set and fit both train and test
        train, x_min, _x_max = self.min_max_scale(train)
        test, _, _ = self.min_max_scale(test, x_min, _x_max)

        # Reshape back to original 3D shapes.
        sl_train = np.reshape(train, newshape=(num_samples_train, sequence_len, num_feature))
        sl_test = np.reshape(test, newshape=(num_samples_test, sequence_len, num_feature))
        return sl_train, sl_test

    def scaled_data(self, train, test):
        """
        Min-max scale data.

        Args:
            train (numpy.ndarray): Training data with shape (num_samples_train, sequence_len, num_feature).
            test (numpy.ndarray): Test data with shape (num_samples_test, sequence_len, num_feature).

        Returns:
            tuple: Tuple containing scaled train and test sequences, x_min values, and x_max values.
        """

        # Store the original shape so that we can revert after computation.
        num_samples_train, sequence_len, num_feature = train.shape
        num_samples_test = test.shape[0]

        # Initialize arrays to store x_min and x_max values for each feature-sequence_len pair
        x_min_values = np.zeros((sequence_len, num_feature))
        x_max_values = np.zeros((sequence_len, num_feature))

        # Initialize lists to store scaled train and test sequences
        train_scaled = []
        test_scaled = []

        # Iterate over each feature-sequence length pair
        for i in range(num_feature):
            for j in range(sequence_len):
                # Get the data for the current feature-sequence length pair for both train and test sets
                train_data = train[:, j, i]
                test_data = test[:, j, i]

                # Calculate x_min and x_max for the current feature-sequence length pair
                x_min = np.min(train_data)
                x_max = np.max(train_data)

                # Store x_min and x_max values
                x_min_values[j, i] = x_min
                x_max_values[j, i] = x_max

                # Perform min-max scaling for the current feature-sequence length pair for both train and test sets
                train_scaled_data = (train_data - x_min) / (x_max - x_min)
                test_scaled_data = (test_data - x_min) / (x_max - x_min)

                # Append scaled data to the lists
                train_scaled.append(train_scaled_data)
                test_scaled.append(test_scaled_data)

        # Reshape the scaled train and test data
        train_scaled = np.array(train_scaled).reshape((num_samples_train, sequence_len, num_feature))
        test_scaled = np.array(test_scaled).reshape((num_samples_test, sequence_len, num_feature))

        return train_scaled, test_scaled, x_min_values, x_max_values




    def robust_scale(self, train, test):
        """
        Scales the train and test sequences using Robust Scaling.

        Args:
            train (numpy.ndarray): Training data with shape (num_samples_train, sequence_len, num_feature).
            test (numpy.ndarray): Test data with shape (num_samples_test, sequence_len, num_feature).

        Returns:
            tuple: Tuple containing scaled train and test sequences, along with the array of scalers.
        """
    
        # Store the original shape so that we can revert after computation.
        num_samples_train, sequence_len, num_feature = train.shape
        num_samples_test = test.shape[0]

        # Initialize array to store scalers
        scalers = np.empty((sequence_len, num_feature), dtype=object)

        # Perform Robust Scaling for each feature and sequence combination
        train_scaled = []
        test_scaled = []

        for i in range(sequence_len):
            for j in range(num_feature):
                # Fit the robust scaler to the slice of train set
                r_scaler = RobustScaler().fit(train[:, i, j].reshape(-1, 1))
                scalers[i, j] = r_scaler
                # Apply the robust scaler to both the train set slice and the corresponding test set slice
                train_scaled.append(r_scaler.transform(train[:, i, j].reshape(-1, 1)).reshape(-1))
                test_scaled.append(r_scaler.transform(test[:, i, j].reshape(-1, 1)).reshape(-1))

        # Reshape the scaled data
        sl_train = np.array(train_scaled).reshape((num_samples_train, sequence_len, num_feature))
        sl_test = np.array(test_scaled).reshape((num_samples_test, sequence_len, num_feature))

        return sl_train, sl_test, scalers



    def z_score_standardize(self, train, test):
        """
        Standardizes the train and test sequences using Z-Score standardization.

        Args:
            train (numpy.ndarray): Training data with shape (num_samples_train, sequence_len, num_feature).
            test (numpy.ndarray): Test data with shape (num_samples_test, sequence_len, num_feature).

        Returns:
            tuple: Tuple containing standardized train and test sequences, along with the array of scalers.
        """

        # Store the original shape so that we can revert after computation.
        num_samples_train, sequence_len, num_feature = train.shape
        num_samples_test = test.shape[0]

        # Initialize array to store scalers
        scalers = np.empty((sequence_len, num_feature), dtype=object)

        # Perform Z-Score standardization for each feature and sequence combination
        train_standardized = []
        test_standardized = []

        for i in range(sequence_len):
            for j in range(num_feature):
                # Fit the standard scaler to the slice of train set
                z_scaler = StandardScaler().fit(train[:, i, j].reshape(-1, 1))
                scalers[i, j] = z_scaler
                # Apply the standard scaler to both the train set slice and the corresponding test set slice
                train_standardized.append(z_scaler.transform(train[:, i, j].reshape(-1, 1)).reshape(-1))
                test_standardized.append(z_scaler.transform(test[:, i, j].reshape(-1, 1)).reshape(-1))

        # Reshape the standardized data
        sl_train = np.array(train_standardized).reshape((num_samples_train, sequence_len, num_feature))
        sl_test = np.array(test_standardized).reshape((num_samples_test, sequence_len, num_feature))

        return sl_train, sl_test, scalers


    # Function to normalize the input data
    def z_score_standardize2(self,input_data, scalers):
        # input_data shape: (1, window_size, 3)
        for t in range(input_data.shape[1]):  # Loop over the sequences
            for f in range(input_data.shape[2]):  # Loop over the features
                scaler = scalers[t, f]
                input_data[:, t, f] = scaler.transform(input_data[:, t, f].reshape(-1, 1)).reshape(-1)
        return input_data
    
        # Function to normalize the input data
    def robust_scale2(self,input_data, scalers):
        # input_data shape: (1, window_size, 3)
        for t in range(input_data.shape[1]):  # Loop over the sequences
            for f in range(input_data.shape[2]):  # Loop over the features
                scaler = scalers[t, f]
                input_data[:, t, f] = scaler.transform(input_data[:, t, f].reshape(-1, 1)).reshape(-1)
        return input_data

    # Function to perform min-max scaling on the input data
    def min_max_scale2(self,input_data, xmin, xmax):
        # input_data shape: (1, window_size, 3)
        for t in range(input_data.shape[1]):  # Loop over the sequences
            for f in range(input_data.shape[2]):  # Loop over the features
                min_val = xmin[t, f]
                max_val = xmax[t, f]
                input_data[:, t, f] = (input_data[:, t, f] - min_val) / (max_val - min_val)
        return input_data

    # Function to inverse z-score standardization on the input data
    def inverse_z_score_standardize2(self,normalized_data, scalers):
        # normalized_data shape: (1, window_size, 3)
        for t in range(normalized_data.shape[0]):  # Loop over the sequences
            for f in range(normalized_data.shape[1]):  # Loop over the features
                scaler = scalers[t, f]
                normalized_data[t, f] = scaler.inverse_transform(normalized_data[t, f].reshape(-1, 1)).reshape(-1)
        return normalized_data

    # Function to inverse min-max scaling on the input data
    def inverse_min_max_scale2(self,scaled_data, xmin, xmax):
        # scaled_data shape: (1, window_size, 3)
        for f in range(scaled_data.shape[1]):  # Loop over the features
                min_val = xmin[f]
                max_val = xmax[f]
                scaled_data[:, f] = scaled_data[:, f] * (max_val - min_val) + min_val
        return scaled_data

    def min_max_scaling(self, X):
        """
        Apply Min-Max Scaling to each feature of each sequence.
        """
        X_scaled = np.empty_like(X)
        scalers = {}
        for i in range(X.shape[0]):  # Iterate over each sequence
            scalers[i] = []
            for j in range(X.shape[2]):  # Iterate over each feature
                scaler = MinMaxScaler()
                X_scaled[i, :, j] = scaler.fit_transform(X[i, :, j].reshape(-1, 1)).reshape(-1)
                scalers[i].append(scaler)
        return X_scaled, scalers

    def robust_scaling(self, X):
        """
        Apply Robust Scaling to each feature of each sequence.
        """
        X_scaled = np.empty_like(X)
        scalers = {}
        for i in range(X.shape[0]):  # Iterate over each sequence
            scalers[i] = []
            for j in range(X.shape[2]):  # Iterate over each feature
                scaler = RobustScaler()
                X_scaled[i, :, j] = scaler.fit_transform(X[i, :, j].reshape(-1, 1)).reshape(-1)
                scalers[i].append(scaler)
        return X_scaled, scalers

    def save_scalers(self, scalers, filename):
        """
        Save scalers to a file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"Scalers saved to {filename}.")

    def min_max_above_threshold(self, array, thresholds):
        """
        Calculate the min and max values of each column of a numpy array, 
        considering only values above the specified minimum thresholds.

        Parameters:
        array (np.ndarray): Input numpy array.
        thresholds (list or np.ndarray): Minimum threshold values for each column.

        Returns:
        min_values (np.ndarray): Minimum values of each column above the threshold.
        max_values (np.ndarray): Maximum values of each column above the threshold.
        scaled_array (np.ndarray): Min-max scaled array using the calculated min and max values.
        """
        
        # Ensure the array and thresholds are numpy arrays
        array = np.array(array)
        thresholds = np.array(thresholds)

        # Initialize min and max values arrays
        min_values = np.full(array.shape[1], np.inf)
        max_values = np.full(array.shape[1], -np.inf)
        
        # Iterate over each column
        for col in range(array.shape[1]):
            # Filter values above the threshold
            filtered_values = array[array[:, col] > thresholds[col], col]
            
            # Calculate min and max for the filtered values
            if filtered_values.size > 0:
                min_values[col] = np.min(filtered_values)
                max_values[col] = np.max(filtered_values)
            else:
                min_values[col] = np.nan
                max_values[col] = np.nan

        # Min-max scaling
        scaled_array = np.copy(array)
        for col in range(array.shape[1]):
            if not np.isnan(min_values[col]) and not np.isnan(max_values[col]) and max_values[col] != min_values[col]:
                scaled_array[:, col] = (array[:, col] - min_values[col]) / (max_values[col] - min_values[col])
            else:
                scaled_array[:, col] = np.nan  # If min and max are the same, or if they are NaN, scale to NaN

        return scaled_array, min_values, max_values



    def plot_histograms_2d(self, data):


        # Plotting histograms
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        axs = np.ravel(axs)  # Ensure axs is an array of Axes objects

        for i in range(1):
            axs[i].hist(data[:, i], bins=300, color='blue', alpha=0.7)
            axs[i].set_title(f'Feature {i+1}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')
            axs[i].set_xlim([900000.0, 1200000.0])

        plt.tight_layout()
        plt.show()



    def plot_histograms_3d(self,data):
        """
        Plots histograms for each sequence-feature combination in the given numpy array and optionally sets axis limits.
    
        Parameters:
        data (numpy array): A numpy array of shape (100000, 100, 3)
        x_limits (tuple): A tuple of two elements specifying the x-axis limits (min, max) for all histograms.
        y_limits (tuple): A tuple of two elements specifying the y-axis limits (min, max) for all histograms.
        """
        num_sequences, num_timepoints, num_features = data.shape

        # Plotting histograms
        fig, axs = plt.subplots(num_features, 1, figsize=(10, 5 * num_features))
        axs = np.ravel(axs)  # Ensure axs is an array of Axes objects

        for i in range(num_features):
            feature_data = data[:, :, i].flatten()
            axs[i].hist(feature_data, bins=50, color='blue', alpha=0.7)
            axs[i].set_title(f'Feature {i+1}')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Frequency')


        plt.tight_layout()
        plt.show()



# Example usage:
# data = np.random.randn(100000, 3)
# plot_histograms(data)


    @staticmethod
    def min_max_scale(x, x_min=None, x_max=None):
        # Calc min/max is not provided
        if x_min is None or x_max is None:
            x_max = np.max(x)
            x_min = np.min(x)
        
        return (x - x_min)/(x_max - x_min), x_min, x_max




    @staticmethod
    def inverse_min_max_scale(scaled_x, x_min=None, x_max=None):
        if x_min is None or x_max is None:
            x_max = np.max(scaled_x)
            x_min = np.min(scaled_x)
        return scaled_x * (x_max - x_min) + x_min
