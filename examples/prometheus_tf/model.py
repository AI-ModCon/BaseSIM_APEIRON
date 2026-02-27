import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.utils import plot_model

from .utils import (
    calculate_target_meta,
    create_feature_target_lists,
    read_dataset,
    unnorm,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# build_model
# The Stacked LSTM architecture extracted as a standalone factory function,
# analogous to the Cnn class in mnist/model.py.
# ---------------------------------------------------------------------------

def build_model(sequence_length, n_features, n_targets):
    """
    Build and compile the Stacked LSTM model.

    Architecture:
        - LSTM(64, tanh, return_sequences=True)
        - LSTM(64, tanh)
        - Dropout(0.1)
        - Dense(32, tanh)
        - Dense(n_targets, sigmoid if single target else linear)

    Parameters
    ----------
    sequence_length : int
        Number of timesteps per input window.
    n_features : int
        Number of input feature columns.
    n_targets : int
        Number of target columns to predict.

    Returns
    -------
    model : tf.keras.Sequential
        Compiled Keras model ready for training.
    """
    model = tf.keras.models.Sequential() # allows layers to be added in a linear fashion
    model.add(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(sequence_length, n_features))) # number of neurons in the LSTM layer, tanh: hyperbolic tangent activation function applied to the output of the LSTM units, 'return_sequences=True' returns the full sequence of outputs for each input, rather than just the last output, 'sequence_length' represents the number of time steps in each input sequence, 'features_variables' represents the number of features at each time step
    model.add(tf.keras.layers.LSTM(64, activation='tanh')) # No return_sequences needed for the last LSTM layer, 2nd layer
    #model.add(Attention(return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.1))  # Dropout for regularization, 10% of the neurons dropped out
    model.add(tf.keras.layers.Dense(32, activation='tanh'))  # Additional Dense layer with some non-linearity

    if n_targets == 1:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Final output layer for predicting 3 features
    else:
        model.add(tf.keras.layers.Dense(n_targets, activation=None)) # output with no specific range restriction, linear activateion can be used.

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # optimizers using a moving average of past gradients prevent the learning rate from decaying too quickly
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001) # Adagrad accumulates the square of all past gradients
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    #model.compile(optimizer='Adagrad', loss=tf.keras.losses.MeanSquaredError()) # MSE = loss function with a sigmoid output layer is typical for regression problems when predicting a continuous value between 0 and 1
    return model


# ---------------------------------------------------------------------------
# ModelGeneration
# Training and evaluation harness, analogous to MNIST_CNN in mnist/model.py.
# Data utilities are imported from utils.py; the model is built via build_model().
# ---------------------------------------------------------------------------

class ModelGeneration(object):

    '''
    A class for data preprocessing and training of models to predict operations of the AGN-201 Reactor.
    ML approach uses a Stacked LSTM (multiple layers of LSTM cells stacked vertically on top of each other) Architecture for mirrored depth to Bi-LSTM Architecture.
    '''

    def __init__(self, training_path = None, testing_path = None, model_path = None, image_path = None, sequence_length = 30, epochs = 50, batch_size = 32, feature_variables=[], target_variable = [], text_file_name = None):

        '''
        Initializes the AGN201_ML Object

        Parameters
        ----------
            df : dataframe
                Input dataframe.
            training_path: str
                File path for folder containing training data.
            testing_path: str
                File path for folder containing testing data.
            model_path: str
                File path where serialized LSTM model will be saved to.
            image_path: str
                File path where prediction vs actual image will be saved to.
            sequence_length: int
                Sequence length controlling the length of the LSTMs hidden memory.
            epochs: int
                Number of epochs which the LSTM will train.
            batch_size: int
                Size of data batches split into LSTM inputs.
            target_variable: int
                Target variable which LSTM will predict.
            text_file_name: str
                Name of text file where all the ML model metrics are output to.
        '''

        # Public Variables
        self.model_path = model_path
        self.training_path = training_path
        self.testing_path = testing_path
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_variable = target_variable
        self.image_path = image_path
        self.text_file_name = text_file_name
        self.features_variables = feature_variables # input variables
        self.all_variables = feature_variables + target_variable

        # Private Variables
        self.model = None
        self.target_meta = {}
        self.trainingdfs = None
        self.testdfs = None
        self.features = []
        self.x_list = [] # train
        self.y_list = []
        self.x1_list = [] # test
        self.y1_list = []
        self.test_target_variable = []

    def data_preprocessing(self):
        """
        Preprocesses feature and target variables for both training and testing datasets.
        Removes all columns except for: Rho, CCR, FCR, Inverse Period.
        Internally saves all arrays to be passed through the LSTM model.
        """

        print('Preprocessing Data')
        self.trainingdfs = read_dataset(self.training_path) # list of panda frameworks
        self.testdfs = read_dataset(self.testing_path)

        # keeping only variables needed.
        for num, df in enumerate(self.trainingdfs):
            self.trainingdfs[num] = df.filter(items=self.all_variables)

        for num, df in enumerate(self.testdfs): # num = index in dfs
            self.testdfs[num] = df.filter(items=self.all_variables) # only keep the values with column names = all_variables

        print(f'Expected Feature Variables: {self.features_variables}\nExpected Target Variables: {self.target_variable}\nVariables In RNN: {list(self.trainingdfs[0].columns)}') # since all training df variables are set with self.all_variables only. we can use df[0].columns reading.
        if len(self.all_variables) != len(self.trainingdfs[0].columns):
            raise ValueError(f'Length of all Target + Feature Variables ({len(self.all_variables)}) does not equal length of training array: {len(self.trainingdfs[0].columns)}. Check column names in data files.')

        print('-----------')

        self.target_meta = calculate_target_meta(self.trainingdfs, self.target_variable)

        self.x_list, self.y_list = create_feature_target_lists(self.trainingdfs, self.features_variables, self.target_variable, self.sequence_length) # normalized values verified.
        self.x1_list, self.y1_list = create_feature_target_lists(self.testdfs, self.features_variables, self.target_variable, self.sequence_length)

    def repeated_trainer(self):
        """
        Initiates Stacked LSTM architecture as follows:
            - 64 LSTM units
            - 2nd 64 LSTM unit layer
            - Dropout layer to prevent overfitting (10% of neurons)
            - 32 Dense units to condense neurons and add non-linearity
            - 1 Dense unit to produce final output

        Repeatedly trains LSTM model according to the number of datasets.
        Final model weights saved as a keras file.
        """

        self.model = build_model(self.sequence_length, len(self.features_variables), len(self.target_variable))
        plot_model(self.model, to_file='./output/_images/model_architecture.png', show_shapes=True, expand_nested=True)


        if np.isnan(self.x_list).any() or np.isnan(self.y_list).any():
            print(f"NaN values found in data for case {d}. Skipping or cleaning data.")
        else:
            None # apply the data cleaning function if needed for sorting NaN values.

        for d in range(len(self.x_list)):
            print(f'Running case {d}')
            train_x = self.x_list[d]
            train_y = self.y_list[d]
            train_y_for_fit = train_y[:, -1, :] # taking the last data point

            # Set up early stopping
            early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # monitoring the training loss can sometimes stop training too early, better to monitor the validation loss; 'val_loss' vs. 'loss'. It tracks the validation loss of the model after each epoch, if the validation loss does not decrease after 10 consecutive epochs, the training process will be halted. The model will revert to the weights that yielded the lowest validation loss during the entire training process, retaining the best-performing model

            weights_before = self.model.get_weights()
            start_time = time.time()
            # Fit the model
            self.model.fit(train_x, train_y_for_fit, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, callbacks=[early_stopper])
            end_time = time.time()
            self.total_time = end_time - start_time
            print('-----------')
            weights_after = self.model.get_weights()

            ## Testing the weight changes per fit
            weights_changed = False
            for i, (w_before, w_after) in enumerate(zip(weights_before, weights_after)):
                if not np.array_equal(w_before, w_after):
                    print(f"Layer {i} weights have changed.")
                    weights_changed = True
                else:
                    print(f"Layer {i} weights are identical.")

            if not weights_changed:
                print("\nCONCLUSION: All model weights remained identical after training.")
            else:
                print("\nCONCLUSION: Some weights were updated.")

            are_close = np.allclose(weights_before[0][0][:5], weights_after[0][0][:5], atol=1e-4) # Adjust tolerance as needed

            if are_close:
                print("The weights are effectively the same (within tolerance).")
            else:
                print("The weights have a noticeable difference.")

            ## Evaluate the model after each training batches
            self.evaluate_model(d)
            print('-----------')


        self.model.save(self.model_path)

    def evaluate_model(self, bat):
        """
        Evaluate model using the following metrics:
            - R2
            - Mean Absolute Error (MAE)
            - Mean Absolute Percent Error (MAPE)
            - Training Time (for computational efficiency evaluations)
        """

        forecast_lstm = self.model.predict(self.x1_list) # generate predictions for the output with input test sequences
        forecast_lstm = pd.DataFrame(forecast_lstm)
        forecast_lstm.columns = self.target_variable
        forecast_lstm = unnorm(forecast_lstm, bat, self.target_meta, self.target_variable) # retrieve the raw data, bat is the batch data

        count = 0
        dfs = read_dataset(self.testing_path)
        for df in dfs:
            df = df.drop(df.index[:self.sequence_length], axis=0)
            df = df[self.target_variable]
            df.reindex()

            # Test R2 and MAE of the model
            forecast_lstm.to_csv("./output/_temp/prediction.csv")
            print("EVALUATION:")
            R2 = r2_score(df, forecast_lstm)
            MAE = mean_absolute_error(df, forecast_lstm)
            MAPE = mean_absolute_percentage_error(df, forecast_lstm)

            print(f"R2: {R2}")
            print(f"MAE: {MAE} $")
            print(f"MAPE: {MAPE} %")

            text_file = open(self.text_file_name, "a")
            text_file.write(f"Rounds of Training sets: {bat} \n")
            text_file.write(f"Rounds of Test sets: {count} \n")
            text_file.write(f"R2: {R2} \n")
            text_file.write(f"MAE: {MAE} $ \n")
            text_file.write(f"MAPE: {MAPE} % \n")
            text_file.write(f"Total Training Time: {self.total_time} seconds \n")
            text_file.close()

            count = count + 1
