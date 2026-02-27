from matplotlib.pylab import f
import pandas as pd
import numpy as np
import warnings
import os
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
import shutil
from tensorflow.keras.utils import plot_model
warnings.filterwarnings("ignore") # ignore all warning messages

class DataConsolidation(object):
    '''
    A class to take operational datasets split then for additional training of LSTM to modify the RNN's weights.
    '''
    def __init__(self, oper_data_path: str = None, output_folder: str = None):
        '''
        Initializes the Dataset_Gen object.

        Parameters
        ----------
            oper_data_path : str
                Folder path containing all the operational data to be used for training.
            output_folder : str
                Folder to store all of the datasets created.
        '''

        self.oper_data_path = oper_data_path # input path
        self.output_folder = output_folder # output path

    def read_dataset(self, path_to_data):
        """
        Reads csv files from a given folder location containing data.

        Parameters
        ----------
        path_to_data: str
            Folder location containing data.
        
        Returns
        -------
        files: list[str]
            List of file paths read from the given folder location.
        """
        return [os.path.join(path_to_data, file) for file in os.listdir(path_to_data) if file.endswith(".csv")]

    def generate_data_folders(self, iteration_percent: float = 0.10): # %VERIFY, Need to fix

        """
        Generates folders containing synthetic and operational datasets for iterative training.

        Parameters
        ----------
        iteration_percent : float, optional
            The percentage of the total operational datasets to be used for each iteration step (default is 0.10).

        Returns
        -------
        folder_amount : float
            The total number of folders created.
        iteration_amount : int
            The number of operational datasets used for each iteration step.
        """
        operational_files = self.read_dataset(self.oper_data_path)
        iteration_amount = int(len(operational_files) * iteration_percent) # iteration_amount is the chunk size

        for i in range(0, len(operational_files) + 1, iteration_amount):
            # Create new folder for this iteration
            iter_folder = os.path.join(self.output_folder, f'operational_data_{i}')
            os.makedirs(iter_folder, exist_ok=True) # Make a directory if it does not exist. 

            # Copy operational data up to the current iteration to iteration folder
            for file in operational_files[(i-iteration_amount):i]:
                shutil.copy(file, iter_folder)

            print(f"Created folder {iter_folder} with {min(i, len(operational_files))} operational datasets.")

        folder_amount = round(int((len(operational_files) + 1)/iteration_amount))
        return round(folder_amount), iteration_amount
    
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

    def normalize (self, df):
        """
        Perform standard normalization on all dataframe columns.

        Parameters
        ----------
        df: dataframe
            Dataframe whose columns are to be normalized.
        
        Returns
        -------
        df: dataframe
            Dataframe with all variables normalized.

        """
        for item in df:
            mean = df[item].mean() # calclulate mean for each variable (each column)
            stdev = df[item].std() # calculate standard deviation for each variable
            df[item] = (df[item] - mean) / stdev # Normalize data w/ [(x - mean)/standard deviation
                    
        return df

    def unnorm(self, df, bat):
        """
        Unnormalize the standard normalized dataframe for proper model evaluation.

        Parameters
        ----------
        df: dataframe
            Dataframe with normalized columns.

        Returns
        -------
        df: dataframe
            Dataframe with values unnormalized.
        """
        
        if bat == 0: # need to drive mu and sigma based on one batch of data only
            target_meta_filtered = self.target_meta[self.target_meta['bc'] == 0]
            for item in target_meta_filtered['tv'].unique(): 
                mu = target_meta_filtered[target_meta_filtered['tv'] == item]['mu'].iloc[0]
                std = target_meta_filtered[target_meta_filtered['tv'] == item]['std'].iloc[0]
                df[item] = df[item] * std + mu 
        else:
            # calculate the weighted mu with all batches of data
            weighted_values = {'muc':[], 'tv':[]}
            for item in self.target_variable: 
                weighted_values['muc'].append(self.target_meta.groupby(['tv'])['muk_nk'].sum()[item]/self.target_meta.groupby(['tv'])['bs'].sum()[item])
                weighted_values['tv'].append(item)
            weighted_values = pd.DataFrame(weighted_values)

            # initialize muc values 
            if 'muc' not in self.target_meta.columns:
                 self.target_meta.loc[:, 'muc'] = 0.0 
            # add muc values to target_meta
            for index, value in self.target_meta['tv'].items():
                if value in self.target_variable:
                    source_value_scalar = weighted_values[weighted_values['tv'] == value]['muc'].iloc[0]
                    self.target_meta.loc[index, 'muc'] += source_value_scalar
            
            weighted_values.loc[:, 'stdc'] = 0.0
    
            # add s1 dataset
            self.target_meta['s2'] = self.target_meta['bs'] * (self.target_meta['mu'] - self.target_meta['muc'])**2

            for index, value in weighted_values['tv'].items(): 
                weighted_std = np.sqrt((self.target_meta.groupby(['tv'])['s1'].sum()[value] + self.target_meta.groupby(['tv'])['s2'].sum()[value])/(self.target_meta.groupby(['tv'])['bs'].sum()[value] - 1))            
                weighted_values.loc[index, 'stdc'] += weighted_std

            for item in df.columns:
                df[item] = df[item] * weighted_values[weighted_values['tv'] == item]['stdc'].item() + weighted_values[weighted_values['tv'] == item]['muc'].item()

        return df 

    def read_dataset(self, path_to_data = None):
        """
        Reads csv file from a given folder location containing data.

        Parameters
        ----------
        path_to_data: str
            File location containing training or testing data.
        
        Returns
        -------
        dfs: list[dataframe]
            List of dataframes read from given folder location.
        """

        dirs = os.listdir(path_to_data)
        dirs.sort() # sort the list of the files alphabetically
        dfs = []
        for file in dirs:
            if file.endswith(".csv"):
                try:
                    training_df = pd.read_csv(os.path.join(path_to_data, file))
                    dfs.append(training_df) # appends panda frameworks
                except:
                    print(f'Failed to read: {file}')

        return dfs # reading all csv files in the desinated data directory.

    def create_feature_target_lists(self, dfs, norm=True):
        """
        Creates arrays for feature/target variables in a readable form for the LSTM model.

        Parameters
        ----------
        dfs: list[dataframes]
            List of dataframes to be read and manipulated according to the given sequence length.
        
        Return
        ------
        x_list: ndarray
            Array of feature variables readable by LSTM.
        y_list: ndarray
            Array of the target variable readable by the LSTM.
        """
        x_list = []
        y_list = []
        for n in range(len(dfs)): # iterate through the indices of a list 
            df = dfs[n]
            if(len(df.index) != 0):
                # normalize all of the data
                data = dfs[n]    
                if norm == True: 
                    data = self.normalize(data) # normalized rho provide inf values, need to figure out how we can mitigate this effect, may be by utilizing the power data directly
                # Select target and features
                feature_data = data[self.features_variables] # panda framework with input feature columns
                target_data = data[self.target_variable] # panda framework with target variable columns
                x = []
                y = []
                # number of rows for len(target_data)
                for i in range(len(target_data) - self.sequence_length): # +1 will capture add the last point, but for the predicted output, we only have 19 smaples excluding the initial 30 datapoints. 
                    x.append(feature_data[i:i+self.sequence_length])
                    y.append(target_data[i:i+self.sequence_length])
                x_list.append(np.array(x))
                y_list.append(np.array(y))
        
        return x_list, y_list

    def data_preprocessing(self):
        """
        Preprocesses feature and target variables for both training and testing datasets.
        Removes all columns except for: Rho, CCR, FCR, Inverse Period.
        Internally saves all arrays to be passed through the LSTM model.
        """

        print('Preprocessing Data')
        self.trainingdfs = self.read_dataset(self.training_path) # list of panda frameworks
        self.testdfs = self.read_dataset(self.testing_path) 

        # keeping only variables needed. 
        for num, df in enumerate(self.trainingdfs):
            self.trainingdfs[num] = df.filter(items=self.all_variables)
                                          
        for num, df in enumerate(self.testdfs): # num = index in dfs
            self.testdfs[num] = df.filter(items=self.all_variables) # only keep the values with column names = all_variables

        print(f'Expected Feature Variables: {self.features_variables}\nExpected Target Variables: {self.target_variable}\nVariables In RNN: {list(self.trainingdfs[0].columns)}') # since all training df variables are set with self.all_variables only. we can use df[0].columns reading.
        if len(self.all_variables) != len(self.trainingdfs[0].columns):
            raise ValueError(f'Length of all Target + Feature Variables ({len(self.all_variables)}) does not equal length of training array: {len(self.trainingdfs[0].columns)}. Check column names in data files.')

        print('-----------')

        self.calculate_target_meta()

        self.x_list, self.y_list = self.create_feature_target_lists(self.trainingdfs) # normalized values verified. 
        self.x1_list, self.y1_list = self.create_feature_target_lists(self.testdfs)

    

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

        self.model = tf.keras.models.Sequential() # allows layers to be added in a linear fashion
        self.model.add(tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True, input_shape=(self.sequence_length, len(self.features_variables)))) # number of neurons in the LSTM layer, tanh: hyperbolic tangent activation function applied to the output of the LSTM units, 'return_sequences=True' returns the full sequence of outputs for each input, rather than just the last output, 'sequence_length' represents the number of time steps in each input sequence, 'features_variables' represents the number of features at each time step
        self.model.add(tf.keras.layers.LSTM(64, activation='tanh')) # No return_sequences needed for the last LSTM layer, 2nd layer 
        #model.add(Attention(return_sequences=False))
        self.model.add(tf.keras.layers.Dropout(0.1))  # Dropout for regularization, 10% of the neurons dropped out
        self.model.add(tf.keras.layers.Dense(32, activation='tanh'))  # Additional Dense layer with some non-linearity
        
        if len(self.target_variable) == 1:
            self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Final output layer for predicting 3 features
        else: 
            self.model.add(tf.keras.layers.Dense(len(self.target_variable), activation=None)) # output with no specific range restriction, linear activateion can be used. 
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # optimizers using a moving average of past gradients prevent the learning rate from decaying too quickly
        #optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001) # Adagrad accumulates the square of all past gradients 
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError()) 
        #self.model.compile(optimizer='Adagrad', loss=tf.keras.losses.MeanSquaredError()) # MSE = loss function with a sigmoid output layer is typical for regression problems when predicting a continuous value between 0 and 1
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

    def calculate_target_meta(self):
        count = 0 # starting with 0, just to be consistent with indexing
        mu = [] 
        std = [] 
        bs = []
        bc = [] 
        tv = [] 
        muk_nk = []
        s1 = []

        for df in self.trainingdfs: 
            for variable in self.target_variable:
                mu.append(df[variable].mean())
                std.append(df[variable].std())
                bs.append(len(df[variable])) 
                bc.append(count)
                tv.append(variable)
                muk_nk.append(df[variable].mean() * len(df[variable]))
                s1.append((len(df[variable])-1)*df[variable].std()**2)
            count = count + 1 

        self.target_meta['mu'] = mu 
        self.target_meta['std'] = std 
        self.target_meta['bs'] = bs
        self.target_meta['bc'] = bc
        self.target_meta['tv'] = tv
        self.target_meta['muk_nk'] = muk_nk
        self.target_meta['s1'] = s1

        self.target_meta = pd.DataFrame(self.target_meta)

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
        forecast_lstm = self.unnorm(forecast_lstm, bat) # retrieve the raw data, bat is the batch data
        
        count = 0 
        dfs = self.read_dataset(self.testing_path)
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