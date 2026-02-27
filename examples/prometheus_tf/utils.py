import os
import shutil
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# DataConsolidation
# Manages operational datasets, splitting them into iterative training folders.
# Analogous to the dataset-download / split helpers in mnist/utils.py.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Standalone data utilities
# Extracted from ModelGeneration so they can be imported independently,
# analogous to the transform/loader helpers in mnist/utils.py.
# ---------------------------------------------------------------------------

def normalize(df):
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


def unnorm(df, bat, target_meta, target_variable):
    """
    Unnormalize the standard normalized dataframe for proper model evaluation.

    Parameters
    ----------
    df: dataframe
        Dataframe with normalized columns.
    bat: int
        Current batch index. When 0, uses only first-batch statistics.
    target_meta: dataframe
        Per-batch statistics built by calculate_target_meta().
    target_variable: list[str]
        Names of the target columns.

    Returns
    -------
    df: dataframe
        Dataframe with values unnormalized.
    """

    if bat == 0: # need to drive mu and sigma based on one batch of data only
        target_meta_filtered = target_meta[target_meta['bc'] == 0]
        for item in target_meta_filtered['tv'].unique():
            mu = target_meta_filtered[target_meta_filtered['tv'] == item]['mu'].iloc[0]
            std = target_meta_filtered[target_meta_filtered['tv'] == item]['std'].iloc[0]
            df[item] = df[item] * std + mu
    else:
        # calculate the weighted mu with all batches of data
        weighted_values = {'muc':[], 'tv':[]}
        for item in target_variable:
            weighted_values['muc'].append(target_meta.groupby(['tv'])['muk_nk'].sum()[item]/target_meta.groupby(['tv'])['bs'].sum()[item])
            weighted_values['tv'].append(item)
        weighted_values = pd.DataFrame(weighted_values)

        # initialize muc values
        if 'muc' not in target_meta.columns:
             target_meta.loc[:, 'muc'] = 0.0
        # add muc values to target_meta
        for index, value in target_meta['tv'].items():
            if value in target_variable:
                source_value_scalar = weighted_values[weighted_values['tv'] == value]['muc'].iloc[0]
                target_meta.loc[index, 'muc'] += source_value_scalar

        weighted_values.loc[:, 'stdc'] = 0.0

        # add s1 dataset
        target_meta['s2'] = target_meta['bs'] * (target_meta['mu'] - target_meta['muc'])**2

        for index, value in weighted_values['tv'].items():
            weighted_std = np.sqrt((target_meta.groupby(['tv'])['s1'].sum()[value] + target_meta.groupby(['tv'])['s2'].sum()[value])/(target_meta.groupby(['tv'])['bs'].sum()[value] - 1))
            weighted_values.loc[index, 'stdc'] += weighted_std

        for item in df.columns:
            df[item] = df[item] * weighted_values[weighted_values['tv'] == item]['stdc'].item() + weighted_values[weighted_values['tv'] == item]['muc'].item()

    return df


def read_dataset(path_to_data=None):
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


def create_feature_target_lists(dfs, features_variables, target_variable, sequence_length, norm=True):
    """
    Creates arrays for feature/target variables in a readable form for the LSTM model.

    Parameters
    ----------
    dfs: list[dataframes]
        List of dataframes to be read and manipulated according to the given sequence length.
    features_variables: list[str]
        Input feature column names.
    target_variable: list[str]
        Target column names.
    sequence_length: int
        Number of timesteps per input window.
    norm: bool
        If True, apply standard normalization before windowing.

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
                data = normalize(data) # normalized rho provide inf values, need to figure out how we can mitigate this effect, may be by utilizing the power data directly
            # Select target and features
            feature_data = data[features_variables] # panda framework with input feature columns
            target_data = data[target_variable] # panda framework with target variable columns
            x = []
            y = []
            # number of rows for len(target_data)
            for i in range(len(target_data) - sequence_length): # +1 will capture add the last point, but for the predicted output, we only have 19 smaples excluding the initial 30 datapoints.
                x.append(feature_data[i:i+sequence_length])
                y.append(target_data[i:i+sequence_length])
            x_list.append(np.array(x))
            y_list.append(np.array(y))

    return x_list, y_list


def calculate_target_meta(trainingdfs, target_variable):
    """
    Calculates per-batch mean, std, batch size, and other statistics for each
    target variable across all training DataFrames.

    Parameters
    ----------
    trainingdfs: list[dataframe]
        List of training DataFrames.
    target_variable: list[str]
        Target column names.

    Returns
    -------
    target_meta: dataframe
        DataFrame containing per-batch statistics (mu, std, bs, bc, tv, muk_nk, s1).
    """
    count = 0 # starting with 0, just to be consistent with indexing
    mu = []
    std = []
    bs = []
    bc = []
    tv = []
    muk_nk = []
    s1 = []

    for df in trainingdfs:
        for variable in target_variable:
            mu.append(df[variable].mean())
            std.append(df[variable].std())
            bs.append(len(df[variable]))
            bc.append(count)
            tv.append(variable)
            muk_nk.append(df[variable].mean() * len(df[variable]))
            s1.append((len(df[variable])-1)*df[variable].std()**2)
        count = count + 1

    target_meta = {}
    target_meta['mu'] = mu
    target_meta['std'] = std
    target_meta['bs'] = bs
    target_meta['bc'] = bc
    target_meta['tv'] = tv
    target_meta['muk_nk'] = muk_nk
    target_meta['s1'] = s1

    return pd.DataFrame(target_meta)
