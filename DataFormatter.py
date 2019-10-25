from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Author: Tyler Arndt <tarndt1@luc.edu>


class DataFormatter(object):
    """Formatting CSV file for Logistic Regression.

    Parameters
    -----------
    file_name : csv file
        Data used for fitting
    """

    def __init__(self, file_name):
        self.file_name = file_name

    def format_data(self):
        """Format the raw data for training, developing, and testing the algorithms

        Returns
        -------
        X_train_std : array of shape [n_samples, n_features]
            Feature scaled data to be used for training algorithms

        y_train: array of shape [n_targets]
            Target results of the training data

        X_test_std : array of shape [n_samples, n_features]
            Feature scaled data to be used for testing algorithms

        y_test: array of shape [n_targets]
            Target results of the testing data

        X_dev_std : array of shape [n_samples, n_features]
            Feature scaled data to be used for tweaking hyperparameters

        y_dev: array of shape [n_targets]
            Target results of the development data
        """

        df = pd.read_csv(self.file_name)

        # reformat column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

        # separate features and targets
        X = df.drop(columns=['name', 'fgm', 'fga', 'fg%', '3p_made', '3pa', '3p%', 'ftm', 'fta', 'ft%', 'oreb', 'dreb', 'target_5yrs'])
        y = df.target_5yrs

        # find average values for NaN replacement
        values = {'gp': X['gp'].mean(),
                  'min': X['min'].mean(),
                  'pts': X['pts'].mean(),
                  'reb': X['reb'].mean(),
                  'ast': X['ast'].mean(),
                  'stl': X['stl'].mean(),
                  'blk': X['blk'].mean(),
                  'tov': X['tov'].mean()}

        X = X.fillna(value=values)

        # split data into 70% train and 30% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1, stratify=y)

        # split test into development and test (15% of total data set, each)
        X_test, X_dev, y_test, y_dev = train_test_split(
            X_test, y_test, test_size=0.5, random_state=1, stratify=y_test)

        # standardization features scaling
        X_test = feature_scaling(X_test)
        X_train = feature_scaling(X_train)
        X_dev = feature_scaling(X_dev)

        return X_train, y_train, X_test, y_test, X_dev, y_dev


def feature_scaling(X):
    """Scale the features with standardization

    Parameters
    -----------
    X : array of shape [n_samples, n_features]
        Data to be scaled

    Returns
    -------
    X_std : array of shape [n_samples, n_features]
        Scaled data
    """

    X = np.copy(X)
    X_std = np.copy(X)

    for i in range(len(X_std[0])):
        X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    return X_std
