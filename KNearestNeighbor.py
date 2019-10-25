import numpy as np
import math

# Author: Tyler Arndt <tarndt1@luc.edu>


class KNearestNeighbor(object):
    def __init__(self, X, y, k=5):
        """
        Parameters
        -----------
        X: array of shape [n_samples, n_features]
            Training data

        y: array of shape [n_samples]
            Training data targets

        k: integer
            Neighbors to be considered
        """

        self.X = X
        self.y = y
        self.k = k

    def predict(self, X_test):
        """
        Parameters
        -----------
        X_test: array of shape [n_samples, n_features]
            Testing data

        Returns
        -------
        prediction_arr: array of shape [n_predictions]
            Predictions on the testing data
        """

        prediction_arr = []
        for i in range(len(X_test)):
            distance_arr = []
            for j in range(len(self.X)):
                distance_arr.append(distance(X_test[i], self.X[j]))

            # sort target array based on the values in the distance array
            sorted_target_arr = [x for _, x in sorted(zip(distance_arr, self.y))]

            # take k-smallest values from sorted target array
            sorted_target_arr = sorted_target_arr[:self.k]

            # turn into np array to easily get the number of 0 and 1 labels
            sorted_target_arr = np.array(sorted_target_arr)
            unique_elements, element_counts = np.unique(sorted_target_arr, return_counts=True)

            if len(element_counts) == 1:
                prediction = sorted_target_arr[0]
            else:
                if element_counts[0] > element_counts[1]:
                    prediction = 0.0
                elif element_counts[0] < element_counts[1]:
                    prediction = 1.0
                else:
                    # tie goes to the closest neighbor
                    prediction = sorted_target_arr[0]

            prediction_arr.append(prediction)

        return prediction_arr


def distance(sample1, sample2):
    """
    Euclidean distance measure between two samples in the dataset

    Returns
    -------
    dist: float distance value
    """
    sum = 0
    for i in range(len(sample1)):
        sum += (sample1[i] - sample2[i])**2
    dist = math.sqrt(sum)
    return dist
