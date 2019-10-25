from sklearn.linear_model import LogisticRegression
from DataFormatter import DataFormatter
import matplotlib.pyplot as plt
from KNearestNeighbor import KNearestNeighbor

# Author: Tyler Arndt <tarndt1@luc.edu>

# format the data
data_formatter = DataFormatter("nba_logreg.csv")
X_train_std, y_train, X_test_std, y_test, X_dev_std, y_dev = data_formatter.format_data()

# avoid curse of dimensionality for K-nearest-neighbors
# minutes/game and points/game
X_train_knn = X_train_std[:, 1:3]
X_dev_knn = X_dev_std[:, 1:3]
X_test_knn = X_test_std[:, 1:3]


def test_algorithm(X, y, alg):
    """Test the accuracy of the algorithm

    Parameters
    -----------
    X : array of shape [n_samples, n_features]
        Data to be tested

    y : array of shape [n_samples]
        Targets of tested data

    alg : algorithm to be tested

    Returns
    -------
    mistakes : number of incorrect predictions by the algorithm
    """

    y_predict = alg.predict(X)
    mistakes = 0
    for i in range(len(y_predict)):
        if y_predict[i] != y.iloc[i]:
            mistakes += 1

    return mistakes


def hw_question_1():
    """Fit and run a scikit-learn implementation of a logistic regression algorithm
    """

    lr = LogisticRegression(penalty='l2', solver='lbfgs')
    lr.fit(X_train_std, y_train)

    print("QUESTION 1 - run Scikit-learn log-regression with default params on development set:")
    mistakes = test_algorithm(X_dev_std, y_dev, lr)
    print_results(y_dev, mistakes)


def hw_question_2():
    """Adjust hyperparameters to improve performance of the Scikit-learn logistic regression algorithm
    """

    c_arr = [1000, 100, 10, 1, 1/10, 1/100, 1/1000]

    mistakes_arr = []

    lr = LogisticRegression(random_state=1, solver='lbfgs')

    # increase regularization strength
    x_axis_placeholder = []
    for i in range(len(c_arr)):
        x_axis_placeholder.append(i)
        lr.C = c_arr[i]
        lr.fit(X_train_std, y_train)

        mistakes_arr.append(test_algorithm(X_dev_std, y_dev, lr))

    plt.plot(mistakes_arr, marker=".")
    plt.title("Prediction Errors over C value", fontsize=15)
    plt.xticks(x_axis_placeholder, c_arr)
    plt.xlabel("C Value")
    plt.ylabel("Number of Errors")
    plt.show()


def hw_question_3():
    """Run K-Nearest Neighbors algorithm.
       Adjust the n_neighbors value to improve performance.
    """
    neighbor_arr= [1, 5, 10, 25, 50, 75, 100, 250, 500, 750]
    mistakes_arr = []

    knn = KNearestNeighbor(X_train_knn, y_train)

    # adjust the 'neighbor' value
    x_axis_placeholder = []
    for i in range(len(neighbor_arr)):
        x_axis_placeholder.append(i)
        n_value = neighbor_arr[i]
        knn.k = n_value

        mistakes_arr.append(test_algorithm(X_dev_knn, y_dev, knn))

    plt.plot(mistakes_arr, marker=".")
    plt.title("Prediction Errors over Neighbor Values", fontsize=15)
    plt.xticks(x_axis_placeholder, neighbor_arr)
    plt.xlabel("Neighbor Values")
    plt.ylabel("Number of Errors")
    plt.show()


def hw_question_4():
    """Run Logistic Regression algorithm and KNN algorithm on the testing data.
    """

    lr = LogisticRegression(C=1, random_state=1, solver='lbfgs')
    lr.fit(X_train_std, y_train)

    knn = KNearestNeighbor(X_train_knn, y_train, 50)

    print("Question 4: Logistic Regression Results:")
    lr_mistakes = test_algorithm(X_test_std, y_test, lr)
    print_results(y_test, lr_mistakes)

    print("Question 4: K-Nearest Neighbors Results:")
    knn_mistakes = test_algorithm(X_test_knn, y_test, knn)
    print_results(y_test, knn_mistakes)


def print_results(y, mistakes):
    accuracy = (len(y)-mistakes) / len(y) * 100

    print("Mistakes: " + str(mistakes) + "\n" +
          "Predictions: " + str(len(y)) + "\n" +
          "Accuracy: " + str(accuracy) + "%\n")


hw_question_1()
hw_question_2()
hw_question_3()
hw_question_4()
