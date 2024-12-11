#################################
# Your name: Dor Liberman
#################################

import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    n, d = data.shape
    w = np.zeros(d)  # initialize weights to zero
    for t in range(1, T + 1):
        i = np.random.randint(0, n)  # randomly pick one data point
        eta_t = eta_0 / t  # decreasing learning rate
        y_i, x_i = labels[i], data[i]

        # Check the condition for the hinge loss update
        if y_i * np.dot(w, x_i) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i  # update rule when yi * <w, xi> < 1
        else:
            w = (1 - eta_t) * w  # update rule otherwise

    return w


#################################

# Place for additional code

#################################

def cross_validate_eta(data, labels, validation_data, validation_labels, C, eta_0_candidates, T):
    accuracies = []
    for eta_0 in eta_0_candidates:
        eta_acc = []
        for _ in range(10):  # average over 10 runs
            w = SGD_hinge(data, labels, C, eta_0, T)
            predictions = np.where(np.dot(validation_data, w) >= 0, 1, -1)
            accuracy = np.mean(predictions == validation_labels)
            eta_acc.append(accuracy)
        accuracies.append(np.mean(eta_acc))
    return accuracies


def cross_validate_C(data, labels, validation_data, validation_labels, best_eta_0, C_candidates, T):
    accuracies = []
    for C in C_candidates:
        C_acc = []
        for _ in range(10):  # average over 10 runs
            w = SGD_hinge(data, labels, C, best_eta_0, T)
            predictions = np.sign(np.dot(validation_data, w))
            accuracy = np.mean(predictions == validation_labels)
            C_acc.append(accuracy)
        accuracies.append(np.mean(C_acc))
    return accuracies


def plot_accuracies(eta_0_candidates, accuracies):
    plt.figure(figsize=(10, 6))
    plt.semilogx(eta_0_candidates, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Learning Rate ($\eta_0$)')
    plt.ylabel('Validation Accuracy')
    plt.title('SGD Hinge Loss Accuracy vs Learning Rate')
    plt.grid(True)
    plt.show()


def plot_C_accuracies(C_candidates, accuracies):
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_candidates, accuracies, marker='o', linestyle='-', color='r')
    plt.xlabel('Regularization Strength ($C$)')
    plt.ylabel('Validation Accuracy')
    plt.title('SGD Hinge Loss Accuracy vs Regularization Strength')
    plt.grid(True)
    plt.show()


def train_with_best_params_and_visualize(train_data, train_labels, best_eta_0, best_C, T):
    """
    Trains the classifier with the best hyperparameters and visualizes the weights as an image.
    """
    # Train with the best parameters
    w = SGD_hinge(train_data, train_labels, best_C, best_eta_0, T)

    # Visualize the resulting weights
    plt.figure(figsize=(6, 6))
    plt.imshow(w.reshape((28, 28)), cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('Learned Weight Vector Visualized as Image')
    plt.show()

    return w


def evaluate_on_test(test_data, test_labels, w):
    """
    Evaluates the classifier on the test set and computes accuracy.
    """
    predictions = np.where(np.dot(test_data, w) >= 0, 1, -1)
    accuracy = np.mean(predictions == test_labels)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()

    # Phase 1: Initial Coarse Search
    eta_0_candidates = np.logspace(-5, 5, num=11)
    C = 1
    T = 1000
    accuracies_eta = cross_validate_eta(train_data, train_labels, validation_data, validation_labels, C, eta_0_candidates, T)
    plot_accuracies(eta_0_candidates, accuracies_eta)
    best_eta_0_initial = eta_0_candidates[np.argmax(accuracies_eta)]
    print("Initial Best eta_0:", best_eta_0_initial)

    # Phase 2: High-Resolution Search
    min_eta = best_eta_0_initial / 10
    max_eta = best_eta_0_initial * 10
    high_res_eta_0_candidates = np.logspace(np.log10(min_eta), np.log10(max_eta), num=50)
    accuracies_high_res_eta = cross_validate_eta(train_data, train_labels, validation_data, validation_labels, C, high_res_eta_0_candidates, T)
    plot_accuracies(high_res_eta_0_candidates, accuracies_high_res_eta)
    best_eta_0_high_res = high_res_eta_0_candidates[np.argmax(accuracies_high_res_eta)]
    print("Refined Best eta_0:", best_eta_0_high_res)

    ultra_min_eta = best_eta_0_high_res / 2
    ultra_max_eta = best_eta_0_high_res * 2
    ultra_high_res_eta_0_candidates = np.logspace(np.log10(ultra_min_eta), np.log10(ultra_max_eta), num=100)
    accuracies_ultra_high_res_eta = cross_validate_eta(train_data, train_labels, validation_data, validation_labels, C, ultra_high_res_eta_0_candidates, T)
    plot_accuracies(ultra_high_res_eta_0_candidates, accuracies_ultra_high_res_eta)
    best_eta_0_ultra_high_res = ultra_high_res_eta_0_candidates[np.argmax(accuracies_ultra_high_res_eta)]
    print("Ultra Refined Best eta_0:", best_eta_0_ultra_high_res)

    # Use the best eta_0 to cross-validate C
    C_candidates = np.logspace(-5, 5, num=11)
    accuracies_C = cross_validate_C(train_data, train_labels, validation_data, validation_labels, best_eta_0_ultra_high_res, C_candidates, T)
    plot_C_accuracies(C_candidates, accuracies_C)

    best_C = C_candidates[np.argmax(accuracies_C)]
    print("Best C:", best_C)

    # Phase 3: Train with best parameters and T=20000, then visualize
    T_final = 20000
    w_final = train_with_best_params_and_visualize(train_data, train_labels, best_eta_0_ultra_high_res, best_C, T_final)
    print("Final Weights:", w_final)

    # Phase 4: Evaluate on the test set
    test_accuracy = evaluate_on_test(test_data, test_labels, w_final)
    print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
