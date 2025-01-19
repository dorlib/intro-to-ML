import pickle
import gzip
import numpy as np


def load_as_matrix_with_labels(train_size, test_size):
    """Return two numpy arrays containing training and test data along with their corresponding labels.

    `train_data` is a 784x60000 numpy array containing the input images.
    `test_data` is a 784x10000 numpy array containing the input images.
    `y_train` and `y_test` are arrays containing the corresponding labels for training and testing datasets.
    """
    # Load the dataset from a gzipped pickle file
    with gzip.open('./mnist.pkl.gz', 'rb') as f:
        train_from_file, _, test_from_file = pickle.load(f, encoding='iso-8859-1')

    train_data = np.transpose(train_from_file[0])
    train_labels = np.array(train_from_file[1])
    test_data = np.transpose(test_from_file[0])
    test_labels = np.array(test_from_file[1])

    # If sizes are specified to be smaller than the original, we shuffle and select
    if train_size < 50000:
        np.random.seed(8)  # For reproducibility
        indices = np.random.choice(50000, train_size, replace=False)
        train_data = train_data[:, indices]
        train_labels = train_labels[indices]

    if test_size < 10000:
        np.random.seed(8)  # Consistent shuffling for reproducibility
        indices = np.random.choice(10000, test_size, replace=False)
        test_data = test_data[:, indices]
        test_labels = test_labels[indices]

    return train_data, train_labels, test_data, test_labels
