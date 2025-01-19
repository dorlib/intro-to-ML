import matplotlib.pyplot as plt
import numpy as np

# Import the network and data loading utilities
from backprop_network import Network
from backprop_data import load_as_matrix_with_labels

def train_and_plot():
    # 1) Load Data
    np.random.seed(0)  # for reproducibility
    n_train = 10000
    n_test = 5000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)
    
    # 2) Define general training settings (besides the LR)
    epochs = 30
    batch_size = 100
    
    # 3) We will loop over these learning rates:
    learning_rates = [0.001, 0.01, 0.1, 1, 10]

    # 4) Prepare storage for results
    all_train_acc = {}
    all_train_loss = {}
    all_test_acc = {}
    
    # 5) Train for each learning rate
    for lr in learning_rates:
        print(f"\n--- Training with learning rate = {lr} ---")
        
        # Create a new network for each LR
        layer_dims = [784, 40, 10]  # one hidden layer of size 40
        net = Network(layer_dims)
        
        # We capture the returned arrays: 
        #   epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc
        params, train_loss_list, test_loss_list, train_acc_list, test_acc_list = net.train(
            x_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            learning_rate=lr, 
            x_test=x_test, 
            y_test=y_test
        )
        
        # Store the results so we can plot them later
        all_train_acc[lr]  = train_acc_list
        all_train_loss[lr] = train_loss_list
        all_test_acc[lr]   = test_acc_list
    
    # 6) Plot results
    # We will make three separate plots:
    #    (a) Training accuracy vs epoch
    #    (b) Training loss vs epoch
    #    (c) Test accuracy vs epoch

    epochs_axis = np.arange(1, epochs+1)

    # (a) Training Accuracy
    plt.figure(figsize=(8,6))
    for lr in learning_rates:
        plt.plot(epochs_axis, all_train_acc[lr], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    # (b) Training Loss
    plt.figure(figsize=(8,6))
    for lr in learning_rates:
        plt.plot(epochs_axis, all_train_loss[lr], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

    # (c) Test Accuracy
    plt.figure(figsize=(8,6))
    for lr in learning_rates:
        plt.plot(epochs_axis, all_test_acc[lr], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


# section (c) : Use the full training and test sets: 
def section_c_training():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

    net = Network([784, 40, 10])
    params, train_loss_list, test_loss_list, train_acc_list, test_acc_list = net.train(
        x_train, 
        y_train, 
        epochs=30, 
        batch_size=10, 
        learning_rate=0.1, 
        x_test=x_test, 
        y_test=y_test
    )


def section_d_training():
    n_train = 50000
    n_test = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)


    layer_dims = [784, 10]  # no hidden layer
    net = Network(layer_dims)
    net.train(
        x_train, 
        y_train, 
        epochs=30, 
        batch_size=100,   # e.g., 10 or 100
        learning_rate=0.1,
        x_test=x_test, 
        y_test=y_test
    )

    W = net.parameters["W1"]  # shape = (10, 784)
    b = net.parameters["b1"]  # shape = (10, 1), typically ignored in the plot

    for i in range(10):
        plt.subplot(2,5,i+1)  # for a 2x5 grid of images
        plt.imshow(W[i].reshape(28,28), cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title(f"Digit {i}")
    plt.tight_layout()
    plt.show()



def section_e_training():
    """
    Trains a deeper MLP on full MNIST to (usually) surpass 97% test accuracy.
    Returns: 
      epoch_train_cost, 
      epoch_test_cost, 
      epoch_train_acc, 
      epoch_test_acc
    for plotting.
    """
    np.random.seed(0)  # reproducibility

    # 1) Load the full MNIST dataset
    n_train = 50000
    n_test  = 10000
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)
    
    # 2) Create a deeper network: 2 hidden layers (128 -> 64)
    layer_dims = [784, 128, 64, 10]
    net = Network(layer_dims)

    # Hyperparameters
    epochs = 40
    batch_size = 64
    base_learning_rate = 0.1

    # For storing metrics each epoch
    epoch_train_cost = []
    epoch_test_cost  = []
    epoch_train_acc  = []
    epoch_test_acc   = []

    # 3) Training loop with a simple *manual* LR schedule
    for epoch in range(epochs):
        # After epoch 15, reduce LR by factor of 10 (a simple schedule)
        if epoch < 15:
            learning_rate = base_learning_rate
        else:
            learning_rate = base_learning_rate * 0.1

        # shuffle training data each epoch for better SGD
        perm = np.random.permutation(x_train.shape[1])
        x_train = x_train[:, perm]
        y_train = y_train[perm]

        # Mini-batch SGD
        costs = []
        accs  = []
        for i in range(0, x_train.shape[1], batch_size):
            X_batch = x_train[:, i : i + batch_size]
            Y_batch = y_train[i : i + batch_size]

            # Forward
            ZL, caches = net.forward_propagation(X_batch)
            cost = net.cross_entropy_loss(ZL, Y_batch)
            costs.append(cost)

            # Backward
            grads = net.backpropagation(ZL, Y_batch, caches)

            # Update
            net.parameters = net.sgd_step(grads, learning_rate)

            # Compute training accuracy on this batch
            preds = np.argmax(ZL, axis=0)
            batch_acc = net.calculate_accuracy(preds, Y_batch, len(Y_batch))
            accs.append(batch_acc)

        # Average metrics across mini-batches
        avg_train_cost = np.mean(costs)
        avg_train_acc  = np.mean(accs)

        # Evaluate on the full test set
        ZL_test, _ = net.forward_propagation(x_test)
        test_cost = net.cross_entropy_loss(ZL_test, y_test)
        preds_test = np.argmax(ZL_test, axis=0)
        test_acc = net.calculate_accuracy(preds_test, y_test, x_test.shape[1])

        epoch_train_cost.append(avg_train_cost)
        epoch_test_cost.append(test_cost)
        epoch_train_acc.append(avg_train_acc)
        epoch_test_acc.append(test_acc)

        # Print per-epoch statistics
        print(f"Epoch: {epoch+1:2d}, "
              f"LR: {learning_rate:.3f}, "
              f"Train Loss: {avg_train_cost:.4f}, "
              f"Train Acc: {avg_train_acc:.4f}, "
              f"Test Loss: {test_cost:.4f}, "
              f"Test Acc: {test_acc:.4f}")

    return epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc


def plot_training_results(train_cost, test_cost, train_acc, test_acc):
    """Generates plots of cost & accuracy across epochs."""

    epochs_axis = np.arange(1, len(train_cost) + 1)

    plt.figure(figsize=(10, 4))

    # Subplot 1: Training vs. Test Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_axis, train_cost, label="Train Loss", marker='o')
    plt.plot(epochs_axis, test_cost,  label="Test Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss Across Epochs")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Training vs. Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_axis, train_acc, label="Train Acc", marker='o')
    plt.plot(epochs_axis, test_acc,  label="Test Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def section_e():
    train_cost, test_cost, train_acc, test_acc = section_e_training()
    plot_training_results(train_cost, test_cost, train_acc, test_acc)



if __name__ == "__main__":
    # train_and_plot()
    # section_c_training()
    # section_d_training()
    section_e()
