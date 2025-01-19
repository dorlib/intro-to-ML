import numpy as np
from scipy.special import softmax, logsumexp

class Network(object):
    
    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        is [784, 40, 10] then it would be a three-layer network, with the
        first layer (the input layer) containing 784 neurons, the second layer 40 neurons,
        and the third layer (the output layer) 10 neurons.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution centered around 0, scaled so as to be appropriate for ReLU.
        """
        self.num_layers = len(sizes) - 1
        self.sizes = sizes
        self.parameters = {}
        # Initialize weights with He initialization for ReLU
        for l in range(1, len(sizes)):
            self.parameters['W' + str(l)] = np.random.randn(sizes[l], sizes[l-1]) * np.sqrt(2. / sizes[l-1])
            self.parameters['b' + str(l)] = np.zeros((sizes[l], 1))


    ############################################################################
    #  1. ReLU and its derivative
    ############################################################################

    def relu(self, x):
        """
        Implements the ReLU activation element-wise: ReLU(x) = max(0, x).
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        Derivative of ReLU:
           d/dx ReLU(x) = 1 if x > 0, and 0 otherwise.
        """
        return (x > 0).astype(float)


    ############################################################################
    #  2. Cross entropy derivative w.r.t. the logits
    ############################################################################

    def cross_entropy_loss(self, logits, y_true):
        """
        Computes the average cross-entropy loss over a batch.

        Inputs:
          - logits: shape (10, batch_size)
          - y_true: shape (batch_size,) with integer labels in [0..9]
        """
        m = y_true.shape[0]
        # stable log-softmax
        log_probs = logits - logsumexp(logits, axis=0)
        # turn y_true into one-hot
        y_one_hot = np.eye(10)[y_true].T  # shape (10, batch_size)
        loss = -np.sum(y_one_hot * log_probs) / m
        return loss

    def cross_entropy_derivative(self, logits, y_true):
        """
        Gradient of the cross entropy w.r.t. the raw 'logits'.
        
        If we denote
           P = softmax(logits),
        then
           dL/d(logits) = P - y_true_one_hot
        """
        m = y_true.shape[0]
        # softmax along columns
        probs = softmax(logits, axis=0)  # shape (10, batch_size)
        y_one_hot = np.eye(10)[y_true].T # shape (10, batch_size)
        # derivative is (P - y)
        return (probs - y_one_hot)


    ############################################################################
    #  3. Forward propagation
    ############################################################################

    def forward_propagation(self, X):
        """
        Forward pass through the network.
        
        Inputs:
          - X: shape (784, batch_size)
        Returns:
          - ZL: the 'logits' of the final layer (shape: (10, batch_size))
          - forward_outputs: list of length self.num_layers storing intermediate results
             so we can do backprop later.
          
          Typically we store, for each layer l,
             (Z^{l-1},  V^l,  Z^l)
          where:
            V^l = W^l * Z^{l-1} + b^l
            Z^l = ReLU(V^l) for l < L, and the final Z^L = V^L (no ReLU).
        """
        forward_outputs = []
        A_prev = X  # 'activation' from previous layer, starting with input

        # Loop over layers 1..(L-1) with ReLU
        for l in range(1, self.num_layers):
            Wl = self.parameters['W' + str(l)]       # shape (sizes[l], sizes[l-1])
            bl = self.parameters['b' + str(l)]       # shape (sizes[l], 1)
            Vl = Wl @ A_prev + bl                    # shape (sizes[l], batch_size)
            Zl = self.relu(Vl)                       # ReLU activation
            forward_outputs.append((A_prev, Vl, Zl))
            A_prev = Zl

        # Final layer L has no ReLU (logits)
        WL = self.parameters['W' + str(self.num_layers)]
        bL = self.parameters['b' + str(self.num_layers)]
        VL = WL @ A_prev + bL   # shape (10, batch_size)
        # We store it, but there's no ReLU
        forward_outputs.append((A_prev, VL, None))

        ZL = VL  # the final 'logits'
        return ZL, forward_outputs


    ############################################################################
    #  4. Back-propagation
    ############################################################################

    def backpropagation(self, ZL, Y, forward_outputs):
        """
        Backward step: compute the gradients of the loss w.r.t. every W^l and b^l.
        
        Inputs:
          - ZL: shape (10, batch_size), the logits from the forward pass
          - Y: shape (batch_size,) integer labels for each example in the batch
          - forward_outputs: the list returned by forward_propagation
        
        Returns:
          - grads: dictionary with
               grads['dW1'], grads['db1'], ..., grads['dW(L)'], grads['db(L)']
        """
        grads = {}
        m = Y.shape[0]                # batch_size
        L = self.num_layers           # number of layers
        # 1) Gradient at the top (d/d logit)
        dZL = self.cross_entropy_derivative(ZL, Y)  # shape (10, batch_size)
        
        # 2) Compute and store grads for the final layer L
        (A_prev, VL, _) = forward_outputs[L-1]  # the last tuple
        # No activation at final layer => dV^L = dZ^L
        dVL = dZL
        grads['dW' + str(L)] = (1./m) * (dVL @ A_prev.T)  # shape (sizes[L], sizes[L-1])
        grads['db' + str(L)] = (1./m) * np.sum(dVL, axis=1, keepdims=True)

        # We'll propagate back to layer L-1, L-2, ..., 1
        dV_next = dVL
        W_next  = self.parameters['W' + str(L)]  # W^L (for using in next step)

        # 3) Loop backward over layers L-1..1
        for l in reversed(range(1, L)):
            A_prev, V_curr, Z_curr = forward_outputs[l-1]  # output from layer (l-1)
            # For layer l, the post-activation is Z^l, and we had V^l = W^l * Z^{l-1} + b^l
            # The next layer's dV is dV_next, so:
            # dZ^l = W^{l+1}^T * dV^{l+1}
            dZ_curr = W_next.T @ dV_next   # shape (sizes[l], batch_size)

            # The activation at layer l is ReLU, so we must multiply by derivative
            dV_curr = dZ_curr * self.relu_derivative(V_curr)  # shape (sizes[l], batch_size)

            # Then compute gradients wrt W^l, b^l
            grads['dW' + str(l)] = (1./m) * (dV_curr @ A_prev.T)
            grads['db' + str(l)] = (1./m) * np.sum(dV_curr, axis=1, keepdims=True)

            # Update variables for next iteration (further down)
            dV_next = dV_curr
            W_next  = self.parameters['W' + str(l)]

        return grads


    ############################################################################
    #  5. Update step (already given)
    ############################################################################

    def sgd_step(self, grads, learning_rate):
        """
        Updates the network parameters via SGD with the given gradients and learning rate.
        """
        parameters = self.parameters
        L = self.num_layers
        for l in range(L):
            parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
        return parameters


    def train(self, x_train, y_train, epochs, batch_size, learning_rate, x_test, y_test):
        """
        Trains the network with mini-batch SGD.
        """
        epoch_train_cost = []
        epoch_test_cost = []
        epoch_train_acc = []
        epoch_test_acc = []
        
        for epoch in range(epochs):
            costs = []
            acc = []
            # Loop over mini-batches
            for i in range(0, x_train.shape[1], batch_size):
                X_batch = x_train[:, i:i+batch_size]
                Y_batch = y_train[i:i+batch_size]

                # Forward
                ZL, caches = self.forward_propagation(X_batch)
                cost = self.cross_entropy_loss(ZL, Y_batch)
                costs.append(cost)

                # Backward
                grads = self.backpropagation(ZL, Y_batch, caches)

                # SGD update
                self.parameters = self.sgd_step(grads, learning_rate)

                preds = np.argmax(ZL, axis=0)
                train_acc = self.calculate_accuracy(preds, Y_batch, len(Y_batch))
                acc.append(train_acc)

            average_train_cost = np.mean(costs)
            average_train_acc = np.mean(acc)
            print(f"Epoch: {epoch + 1}, "
                  f"Training loss: {average_train_cost:.6f}, "
                  f"Training accuracy: {average_train_acc:.4f}")

            epoch_train_cost.append(average_train_cost)
            epoch_train_acc.append(average_train_acc)

            # Evaluate on test set
            ZL_test, _ = self.forward_propagation(x_test)
            test_cost = self.cross_entropy_loss(ZL_test, y_test)
            preds_test = np.argmax(ZL_test, axis=0)
            test_acc = self.calculate_accuracy(preds_test, y_test, len(y_test))
            
            print(f"  >> Test loss: {test_cost:.6f}, Test accuracy: {test_acc:.4f}")
            epoch_test_cost.append(test_cost)
            epoch_test_acc.append(test_acc)

        return self.parameters, epoch_train_cost, epoch_test_cost, epoch_train_acc, epoch_test_acc


    def calculate_accuracy(self, y_pred, y_true, batch_size):
        """Returns the average accuracy of the prediction over the batch."""
        return np.sum(y_pred == y_true) / batch_size
