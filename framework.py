import numpy as np

class Linear:
    """
    A linear (fully connected) layer in a neural network.
    
    Parameters:
    - input_dim (int): Number of input features.
    - output_dim (int): Number of output features.
    """
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.zeros((1, output_dim))

    def forward(self, X):
        """
        Performs the forward pass through the linear layer.
        
        Parameters:
        - X (ndarray): Input data of shape (batch_size, input_dim).
        
        Returns:
        - ndarray: Output data of shape (batch_size, output_dim).
        """
        self.input = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, grad_output):
        """
        Performs the backward pass and computes the gradients.
        
        Parameters:
        - grad_output (ndarray): Gradient of the loss with respect to the output.
        
        Returns:
        - ndarray: Gradient of the loss with respect to the input.
        """
        grad_input = np.dot(grad_output, self.weights.T)
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input

class ReLU:
    """
    ReLU (Rectified Linear Unit) activation layer.
    """
    def forward(self, X):
        """
        Applies the ReLU activation function.
        
        Parameters:
        - X (ndarray): Input data.
        
        Returns:
        - ndarray: Output data after applying ReLU.
        """
        self.input = X
        return np.maximum(0, X)

    def backward(self, grad_output):
        """
        Computes the gradient of the loss with respect to the input.
        
        Parameters:
        - grad_output (ndarray): Gradient of the loss with respect to the output.
        
        Returns:
        - ndarray: Gradient of the loss with respect to the input.
        """
        grad_input = grad_output * (self.input > 0)
        return grad_input

class Softmax:
    """
    Softmax activation layer.
    """
    def forward(self, X):
        """
        Applies the softmax activation function.
        
        Parameters:
        - X (ndarray): Input data.
        
        Returns:
        - ndarray: Output data after applying softmax.
        """
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        """
        Computes the gradient of the loss with respect to the input.
        
        Parameters:
        - grad_output (ndarray): Gradient of the loss with respect to the output.
        
        Returns:
        - ndarray: Gradient of the loss with respect to the input.
        """
        return grad_output

class CrossEntropyLoss:
    """
    Cross-entropy loss function.
    """
    def forward(self, y_pred, y_true):
        """
        Computes the forward pass of the cross-entropy loss.
        
        Parameters:
        - y_pred (ndarray): Predicted probabilities.
        - y_true (ndarray): True labels.
        
        Returns:
        - float: Loss value.
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-12, 1. - 1e-12)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        return -np.mean(np.log(correct_confidences))

    def backward(self, y_pred, y_true):
        """
        Computes the backward pass of the cross-entropy loss.
        
        Parameters:
        - y_pred (ndarray): Predicted probabilities.
        - y_true (ndarray): True labels.
        
        Returns:
        - ndarray: Gradient of the loss with respect to the input.
        """
        samples = len(y_pred)
        grad = y_pred
        grad[range(samples), y_true] -= 1
        return grad / samples

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    Parameters:
    - learning_rate (float): Learning rate for the optimizer.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Updates the weights and biases of each layer.
        
        Parameters:
        - layers (list): List of layers in the model.
        """
        for layer in layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias

class Model:
    """
    Neural network model.
    """
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add_layer(self, layer):
        """
        Adds a layer to the model.
        
        Parameters:
        - layer: Layer to be added to the model.
        """
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """
        Compiles the model by specifying the loss function and optimizer.
        
        Parameters:
        - loss: Loss function.
        - optimizer: Optimizer.
        """
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        """
        Performs the forward pass through all the layers.
        
        Parameters:
        - X (ndarray): Input data.
        
        Returns:
        - ndarray: Output after passing through all layers.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad_output):
        """
        Performs the backward pass through all the layers.
        
        Parameters:
        - grad_output (ndarray): Gradient of the loss with respect to the output.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def train(self, X, y, epochs):
        """
        Trains the model for a specified number of epochs.
        
        Parameters:
        - X (ndarray): Training data.
        - y (ndarray): True labels.
        - epochs (int): Number of epochs to train for.
        """
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss_value = self.loss.forward(y_pred, y)
            grad_output = self.loss.backward(y_pred, y)
            self.backward(grad_output)
            self.optimizer.step(self.layers)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.4f}')

    def predict(self, X):
        """
        Predicts the output for the given input data.
        
        Parameters:
        - X (ndarray): Input data.
        
        Returns:
        - ndarray: Predicted output.
        """
        return self.forward(X)

    def evaluate(self, X, y):
        """
        Evaluates the model on the given data.
        
        Parameters:
        - X (ndarray): Input data.
        - y (ndarray): True labels.
        
        Returns:
        - tuple: Loss value and accuracy.
        """
        y_pred = self.predict(X)
        loss_value = self.loss.forward(y_pred, y)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
        print(f'Loss: {loss_value:.4f}, Accuracy: {accuracy:.4f}')
        return loss_value, accuracy
