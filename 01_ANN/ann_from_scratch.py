"""
Artificial Neural Network (ANN) Implementation from Scratch
Using only NumPy to understand the fundamentals
"""

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network with random weights
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes/values
            learning_rate: Learning rate for gradient descent
        """
        self.learning_rate = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """
        Forward pass through the network
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            output: Network predictions
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward_propagation(self, X, y, output):
        """
        Backward pass to compute gradients
        
        Args:
            X: Input data
            y: True labels
            output: Network predictions
        """
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training data
            y: Training labels
            epochs: Number of training epochs
            verbose: Print training progress
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward_propagation(X)
            
            # Compute loss (Binary Cross-Entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            
            # Backward propagation
            self.backward_propagation(X, y, output)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions on new data"""
        output = self.forward_propagation(X)
        return (output > 0.5).astype(int)


def generate_sample_data(n_samples=1000):
    """Generate sample data for binary classification"""
    np.random.seed(42)
    
    # Class 0: centered at (-1, -1)
    X1 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
    y1 = np.zeros((n_samples // 2, 1))
    
    # Class 1: centered at (1, 1)
    X2 = np.random.randn(n_samples // 2, 2) + np.array([1, 1])
    y2 = np.ones((n_samples // 2, 1))
    
    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def plot_decision_boundary(model, X, y):
    """Visualize the decision boundary"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('ANN Decision Boundary')
    plt.colorbar(label='Class')
    plt.savefig('/tmp/ann_decision_boundary.png')
    print("Decision boundary plot saved to /tmp/ann_decision_boundary.png")


def main():
    """Main function to demonstrate ANN"""
    print("=" * 60)
    print("Artificial Neural Network (ANN) - From Scratch Implementation")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    X, y = generate_sample_data(n_samples=1000)
    print(f"   Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create and train model
    print("\n2. Creating neural network...")
    model = NeuralNetwork(input_size=2, hidden_size=10, output_size=1, learning_rate=0.1)
    
    print("\n3. Training the model...")
    losses = model.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate model
    print("\n4. Evaluating the model...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_accuracy = np.mean(train_predictions == y_train) * 100
    test_accuracy = np.mean(test_predictions == y_test) * 100
    
    print(f"   Training Accuracy: {train_accuracy:.2f}%")
    print(f"   Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot results
    print("\n5. Visualizing results...")
    plot_decision_boundary(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Training complete! Check /tmp/ann_decision_boundary.png for visualization")
    print("=" * 60)


if __name__ == "__main__":
    main()
