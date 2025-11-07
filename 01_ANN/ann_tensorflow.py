"""
Artificial Neural Network using TensorFlow/Keras
Demonstrates how to build ANNs with high-level APIs
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def create_ann_model(input_shape, hidden_units, output_units):
    """
    Create an ANN model using Keras Sequential API
    
    Args:
        input_shape: Shape of input features
        hidden_units: List of hidden layer sizes
        output_units: Number of output units
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_shape,)))
    
    # Hidden layers
    for units in hidden_units:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.2))  # Regularization
    
    # Output layer
    if output_units == 1:
        model.add(layers.Dense(output_units, activation='sigmoid'))
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    else:
        model.add(layers.Dense(output_units, activation='softmax'))
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    return model


def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    
    return (X_train, y_train), (X_test, y_test)


def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/ann_tensorflow_training.png')
    print("Training history plot saved to /tmp/ann_tensorflow_training.png")


def visualize_predictions(model, X_test, y_test, n_samples=10):
    """Visualize some predictions"""
    predictions = model.predict(X_test[:n_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(n_samples):
        img = X_test[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'True: {y_test[i]}\nPred: {predicted_classes[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/ann_predictions.png')
    print("Predictions visualization saved to /tmp/ann_predictions.png")


def main():
    """Main function to demonstrate ANN with TensorFlow"""
    print("=" * 60)
    print("Artificial Neural Network using TensorFlow/Keras")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Input features: {X_train.shape[1]}")
    
    # Create model
    print("\n2. Creating ANN model...")
    model = create_ann_model(
        input_shape=784,
        hidden_units=[128, 64, 32],
        output_units=10
    )
    
    print("\n   Model Architecture:")
    model.summary()
    
    # Train model
    print("\n3. Training the model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("\n4. Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Plot results
    print("\n5. Visualizing results...")
    plot_training_history(history)
    visualize_predictions(model, X_test, y_test)
    
    # Save model
    print("\n6. Saving the model...")
    model.save('/tmp/ann_mnist_model.h5')
    print("   Model saved to /tmp/ann_mnist_model.h5")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
