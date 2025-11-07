"""
Convolutional Neural Network (CNN) for Image Classification
Using TensorFlow/Keras to classify MNIST and CIFAR-10 datasets
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def create_cnn_model(input_shape, num_classes):
    """
    Create a CNN model for image classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_and_preprocess_mnist():
    """Load and preprocess MNIST dataset"""
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return (X_train, y_train), (X_test, y_test)


def load_and_preprocess_cifar10():
    """Load and preprocess CIFAR-10 dataset"""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return (X_train, y_train), (X_test, y_test)


def plot_training_history(history, dataset_name):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'{dataset_name} - Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'{dataset_name} - Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'/tmp/cnn_{dataset_name.lower()}_training.png')
    print(f"Training history saved to /tmp/cnn_{dataset_name.lower()}_training.png")


def visualize_predictions(model, X_test, y_test, class_names, dataset_name, n_samples=15):
    """Visualize predictions"""
    predictions = model.predict(X_test[:n_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.ravel()
    
    for i in range(n_samples):
        if X_test[i].shape[-1] == 1:
            axes[i].imshow(X_test[i].squeeze(), cmap='gray')
        else:
            axes[i].imshow(X_test[i])
        
        true_label = class_names[y_test[i]]
        pred_label = class_names[predicted_classes[i]]
        color = 'green' if y_test[i] == predicted_classes[i] else 'red'
        
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'/tmp/cnn_{dataset_name.lower()}_predictions.png')
    print(f"Predictions saved to /tmp/cnn_{dataset_name.lower()}_predictions.png")


def train_on_mnist():
    """Train CNN on MNIST dataset"""
    print("\n" + "=" * 60)
    print("Training CNN on MNIST Dataset")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]}")
    
    # Create model
    print("\n2. Creating CNN model...")
    model = create_cnn_model(input_shape=(28, 28, 1), num_classes=10)
    print(f"   Total parameters: {model.count_params():,}")
    
    # Train model
    print("\n3. Training the model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("\n4. Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Visualize
    print("\n5. Visualizing results...")
    plot_training_history(history, "MNIST")
    class_names = [str(i) for i in range(10)]
    visualize_predictions(model, X_test, y_test, class_names, "MNIST")
    
    return model


def train_on_cifar10():
    """Train CNN on CIFAR-10 dataset"""
    print("\n" + "=" * 60)
    print("Training CNN on CIFAR-10 Dataset")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_cifar10()
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]}")
    
    # Create model
    print("\n2. Creating CNN model...")
    model = create_cnn_model(input_shape=(32, 32, 3), num_classes=10)
    print(f"   Total parameters: {model.count_params():,}")
    
    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    
    # Train model
    print("\n3. Training the model with data augmentation...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("\n4. Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Visualize
    print("\n5. Visualizing results...")
    plot_training_history(history, "CIFAR10")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    visualize_predictions(model, X_test, y_test, class_names, "CIFAR10")
    
    return model


def main():
    """Main function"""
    print("=" * 60)
    print("Convolutional Neural Network (CNN) for Image Classification")
    print("=" * 60)
    
    # Train on MNIST
    mnist_model = train_on_mnist()
    
    # Train on CIFAR-10
    cifar_model = train_on_cifar10()
    
    print("\n" + "=" * 60)
    print("All training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
