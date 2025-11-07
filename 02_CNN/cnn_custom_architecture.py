"""
Custom CNN Architecture Design
Demonstrates how to design and experiment with different CNN architectures
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


class CustomCNNArchitecture:
    """
    Custom CNN architecture builder with flexible configurations
    """
    
    @staticmethod
    def simple_cnn(input_shape, num_classes):
        """Simple CNN with basic structure"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ], name='SimpleCNN')
        
        return model
    
    @staticmethod
    def vgg_like(input_shape, num_classes):
        """VGG-like architecture with multiple conv blocks"""
        model = models.Sequential([
            # Block 1
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            # Fully connected
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ], name='VGGLike')
        
        return model
    
    @staticmethod
    def residual_block(x, filters, kernel_size=3, stride=1):
        """Residual block inspired by ResNet"""
        shortcut = x
        
        # First conv
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    @staticmethod
    def resnet_like(input_shape, num_classes):
        """ResNet-like architecture with residual connections"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial conv
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = CustomCNNArchitecture.residual_block(x, 64)
        x = CustomCNNArchitecture.residual_block(x, 64)
        x = CustomCNNArchitecture.residual_block(x, 128, stride=2)
        x = CustomCNNArchitecture.residual_block(x, 128)
        x = CustomCNNArchitecture.residual_block(x, 256, stride=2)
        x = CustomCNNArchitecture.residual_block(x, 256)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=x, name='ResNetLike')
        
        return model
    
    @staticmethod
    def inception_module(x, filters):
        """Inception module with multiple kernel sizes"""
        # 1x1 conv
        branch1 = layers.Conv2D(filters, 1, padding='same', activation='relu')(x)
        
        # 3x3 conv
        branch2 = layers.Conv2D(filters, 1, padding='same', activation='relu')(x)
        branch2 = layers.Conv2D(filters, 3, padding='same', activation='relu')(branch2)
        
        # 5x5 conv
        branch3 = layers.Conv2D(filters, 1, padding='same', activation='relu')(x)
        branch3 = layers.Conv2D(filters, 5, padding='same', activation='relu')(branch3)
        
        # Max pooling
        branch4 = layers.MaxPooling2D(3, strides=1, padding='same')(x)
        branch4 = layers.Conv2D(filters, 1, padding='same', activation='relu')(branch4)
        
        # Concatenate
        output = layers.Concatenate()([branch1, branch2, branch3, branch4])
        
        return output
    
    @staticmethod
    def inception_like(input_shape, num_classes):
        """Inception-like architecture"""
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolutions
        x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Inception modules
        x = CustomCNNArchitecture.inception_module(x, 32)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        x = CustomCNNArchitecture.inception_module(x, 64)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=x, name='InceptionLike')
        
        return model


def compare_architectures(architectures, X_train, y_train, X_test, y_test, epochs=5):
    """Compare different architectures"""
    results = {}
    
    for name, model in architectures.items():
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}")
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nModel Summary:")
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        results[name] = {
            'model': model,
            'history': history,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'params': model.count_params()
        }
        
        print(f"\n{name} Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
    
    return results


def plot_comparison(results):
    """Plot comparison of different architectures"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    names = list(results.keys())
    
    # Plot accuracy comparison
    for name in names:
        history = results[name]['history']
        axes[0, 0].plot(history.history['accuracy'], label=f'{name} Train')
        axes[0, 1].plot(history.history['val_accuracy'], label=f'{name} Val')
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Training Accuracy Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Bar chart for test accuracy
    test_accs = [results[name]['test_accuracy'] * 100 for name in names]
    axes[1, 0].bar(names, test_accs, color='skyblue')
    axes[1, 0].set_ylabel('Test Accuracy (%)')
    axes[1, 0].set_title('Final Test Accuracy Comparison')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, axis='y')
    
    # Bar chart for parameters
    params = [results[name]['params'] / 1000 for name in names]
    axes[1, 1].bar(names, params, color='lightcoral')
    axes[1, 1].set_ylabel('Parameters (K)')
    axes[1, 1].set_title('Model Size Comparison')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('/tmp/cnn_architecture_comparison.png')
    print("\nComparison plot saved to /tmp/cnn_architecture_comparison.png")


def main():
    """Main function"""
    print("=" * 60)
    print("Custom CNN Architecture Design and Comparison")
    print("=" * 60)
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Use subset for faster comparison
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:2000]
    y_test = y_test[:2000]
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Create different architectures
    print("\nCreating different CNN architectures...")
    architectures = {
        'SimpleCNN': CustomCNNArchitecture.simple_cnn((28, 28, 1), 10),
        'VGGLike': CustomCNNArchitecture.vgg_like((28, 28, 1), 10),
        'ResNetLike': CustomCNNArchitecture.resnet_like((28, 28, 1), 10),
        'InceptionLike': CustomCNNArchitecture.inception_like((28, 28, 1), 10),
    }
    
    # Compare architectures
    results = compare_architectures(architectures, X_train, y_train, X_test, y_test, epochs=5)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(results)
    
    print("\n" + "=" * 60)
    print("Architecture comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
