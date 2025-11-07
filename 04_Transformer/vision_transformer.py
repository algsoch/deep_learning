"""
Vision Transformer (ViT) for Image Classification
Demonstrates how to apply Transformer architecture to image data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class PatchExtractor(layers.Layer):
    """Extract patches from images"""
    
    def __init__(self, patch_size):
        super(PatchExtractor, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    """Encode patches with position embeddings"""
    
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate):
    """Multi-layer perceptron"""
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier(
    input_shape,
    num_classes,
    patch_size=4,
    projection_dim=64,
    num_heads=4,
    transformer_layers=4,
    mlp_head_units=[128, 64],
):
    """
    Create Vision Transformer (ViT) model
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        patch_size: Size of image patches
        projection_dim: Dimension of patch embeddings
        num_heads: Number of attention heads
        transformer_layers: Number of transformer layers
        mlp_head_units: Units in MLP head
    
    Returns:
        ViT model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Create patches
    num_patches = (input_shape[0] // patch_size) ** 2
    patches = PatchExtractor(patch_size)(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    # Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=[projection_dim * 2, projection_dim], dropout_rate=0.1)
        
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])
    
    # Final layer normalization
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    # Classification head
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=logits)
    
    return model


def load_and_preprocess_data():
    """Load CIFAR-10 dataset"""
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    # Flatten labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    return (X_train, y_train), (X_test, y_test)


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Vision Transformer - Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Vision Transformer - Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/vision_transformer_training.png')
    print("Training history saved to /tmp/vision_transformer_training.png")


def visualize_patches(images, patch_size):
    """Visualize how images are divided into patches"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]
            ax.imshow(img)
            
            # Draw grid for patches
            for j in range(0, img.shape[0], patch_size):
                ax.axhline(y=j, color='red', linewidth=0.5)
            for j in range(0, img.shape[1], patch_size):
                ax.axvline(x=j, color='red', linewidth=0.5)
            
            ax.set_title(f'Image {i+1} with {patch_size}x{patch_size} patches')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/vision_transformer_patches.png')
    print("Patch visualization saved to /tmp/vision_transformer_patches.png")


def main():
    """Main function"""
    print("=" * 60)
    print("Vision Transformer (ViT) for Image Classification")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading CIFAR-10 dataset...")
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    
    # Use subset for faster training
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]}")
    
    # Visualize patches
    print("\n2. Visualizing image patches...")
    patch_size = 4
    visualize_patches(X_test[:10], patch_size)
    
    # Create model
    print("\n3. Creating Vision Transformer model...")
    model = create_vit_classifier(
        input_shape=(32, 32, 3),
        num_classes=10,
        patch_size=patch_size,
        projection_dim=64,
        num_heads=4,
        transformer_layers=4,
        mlp_head_units=[128, 64],
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    
    print("\n   Model Summary:")
    model.summary()
    print(f"\n   Total parameters: {model.count_params():,}")
    
    # Train model
    print("\n4. Training the model...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=10,
        validation_split=0.2,
        verbose=1,
    )
    
    # Evaluate
    print("\n5. Evaluating the model...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {results[0]:.4f}")
    print(f"   Test Accuracy: {results[1] * 100:.2f}%")
    print(f"   Test Top-5 Accuracy: {results[2] * 100:.2f}%")
    
    # Visualize training
    print("\n6. Visualizing training history...")
    plot_training_history(history)
    
    # Make predictions
    print("\n7. Making predictions on test samples...")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    predictions = model.predict(X_test[:5])
    predicted_classes = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X_test[i])
        true_label = class_names[y_test[i]]
        pred_label = class_names[predicted_classes[i]]
        color = 'green' if y_test[i] == predicted_classes[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/vision_transformer_predictions.png')
    print("Predictions saved to /tmp/vision_transformer_predictions.png")
    
    # Save model
    print("\n8. Saving model...")
    model.save('/tmp/vision_transformer_model.h5')
    print("   Model saved to /tmp/vision_transformer_model.h5")
    
    print("\n" + "=" * 60)
    print("Vision Transformer training complete!")
    print("=" * 60)
    print("\nKey advantages of Vision Transformers:")
    print("- Can capture long-range dependencies in images")
    print("- More interpretable attention patterns")
    print("- Better scalability with data and compute")
    print("- No need for complex inductive biases")


if __name__ == "__main__":
    main()
