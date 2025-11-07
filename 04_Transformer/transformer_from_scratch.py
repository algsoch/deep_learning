"""
Transformer Architecture Implementation from Scratch
Demonstrates the core components of the Transformer model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def positional_encoding(length, depth):
    """
    Generate positional encodings
    
    Args:
        length: Sequence length
        depth: Embedding dimension
    
    Returns:
        Positional encoding matrix
    """
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    
    pos_encoding = np.concatenate([
        np.sin(angle_rads),
        np.cos(angle_rads)
    ], axis=-1)
    
    return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(layers.Layer):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        
        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        
        return output, attention_weights


class FeedForward(layers.Layer):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.dense1 = layers.Dense(dff, activation='relu')
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


class EncoderLayer(layers.Layer):
    """Single Transformer Encoder Layer"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff, dropout_rate)
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


class TransformerEncoder(layers.Layer):
    """Stack of Transformer Encoder Layers"""
    
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, 
                 maximum_position_encoding, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False, mask=None):
        seq_len = tf.shape(x)[1]
        
        # Embedding + positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:seq_len, :]
        
        x = self.dropout(x, training=training)
        
        # Pass through encoder layers
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)
        
        return x


def create_transformer_classifier(vocab_size, max_length, num_classes, 
                                  d_model=128, num_heads=4, dff=512, 
                                  num_layers=4, dropout_rate=0.1):
    """
    Create a Transformer model for classification
    
    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
        num_classes: Number of output classes
        d_model: Dimension of model
        num_heads: Number of attention heads
        dff: Dimension of feed-forward network
        num_layers: Number of encoder layers
        dropout_rate: Dropout rate
    
    Returns:
        Compiled Transformer model
    """
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32)
    
    # Transformer encoder
    encoder = TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        vocab_size=vocab_size,
        maximum_position_encoding=max_length,
        dropout_rate=dropout_rate
    )
    
    x = encoder(inputs, training=True)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def generate_synthetic_sequence_data(n_samples=1000, seq_length=50, vocab_size=100):
    """Generate synthetic sequence data for classification"""
    np.random.seed(42)
    
    X = np.random.randint(0, vocab_size, size=(n_samples, seq_length))
    
    # Simple rule: if sum of first 10 elements is even, class 0, else class 1
    y = np.array([np.sum(x[:10]) % 2 for x in X])
    
    return X, y


def visualize_attention_weights(attention_weights, sentence):
    """Visualize attention weights"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(attention_weights, cmap='viridis')
    
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))
    ax.set_xticklabels(sentence, rotation=90)
    ax.set_yticklabels(sentence)
    
    plt.colorbar(im, ax=ax)
    plt.title('Attention Weights')
    plt.tight_layout()
    plt.savefig('/tmp/transformer_attention_weights.png')
    print("Attention weights visualization saved to /tmp/transformer_attention_weights.png")


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/transformer_training.png')
    print("Training history saved to /tmp/transformer_training.png")


def main():
    """Main function"""
    print("=" * 60)
    print("Transformer Architecture from Scratch")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating synthetic sequence data...")
    vocab_size = 100
    max_length = 50
    n_samples = 2000
    
    X, y = generate_synthetic_sequence_data(n_samples, max_length, vocab_size)
    
    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Sequence length: {max_length}")
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Create model
    print("\n2. Creating Transformer model...")
    model = create_transformer_classifier(
        vocab_size=vocab_size,
        max_length=max_length,
        num_classes=2,
        d_model=128,
        num_heads=4,
        dff=512,
        num_layers=2,
        dropout_rate=0.1
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n   Model Summary:")
    model.summary()
    
    # Train model
    print("\n3. Training the model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
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
    plot_training_history(history)
    
    # Save model
    print("\n6. Saving model...")
    model.save('/tmp/transformer_classifier.h5')
    print("   Model saved to /tmp/transformer_classifier.h5")
    
    print("\n" + "=" * 60)
    print("Transformer training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
