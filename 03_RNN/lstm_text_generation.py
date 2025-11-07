"""
LSTM Text Generation
Generate text using LSTM trained on sample text data
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def load_sample_text():
    """
    Load or create sample text for training
    Using Shakespeare text as an example
    """
    # Sample text - in practice, you would load a larger corpus
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """
    
    return text.lower()


def create_sequences(text, seq_length=40):
    """
    Create training sequences from text
    
    Args:
        text: Input text
        seq_length: Length of each sequence
    
    Returns:
        X, y, char_to_idx, idx_to_char
    """
    # Get unique characters
    chars = sorted(list(set(text)))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    # Create sequences
    X = []
    y = []
    
    for i in range(len(text) - seq_length):
        sequence = text[i:i + seq_length]
        target = text[i + seq_length]
        
        X.append([char_to_idx[char] for char in sequence])
        y.append(char_to_idx[target])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, char_to_idx, idx_to_char


def create_text_generation_model(vocab_size, seq_length, embedding_dim=64):
    """
    Create LSTM model for text generation
    
    Args:
        vocab_size: Number of unique characters
        seq_length: Length of input sequences
        embedding_dim: Dimension of character embeddings
    
    Returns:
        Compiled model
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=seq_length),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def generate_text(model, seed_text, char_to_idx, idx_to_char, length=200, temperature=1.0):
    """
    Generate text using trained model
    
    Args:
        model: Trained model
        seed_text: Starting text
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        length: Length of text to generate
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated text
    """
    generated = seed_text
    seq_length = model.input_shape[1]
    
    for _ in range(length):
        # Prepare input
        x = np.zeros((1, seq_length))
        for i, char in enumerate(seed_text[-seq_length:]):
            if char in char_to_idx:
                x[0, i] = char_to_idx[char]
        
        # Predict next character
        predictions = model.predict(x, verbose=0)[0]
        
        # Apply temperature
        predictions = np.log(predictions + 1e-8) / temperature
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))
        
        # Sample from distribution
        next_idx = np.random.choice(len(predictions), p=predictions)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        seed_text += next_char
    
    return generated


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['loss'], label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/tmp/lstm_text_generation_training.png')
    print("Training history saved to /tmp/lstm_text_generation_training.png")


def main():
    """Main function"""
    print("=" * 60)
    print("LSTM Text Generation")
    print("=" * 60)
    
    # Load text
    print("\n1. Loading text data...")
    text = load_sample_text()
    print(f"   Text length: {len(text)} characters")
    print(f"   Unique characters: {len(set(text))}")
    
    # Create sequences
    print("\n2. Creating sequences...")
    seq_length = 40
    X, y, char_to_idx, idx_to_char = create_sequences(text, seq_length)
    vocab_size = len(char_to_idx)
    
    print(f"   Number of sequences: {len(X)}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Vocabulary size: {vocab_size}")
    
    # Create model
    print("\n3. Creating LSTM model...")
    model = create_text_generation_model(vocab_size, seq_length)
    
    print("\n   Model Summary:")
    model.summary()
    
    # Train model
    print("\n4. Training the model...")
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Plot training
    print("\n5. Plotting training history...")
    plot_training_history(history)
    
    # Generate text
    print("\n6. Generating text samples...")
    seed_texts = [
        "to be, or not to be",
        "the question",
        "to sleep"
    ]
    
    temperatures = [0.5, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n   Temperature: {temp}")
        print("   " + "=" * 50)
        for seed in seed_texts:
            generated = generate_text(
                model, seed, char_to_idx, idx_to_char,
                length=100, temperature=temp
            )
            print(f"\n   Seed: '{seed}'")
            print(f"   Generated: {generated[:150]}...")
    
    # Save model
    print("\n7. Saving model...")
    model.save('/tmp/lstm_text_generator.h5')
    print("   Model saved to /tmp/lstm_text_generator.h5")
    
    print("\n" + "=" * 60)
    print("Text generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
