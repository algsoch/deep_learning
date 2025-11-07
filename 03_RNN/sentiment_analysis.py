"""
Sentiment Analysis using RNN/LSTM
Demonstrates text classification for sentiment analysis
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


def load_sample_sentiment_data():
    """
    Create sample sentiment data for demonstration
    In practice, you would use IMDB, Twitter sentiment, or other datasets
    """
    # Positive reviews
    positive_reviews = [
        "This movie was excellent and amazing",
        "I loved this film, it was fantastic",
        "Great acting and wonderful story",
        "Best movie I have seen in years",
        "Absolutely brilliant and entertaining",
        "Superb performance by all actors",
        "Highly recommend this masterpiece",
        "Perfect movie for everyone",
        "Outstanding and memorable experience",
        "Incredible cinematography and direction",
        "This is a must watch film",
        "Exceptional quality and production",
        "Thoroughly enjoyed every moment",
        "Remarkable storytelling and plot",
        "Five stars, absolutely amazing"
    ] * 10  # Repeat to get more samples
    
    # Negative reviews
    negative_reviews = [
        "This movie was terrible and boring",
        "I hated this film, complete waste of time",
        "Poor acting and awful story",
        "Worst movie I have ever seen",
        "Absolutely dreadful and disappointing",
        "Terrible performance by all actors",
        "Do not recommend this disaster",
        "Horrible movie, avoid at all costs",
        "Boring and forgettable experience",
        "Bad cinematography and direction",
        "This is a terrible film",
        "Poor quality and production",
        "Could not finish watching",
        "Terrible storytelling and plot",
        "One star, absolutely terrible"
    ] * 10  # Repeat to get more samples
    
    # Combine and create labels
    texts = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    return texts, labels


def preprocess_text_data(texts, labels, max_words=1000, max_len=50):
    """
    Preprocess text data for RNN
    
    Args:
        texts: List of text samples
        labels: List of labels
        max_words: Maximum vocabulary size
        max_len: Maximum sequence length
    
    Returns:
        X_train, X_test, y_train, y_test, tokenizer
    """
    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = np.array(labels)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, tokenizer


def create_sentiment_model(vocab_size, max_len, model_type='LSTM'):
    """
    Create sentiment analysis model
    
    Args:
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        model_type: Type of RNN ('SimpleRNN', 'LSTM', 'GRU', 'BiLSTM')
    
    Returns:
        Compiled model
    """
    model = keras.Sequential()
    
    # Embedding layer
    model.add(layers.Embedding(vocab_size, 64, input_length=max_len))
    
    # RNN layers based on type
    if model_type == 'SimpleRNN':
        model.add(layers.SimpleRNN(64, return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.SimpleRNN(32))
    elif model_type == 'LSTM':
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.LSTM(32))
    elif model_type == 'GRU':
        model.add(layers.GRU(64, return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.GRU(32))
    elif model_type == 'BiLSTM':
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
        model.add(layers.Dropout(0.3))
        model.add(layers.Bidirectional(layers.LSTM(32)))
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def predict_sentiment(model, tokenizer, text, max_len=50):
    """
    Predict sentiment of a text
    
    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        text: Input text
        max_len: Maximum sequence length
    
    Returns:
        Sentiment (Positive/Negative) and confidence
    """
    # Preprocess text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    if prediction > 0.5:
        sentiment = "Positive"
        confidence = prediction
    else:
        sentiment = "Negative"
        confidence = 1 - prediction
    
    return sentiment, confidence


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
    plt.savefig('/tmp/sentiment_analysis_training.png')
    print("Training history saved to /tmp/sentiment_analysis_training.png")


def main():
    """Main function"""
    print("=" * 60)
    print("Sentiment Analysis using LSTM")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading sentiment data...")
    texts, labels = load_sample_sentiment_data()
    print(f"   Total samples: {len(texts)}")
    print(f"   Positive samples: {sum(labels)}")
    print(f"   Negative samples: {len(labels) - sum(labels)}")
    
    # Preprocess data
    print("\n2. Preprocessing text data...")
    max_words = 1000
    max_len = 50
    X_train, X_test, y_train, y_test, tokenizer = preprocess_text_data(
        texts, labels, max_words, max_len
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Vocabulary size: {len(tokenizer.word_index)}")
    print(f"   Sequence length: {max_len}")
    
    # Create and train model
    print("\n3. Creating LSTM model...")
    model = create_sentiment_model(max_words, max_len, model_type='LSTM')
    
    print("\n   Model Summary:")
    model.summary()
    
    print("\n4. Training the model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    print("\n5. Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Plot training history
    print("\n6. Plotting training history...")
    plot_training_history(history)
    
    # Test predictions
    print("\n7. Testing predictions on new texts...")
    test_texts = [
        "This movie is absolutely wonderful and amazing",
        "I really enjoyed watching this film",
        "This was a terrible waste of my time",
        "Boring and disappointing movie",
        "Great performances by the entire cast",
    ]
    
    print("\n   Predictions:")
    print("   " + "=" * 50)
    for text in test_texts:
        sentiment, confidence = predict_sentiment(model, tokenizer, text, max_len)
        print(f"\n   Text: '{text}'")
        print(f"   Sentiment: {sentiment} (Confidence: {confidence*100:.2f}%)")
    
    # Save model
    print("\n8. Saving model...")
    model.save('/tmp/sentiment_analysis_model.h5')
    print("   Model saved to /tmp/sentiment_analysis_model.h5")
    
    print("\n" + "=" * 60)
    print("Sentiment analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
