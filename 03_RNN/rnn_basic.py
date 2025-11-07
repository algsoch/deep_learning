"""
Recurrent Neural Network (RNN) Basic Implementation
Demonstrates RNN, LSTM, and GRU for sequence processing
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


class RNNModels:
    """Collection of different RNN architectures"""
    
    @staticmethod
    def simple_rnn(input_shape, num_classes):
        """Simple RNN model"""
        model = models.Sequential([
            layers.SimpleRNN(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.SimpleRNN(32),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ], name='SimpleRNN')
        
        return model
    
    @staticmethod
    def lstm_model(input_shape, num_classes):
        """LSTM model"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ], name='LSTM')
        
        return model
    
    @staticmethod
    def gru_model(input_shape, num_classes):
        """GRU model"""
        model = models.Sequential([
            layers.GRU(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.GRU(64),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ], name='GRU')
        
        return model
    
    @staticmethod
    def bidirectional_lstm(input_shape, num_classes):
        """Bidirectional LSTM model"""
        model = models.Sequential([
            layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ], name='BiLSTM')
        
        return model
    
    @staticmethod
    def stacked_lstm(input_shape, num_classes):
        """Stacked LSTM with multiple layers"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ], name='StackedLSTM')
        
        return model


def create_sequence_data(n_samples=1000, seq_length=50, n_features=1):
    """
    Create synthetic sequence classification data
    Sequences with sum > threshold are class 1, else class 0
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Generate random sequence
        seq = np.random.randn(seq_length, n_features)
        
        # Label based on sequence characteristics
        # If mean of sequence > 0, class 1, else class 0
        label = 1 if np.mean(seq) > 0 else 0
        
        X.append(seq)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def create_sine_wave_prediction_data(n_samples=1000, seq_length=50):
    """Create data for time series prediction"""
    np.random.seed(42)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Generate sine wave
        t = np.linspace(0, 10, seq_length + 1)
        signal = np.sin(freq * t + phase) + np.random.randn(seq_length + 1) * 0.1
        
        X.append(signal[:-1].reshape(-1, 1))
        y.append(signal[-1])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def train_and_evaluate(model, X_train, y_train, X_test, y_test, task='classification', epochs=20):
    """Train and evaluate a model"""
    print(f"\n{'='*60}")
    print(f"Training {model.name}")
    print(f"{'='*60}")
    
    # Compile model
    if task == 'classification':
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    print(f"\nModel Summary:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    if task == 'classification':
        test_loss, test_metric = model.evaluate(X_test, y_test, verbose=0)
        metric_name = 'Accuracy'
        metric_value = test_metric * 100
        print(f"\n{model.name} Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test {metric_name}: {metric_value:.2f}%")
    else:
        test_loss, test_metric = model.evaluate(X_test, y_test, verbose=0)
        metric_name = 'MAE'
        metric_value = test_metric
        print(f"\n{model.name} Results:")
        print(f"  Test Loss (MSE): {test_loss:.4f}")
        print(f"  Test {metric_name}: {metric_value:.4f}")
    
    return history, test_loss, metric_value


def plot_comparison(results, task='classification'):
    """Plot comparison of different models"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    names = list(results.keys())
    
    # Plot training curves
    for name in names:
        history = results[name]['history']
        if task == 'classification':
            axes[0].plot(history.history['val_accuracy'], label=name)
        else:
            axes[0].plot(history.history['val_loss'], label=name)
    
    axes[0].set_xlabel('Epoch')
    if task == 'classification':
        axes[0].set_ylabel('Validation Accuracy')
        axes[0].set_title('Validation Accuracy Comparison')
    else:
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Validation Loss Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Bar chart for final metrics
    if task == 'classification':
        metrics = [results[name]['test_metric'] for name in names]
        axes[1].bar(names, metrics, color='skyblue')
        axes[1].set_ylabel('Test Accuracy (%)')
        axes[1].set_title('Test Accuracy Comparison')
    else:
        metrics = [results[name]['test_metric'] for name in names]
        axes[1].bar(names, metrics, color='lightcoral')
        axes[1].set_ylabel('Test MAE')
        axes[1].set_title('Test MAE Comparison')
    
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'/tmp/rnn_comparison_{task}.png')
    print(f"\nComparison plot saved to /tmp/rnn_comparison_{task}.png")


def main():
    """Main function"""
    print("=" * 60)
    print("Recurrent Neural Network (RNN) Models Comparison")
    print("=" * 60)
    
    # Classification task
    print("\n" + "=" * 60)
    print("TASK 1: Sequence Classification")
    print("=" * 60)
    
    print("\nGenerating sequence classification data...")
    X_class, y_class = create_sequence_data(n_samples=2000, seq_length=50, n_features=1)
    
    # Split data
    split = int(0.8 * len(X_class))
    X_train_class = X_class[:split]
    y_train_class = y_class[:split]
    X_test_class = X_class[split:]
    y_test_class = y_class[split:]
    
    print(f"Training samples: {X_train_class.shape[0]}")
    print(f"Test samples: {X_test_class.shape[0]}")
    print(f"Sequence length: {X_train_class.shape[1]}")
    
    # Create models
    models_class = {
        'SimpleRNN': RNNModels.simple_rnn((50, 1), 2),
        'LSTM': RNNModels.lstm_model((50, 1), 2),
        'GRU': RNNModels.gru_model((50, 1), 2),
        'BiLSTM': RNNModels.bidirectional_lstm((50, 1), 2),
    }
    
    # Train and compare
    results_class = {}
    for name, model in models_class.items():
        history, test_loss, test_metric = train_and_evaluate(
            model, X_train_class, y_train_class, X_test_class, y_test_class,
            task='classification', epochs=15
        )
        results_class[name] = {
            'history': history,
            'test_loss': test_loss,
            'test_metric': test_metric
        }
    
    # Plot results
    plot_comparison(results_class, task='classification')
    
    print("\n" + "=" * 60)
    print("All RNN experiments complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
