# Recurrent Neural Network (RNN) Project

## Overview
This project demonstrates the implementation of Recurrent Neural Networks designed for sequential data processing.

## Theory
RNNs maintain internal state (memory) to process sequences:
- **Recurrent Connections**: Output feeds back as input for next time step
- **LSTM (Long Short-Term Memory)**: Addresses vanishing gradient problem
- **GRU (Gated Recurrent Unit)**: Simplified version of LSTM
- **Bidirectional RNN**: Process sequences in both directions
- **Sequence-to-Sequence**: Encoder-decoder architecture

## Project Structure
- `rnn_basic.py`: Basic RNN implementation
- `lstm_text_generation.py`: Text generation using LSTM
- `sentiment_analysis.py`: Sentiment classification with RNN
- `time_series_prediction.py`: Stock price/weather prediction
- `sequence_to_sequence.py`: Translation or summarization tasks

## Use Cases
- Natural language processing
- Time series prediction
- Speech recognition
- Machine translation
- Music generation
- Video analysis

## How to Run
```bash
# Basic RNN
python rnn_basic.py

# LSTM text generation
python lstm_text_generation.py

# Sentiment analysis
python sentiment_analysis.py

# Time series prediction
python time_series_prediction.py
```

## Key Concepts Demonstrated
1. Sequential data processing
2. LSTM and GRU cells
3. Bidirectional processing
4. Attention mechanism basics
5. Sequence padding and masking
6. Teacher forcing
