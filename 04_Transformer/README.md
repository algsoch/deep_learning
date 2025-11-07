# Transformer Architecture Project

## Overview
This project demonstrates the implementation of the Transformer architecture, which revolutionized NLP and beyond.

## Theory
Transformers rely entirely on attention mechanisms:
- **Self-Attention**: Compute relationships between all positions in a sequence
- **Multi-Head Attention**: Parallel attention mechanisms for different representations
- **Positional Encoding**: Add position information to embeddings
- **Encoder-Decoder Architecture**: For sequence-to-sequence tasks
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: Stabilize training

## Project Structure
- `transformer_from_scratch.py`: Transformer implementation from scratch
- `text_classification_transformer.py`: Transformer for classification
- `machine_translation.py`: Translation using Transformer
- `bert_fine_tuning.py`: Fine-tuning BERT for downstream tasks
- `gpt_text_generation.py`: Text generation with GPT-style models
- `vision_transformer.py`: Vision Transformer (ViT) for images

## Use Cases
- Machine translation
- Text summarization
- Question answering
- Text generation
- Image classification (Vision Transformer)
- Protein structure prediction

## How to Run
```bash
# Transformer from scratch
python transformer_from_scratch.py

# Text classification
python text_classification_transformer.py

# Machine translation
python machine_translation.py

# BERT fine-tuning
python bert_fine_tuning.py
```

## Key Concepts Demonstrated
1. Self-attention mechanism
2. Multi-head attention
3. Positional encoding
4. Encoder and decoder stacks
5. Masked attention
6. Pre-training and fine-tuning (BERT, GPT)
7. Vision Transformers
