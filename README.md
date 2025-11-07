# Deep Learning: Theory and Practice

A comprehensive repository covering the fundamental concepts of deep learning with practical implementations. Each concept is organized into its own project directory with theory, implementations, and examples.

## 📚 Repository Structure

This repository is organized into four main sections, each focusing on a different deep learning architecture:

### 1. 🧠 [Artificial Neural Networks (ANN)](./01_ANN/)
Basic building blocks of deep learning - feedforward neural networks

**Key Topics:**
- Forward and backward propagation
- Gradient descent optimization
- Activation functions (ReLU, Sigmoid, Tanh)
- Loss functions and regularization

**Implementations:**
- From-scratch implementation using NumPy
- TensorFlow/Keras implementation
- PyTorch implementation

### 2. 👁️ [Convolutional Neural Networks (CNN)](./02_CNN/)
Specialized networks for processing grid-like data (images)

**Key Topics:**
- Convolutional layers and filters
- Pooling operations
- Batch normalization
- Transfer learning
- Popular architectures (VGG, ResNet, Inception)

**Implementations:**
- Image classification (MNIST, CIFAR-10)
- Custom architecture design
- Transfer learning with pre-trained models

### 3. 🔄 [Recurrent Neural Networks (RNN)](./03_RNN/)
Networks designed for sequential data processing

**Key Topics:**
- Sequential data processing
- LSTM and GRU cells
- Bidirectional RNNs
- Sequence-to-sequence models
- Attention mechanisms

**Implementations:**
- Basic RNN, LSTM, and GRU
- Text generation
- Sentiment analysis
- Time series prediction

### 4. 🤖 [Transformer Architecture](./04_Transformer/)
State-of-the-art architecture based on attention mechanisms

**Key Topics:**
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Encoder-decoder architecture
- Pre-training and fine-tuning (BERT, GPT)

**Implementations:**
- Transformer from scratch
- Text classification
- Machine translation
- Vision Transformers

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/algsoch/deep_learning.git
cd deep_learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Examples

Each project directory contains standalone Python scripts that can be run independently:

```bash
# ANN Examples
python 01_ANN/ann_from_scratch.py
python 01_ANN/ann_tensorflow.py
python 01_ANN/ann_pytorch.py

# CNN Examples
python 02_CNN/cnn_image_classification.py
python 02_CNN/cnn_custom_architecture.py

# RNN Examples
python 03_RNN/rnn_basic.py
python 03_RNN/lstm_text_generation.py

# Transformer Examples
python 04_Transformer/transformer_from_scratch.py
```

## 📖 Learning Path

For beginners, we recommend following this learning path:

1. **Start with ANNs** - Understand the basics of neural networks
2. **Move to CNNs** - Learn about spatial feature extraction
3. **Explore RNNs** - Understand sequential data processing
4. **Master Transformers** - Learn state-of-the-art architectures

Each directory contains a README with detailed explanations and theory.

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow 2.x** - High-level deep learning framework
- **PyTorch** - Dynamic deep learning framework
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **scikit-learn** - Machine learning utilities

## 📊 Project Features

- ✅ Complete implementations from scratch
- ✅ Framework-based implementations (TensorFlow, PyTorch)
- ✅ Detailed comments and documentation
- ✅ Visualization of results
- ✅ Comparison of different architectures
- ✅ Real-world datasets and examples

## 🎯 Learning Objectives

By working through this repository, you will:

- Understand core deep learning concepts
- Implement neural networks from scratch
- Use popular deep learning frameworks
- Design custom architectures
- Apply deep learning to real-world problems
- Understand state-of-the-art models

## 📝 Additional Resources

Each project directory contains:
- **README.md** - Theory and concepts
- **Implementation files** - Working code examples
- **Comments** - Detailed explanations in code

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new examples
- Improve documentation
- Fix bugs
- Suggest new features

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

This repository is designed for learning and understanding deep learning fundamentals. It includes implementations inspired by research papers and educational resources in the field.

---

**Happy Learning! 🎓**
