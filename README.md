# Transformer Architecture Implementation from Scratch

This repository contains a complete implementation of a GPT-style decoder-only Transformer architecture using only NumPy for mathematical operations. This is an individual assignment for NLP course focusing on understanding the internal mechanics of Transformers.

## Features

- **Token Embedding**: Maps token IDs to dense vector representations
- **Positional Encoding**: Sinusoidal encoding to add positional information
- **Scaled Dot-Product Attention**: Core attention mechanism with softmax normalization
- **Multi-Head Attention**: Parallel attention heads with concatenation and projection
- **Feed-Forward Network**: Two-layer FFN with GELU activation
- **Layer Normalization**: Pre-norm architecture with residual connections
- **Causal Masking**: Prevents attention to future tokens in decoder-only setup
- **Output Layer**: Projects to vocabulary size with softmax for next token prediction

## Requirements

- Python 3.7+
- NumPy 1.21.0+

## Installation

1. Clone this repository:
```bash
git clone <your-github-repo-url>
cd TransformerModel
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import numpy as np
from transformer_numpy import GPTTransformer

# Model configuration
config = {
    'vocab_size': 1000,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 6,
    'ff_dim': 2048,
    'max_seq_len': 128
}

# Initialize model
model = GPTTransformer(**config)

# Sample input (batch of token sequences)
batch_size = 2
seq_len = 10
token_ids = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))

# Forward pass
logits, attention_weights = model.forward(token_ids)

print(f"Output logits shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
print(f"Next token probabilities shape: {logits[:, -1, :].shape}")  # [batch_size, vocab_size]
```

### Running Tests

```python
python transformer_numpy.py
```

This will run basic tests for all components and verify tensor dimensions.

### Running the Test Suite

```python
python test_transformer.py
```

## Architecture Details

### Model Components

1. **Token Embedding**: Learnable embedding matrix that maps discrete tokens to continuous vectors
2. **Positional Encoding**: Sinusoidal encoding that adds position information without learnable parameters
3. **Multi-Head Attention**: 8 parallel attention heads that capture different types of relationships
4. **Feed-Forward Network**: Two linear transformations with GELU activation in between
5. **Layer Normalization**: Applied before each sub-layer (pre-norm) with residual connections
6. **Causal Masking**: Upper triangular mask ensuring autoregressive property

### Key Design Decisions

- **Positional Encoding**: Chose sinusoidal encoding for its ability to extrapolate to longer sequences
- **Activation Function**: Used GELU instead of ReLU for smoother gradients
- **Normalization**: Pre-norm architecture for better gradient flow
- **Attention**: Scaled dot-product with causal masking for decoder-only setup

## Testing

The implementation includes comprehensive tests for:

- ✅ Tensor dimension verification
- ✅ Softmax probability validation (sums to 1)
- ✅ Causal mask correctness
- ✅ Layer normalization properties
- ✅ Attention weight distributions
- ✅ Forward pass completeness

## Project Structure

```
TransformerModel/
├── transformer_numpy.py      # Main implementation
├── test_transformer.py       # Comprehensive test suite
├── requirements.txt          # Dependencies
├── README.md                # This file
└── examples/                # Usage examples (coming soon)
```

## Mathematical Foundation

### Attention Mechanism
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Layer Normalization
```
LayerNorm(x) = γ * (x - μ) / √(σ² + ε) + β
```

## Performance Characteristics

- **Memory**: O(n²) for attention computation where n is sequence length
- **Time Complexity**: O(n²d + nd²) per layer where d is embedding dimension
- **Scalability**: Efficient for sequences up to 512 tokens on standard hardware

## Limitations

- No gradient computation (forward pass only)
- No training capabilities (weights are randomly initialized)
- No optimization for very long sequences
- CPU-only implementation (no GPU acceleration)

## Contributing

This is an individual academic assignment. Please do not submit pull requests.

## License

Academic use only - Individual Assignment for NLP Course

## Author

[Your Name]  
NLP Course - Semester 7  
[Your University]