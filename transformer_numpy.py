"""
Transformer Architecture Implementation from Scratch using NumPy
================================================================

This module implements a complete GPT-style decoder-only Transformer
architecture using only NumPy for mathematical operations.

Author: Your Name
Course: NLP - Individual Assignment
"""

import numpy as np
from typing import Tuple, Optional, List
import math


class TokenEmbedding:
    """Token embedding layer that maps token IDs to dense vectors."""
    
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 42):
        """
        Initialize token embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embedding vectors
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize embedding matrix with small random values
        self.embedding_matrix = np.random.normal(0, 0.02, (vocab_size, embed_dim))
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass of token embedding.
        
        Args:
            token_ids: Token IDs of shape [batch_size, seq_len]
            
        Returns:
            Embedded tokens of shape [batch_size, seq_len, embed_dim]
        """
        return self.embedding_matrix[token_ids]


class PositionalEncoding:
    """Sinusoidal positional encoding for adding position information."""
    
    def __init__(self, max_len: int, embed_dim: int):
        """
        Initialize positional encoding.
        
        Args:
            max_len: Maximum sequence length
            embed_dim: Dimension of embeddings
        """
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        self.pos_encoding = self._create_positional_encoding()
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding matrix."""
        pos_encoding = np.zeros((self.max_len, self.embed_dim))
        
        position = np.arange(0, self.max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.embed_dim, 2, dtype=np.float32) * 
                         -(math.log(10000.0) / self.embed_dim))
        
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Embeddings with positional encoding added
        """
        batch_size, seq_len, embed_dim = x.shape
        return x + self.pos_encoding[:seq_len, :]


class LayerNormalization:
    """Layer normalization implementation."""
    
    def __init__(self, embed_dim: int, eps: float = 1e-6):
        """
        Initialize layer normalization.
        
        Args:
            embed_dim: Dimension of embeddings
            eps: Small constant for numerical stability
        """
        self.embed_dim = embed_dim
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape [..., embed_dim]
            
        Returns:
            Normalized tensor
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask to prevent attention to future tokens.
    
    Args:
        seq_len: Length of sequence
        
    Returns:
        Causal mask of shape [seq_len, seq_len]
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask == 0  # True where attention is allowed


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation.
    
    Args:
        x: Input array
        axis: Axis along which to apply softmax
        
    Returns:
        Softmax probabilities
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation function implementation.
    
    Args:
        x: Input array
        
    Returns:
        GELU activated values
    """
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class ScaledDotProductAttention:
    """Scaled dot-product attention mechanism."""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        """
        Initialize attention mechanism.
        
        Args:
            embed_dim: Dimension of embeddings
            dropout: Dropout rate (not implemented in this version)
        """
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)
        self.dropout = dropout
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute scaled dot-product attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            mask: Optional causal mask [seq_len, seq_len]
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        scores = np.matmul(query, key.transpose(0, 2, 1)) / self.scale
        
        # Apply causal mask if provided
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        
        # Apply softmax to get attention weights
        attention_weights = softmax(scores, axis=-1)
        
        # Apply attention to values
        attention_output = np.matmul(attention_weights, value)
        
        return attention_output, attention_weights


class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, seed: int = 42):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout rate (not implemented in this version)
            seed: Random seed for reproducibility
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        np.random.seed(seed)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        # Initialize projection matrices for Q, K, V
        self.W_q = np.random.normal(0, 0.02, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, 0.02, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, 0.02, (embed_dim, embed_dim))
        
        # Output projection matrix
        self.W_o = np.random.normal(0, 0.02, (embed_dim, embed_dim))
        
        # Initialize attention mechanism
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            mask: Optional causal mask [seq_len, seq_len]
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        batch_size, seq_len, embed_dim = query.shape
        
        # Linear projections for Q, K, V
        Q = np.matmul(query, self.W_q)  # [batch_size, seq_len, embed_dim]
        K = np.matmul(key, self.W_k)    # [batch_size, seq_len, embed_dim]
        V = np.matmul(value, self.W_v)  # [batch_size, seq_len, embed_dim]
        
        # Reshape to separate heads: [batch_size, seq_len, num_heads, head_dim]
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose to: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Apply attention to each head
        attention_outputs = []
        attention_weights_list = []
        
        for i in range(self.num_heads):
            head_output, head_weights = self.attention.forward(
                Q[:, i, :, :],  # [batch_size, seq_len, head_dim]
                K[:, i, :, :],  # [batch_size, seq_len, head_dim]
                V[:, i, :, :],  # [batch_size, seq_len, head_dim]
                mask
            )
            attention_outputs.append(head_output)
            attention_weights_list.append(head_weights)
        
        # Concatenate all heads: [batch_size, seq_len, embed_dim]
        concat_output = np.concatenate(attention_outputs, axis=-1)
        
        # Final output projection
        output = np.matmul(concat_output, self.W_o)
        
        # Stack attention weights: [batch_size, num_heads, seq_len, seq_len]
        attention_weights = np.stack(attention_weights_list, axis=1)
        
        return output, attention_weights


class FeedForwardNetwork:
    """Position-wise feed-forward network with two linear transformations."""
    
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1, seed: int = 42):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Dimension of input embeddings
            ff_dim: Dimension of hidden layer (typically 4 * embed_dim)
            dropout: Dropout rate (not implemented in this version)
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Initialize weight matrices
        self.W1 = np.random.normal(0, 0.02, (embed_dim, ff_dim))
        self.b1 = np.zeros(ff_dim)
        
        self.W2 = np.random.normal(0, 0.02, (ff_dim, embed_dim))
        self.b2 = np.zeros(embed_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        # First linear transformation + GELU activation
        hidden = np.matmul(x, self.W1) + self.b1
        hidden = gelu(hidden)
        
        # Second linear transformation
        output = np.matmul(hidden, self.W2) + self.b2
        
        return output


class TransformerBlock:
    """Complete transformer block with multi-head attention and feed-forward network."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, 
                 dropout: float = 0.1, seed: int = 42):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feed-forward network hidden layer
            dropout: Dropout rate (not implemented in this version)
            seed: Random seed for reproducibility
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        # Initialize sub-layers
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout, seed)
        self.feed_forward = FeedForwardNetwork(embed_dim, ff_dim, dropout, seed + 1)
        
        # Layer normalization layers
        self.layer_norm1 = LayerNormalization(embed_dim)
        self.layer_norm2 = LayerNormalization(embed_dim)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of transformer block with pre-norm architecture.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional causal mask [seq_len, seq_len]
            
        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        # Pre-norm multi-head attention with residual connection
        normalized_x = self.layer_norm1.forward(x)
        attn_output, attn_weights = self.multi_head_attention.forward(
            normalized_x, normalized_x, normalized_x, mask
        )
        x = x + attn_output  # Residual connection
        
        # Pre-norm feed-forward network with residual connection
        normalized_x = self.layer_norm2.forward(x)
        ff_output = self.feed_forward.forward(normalized_x)
        x = x + ff_output  # Residual connection
        
        return x, attn_weights


class GPTTransformer:
    """Complete GPT-style decoder-only transformer model."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, ff_dim: int, max_seq_len: int = 512, 
                 dropout: float = 0.1, seed: int = 42):
        """
        Initialize GPT transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads per layer
            num_layers: Number of transformer layers
            ff_dim: Dimension of feed-forward network (typically 4 * embed_dim)
            max_seq_len: Maximum sequence length
            dropout: Dropout rate (not implemented in this version)
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        
        # Initialize components
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim, seed)
        self.positional_encoding = PositionalEncoding(max_seq_len, embed_dim)
        
        # Transformer layers
        self.transformer_blocks = []
        for i in range(num_layers):
            block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout, seed + i + 1)
            self.transformer_blocks.append(block)
        
        # Final layer normalization
        self.final_layer_norm = LayerNormalization(embed_dim)
        
        # Output projection to vocabulary
        np.random.seed(seed + num_layers + 1)
        self.output_projection = np.random.normal(0, 0.02, (embed_dim, vocab_size))
    
    def forward(self, token_ids: np.ndarray, 
                return_attention: bool = False) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Forward pass of GPT transformer.
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (logits, attention_weights_list)
            - logits: [batch_size, seq_len, vocab_size]
            - attention_weights_list: List of attention weights from each layer
        """
        batch_size, seq_len = token_ids.shape
        
        # Create causal mask
        causal_mask = create_causal_mask(seq_len)
        
        # Token embedding + positional encoding
        x = self.token_embedding.forward(token_ids)
        x = self.positional_encoding.forward(x)
        
        # Pass through transformer blocks
        attention_weights_list = []
        
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block.forward(x, causal_mask)
            if return_attention:
                attention_weights_list.append(attn_weights)
        
        # Final layer normalization
        x = self.final_layer_norm.forward(x)
        
        # Project to vocabulary size
        logits = np.matmul(x, self.output_projection)
        
        return logits, (attention_weights_list if return_attention else None)
    
    def predict_next_token(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Get probability distribution for next token prediction.
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            
        Returns:
            Next token probabilities [batch_size, vocab_size]
        """
        logits, _ = self.forward(token_ids)
        
        # Get logits for the last position (next token prediction)
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Apply softmax to get probabilities
        next_token_probs = softmax(next_token_logits, axis=-1)
        
        return next_token_probs
    
    def generate_text(self, prompt_tokens: np.ndarray, max_new_tokens: int = 50, 
                      temperature: float = 1.0) -> np.ndarray:
        """
        Generate text autoregressively.
        
        Args:
            prompt_tokens: Initial prompt tokens [1, prompt_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated sequence [1, prompt_len + generated_len]
        """
        generated = prompt_tokens.copy()
        
        for _ in range(max_new_tokens):
            # Get next token probabilities
            next_probs = self.predict_next_token(generated)
            
            # Apply temperature
            if temperature != 1.0:
                logits = np.log(next_probs + 1e-8) / temperature
                next_probs = softmax(logits, axis=-1)
            
            # Sample next token (greedy for simplicity)
            next_token = np.argmax(next_probs, axis=-1).reshape(1, 1)
            
            # Append to sequence
            generated = np.concatenate([generated, next_token], axis=1)
            
            # Stop if sequence becomes too long
            if generated.shape[1] >= self.max_seq_len:
                break
        
        return generated


if __name__ == "__main__":
    print("🚀 Transformer Implementation from Scratch using NumPy")
    print("=" * 60)
    
    # Model configuration
    config = {
        'vocab_size': 1000,
        'embed_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'ff_dim': 2048,
        'max_seq_len': 128
    }
    
    batch_size = 2
    seq_len = 10
    
    print(f"Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize complete model
    print("🔧 Initializing GPT Transformer model...")
    model = GPTTransformer(**config)
    print("✅ Model initialized successfully!")
    
    # Generate sample input
    token_ids = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))
    print(f"\n📝 Sample input shape: {token_ids.shape}")
    print(f"Sample tokens: {token_ids[0][:5]}...")
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    logits, attention_weights = model.forward(token_ids, return_attention=True)
    
    print(f"✅ Forward pass completed!")
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Number of attention layers: {len(attention_weights)}")
    print(f"   Attention weights shape per layer: {attention_weights[0].shape}")
    
    # Test next token prediction
    print("\n🎯 Testing next token prediction...")
    next_token_probs = model.predict_next_token(token_ids)
    print(f"   Next token probabilities shape: {next_token_probs.shape}")
    print(f"   Probability sum (should be ~1.0): {np.sum(next_token_probs[0]):.6f}")
    print(f"   Max probability: {np.max(next_token_probs[0]):.6f}")
    
    # Test text generation
    print("\n📖 Testing text generation...")
    prompt = token_ids[:1, :3]  # Use first 3 tokens as prompt
    generated = model.generate_text(prompt, max_new_tokens=5)
    print(f"   Prompt: {prompt[0]}")
    print(f"   Generated: {generated[0]}")
    print(f"   Generated length: {generated.shape[1]}")
    
    # Verify tensor dimensions and properties
    print("\n🧪 Running verification tests...")
    
    # Test 1: Causal mask
    mask = create_causal_mask(seq_len)
    lower_triangular = np.tril(np.ones((seq_len, seq_len)))
    assert np.array_equal(mask, lower_triangular.astype(bool)), "Causal mask test failed"
    print("   ✅ Causal mask is correctly lower triangular")
    
    # Test 2: Softmax probabilities sum to 1
    test_logits = np.random.randn(batch_size, seq_len, config['vocab_size'])
    probs = softmax(test_logits, axis=-1)
    sums = np.sum(probs, axis=-1)
    assert np.allclose(sums, 1.0), "Softmax test failed"
    print("   ✅ Softmax probabilities sum to 1.0")
    
    # Test 3: Layer normalization properties
    test_input = np.random.randn(batch_size, seq_len, config['embed_dim'])
    layer_norm = LayerNormalization(config['embed_dim'])
    normalized = layer_norm.forward(test_input)
    mean = np.mean(normalized, axis=-1)
    var = np.var(normalized, axis=-1)
    assert np.allclose(mean, 0, atol=1e-6), "Layer norm mean test failed"
    assert np.allclose(var, 1, atol=1e-6), "Layer norm variance test failed"
    print("   ✅ Layer normalization: mean ≈ 0, variance ≈ 1")
    
    # Test 4: Multi-head attention dimensions
    mha = MultiHeadAttention(config['embed_dim'], config['num_heads'])
    attn_out, attn_weights = mha.forward(test_input, test_input, test_input, mask)
    assert attn_out.shape == test_input.shape, "MHA output shape test failed"
    expected_attn_shape = (batch_size, config['num_heads'], seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, "MHA weights shape test failed"
    print("   ✅ Multi-head attention dimensions are correct")
    
    # Test 5: Feed-forward network dimensions
    ffn = FeedForwardNetwork(config['embed_dim'], config['ff_dim'])
    ff_out = ffn.forward(test_input)
    assert ff_out.shape == test_input.shape, "FFN output shape test failed"
    print("   ✅ Feed-forward network dimensions are correct")
    
    print(f"\n🎉 All tests passed! Implementation is working correctly.")
    print(f"🔗 Ready for deployment and further testing.")
    
    # Display model statistics
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: ~{(config['vocab_size'] * config['embed_dim'] + config['embed_dim'] * config['vocab_size']):,}")
    print(f"   Memory per forward pass: ~{logits.nbytes / 1024 / 1024:.2f} MB")
    print(f"   Attention patterns captured: {len(attention_weights) * config['num_heads']}")
    
    print(f"\n🏁 Transformer implementation demo completed successfully!")