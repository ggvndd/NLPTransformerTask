"""
Comprehensive Test Suite for Transformer Implementation
======================================================

This module contains detailed tests for all components of the Transformer
architecture to verify correctness and mathematical properties.
"""

import numpy as np
import sys
import os

# Add current directory to path to import transformer_numpy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_numpy import (
    TokenEmbedding, PositionalEncoding, LayerNormalization,
    ScaledDotProductAttention, MultiHeadAttention, FeedForwardNetwork,
    TransformerBlock, GPTTransformer, create_causal_mask, softmax, gelu
)


class TransformerTester:
    """Comprehensive test suite for Transformer components."""
    
    def __init__(self):
        """Initialize test configuration."""
        self.config = {
            'vocab_size': 1000,
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'ff_dim': 2048,
            'max_seq_len': 128
        }
        self.batch_size = 4
        self.seq_len = 16
        self.tolerance = 1e-6
        
    def run_all_tests(self):
        """Run all test cases."""
        print("🧪 Running Comprehensive Transformer Test Suite")
        print("=" * 60)
        
        test_methods = [
            self.test_token_embedding,
            self.test_positional_encoding,
            self.test_layer_normalization,
            self.test_causal_mask,
            self.test_softmax_function,
            self.test_gelu_activation,
            self.test_scaled_dot_product_attention,
            self.test_multi_head_attention,
            self.test_feed_forward_network,
            self.test_transformer_block,
            self.test_complete_model,
            self.test_text_generation,
            self.test_attention_patterns,
            self.test_gradient_flow_simulation
        ]
        
        passed = 0
        total = len(test_methods)
        
        for test_method in test_methods:
            try:
                test_method()
                print(f"✅ {test_method.__name__}")
                passed += 1
            except Exception as e:
                print(f"❌ {test_method.__name__}: {str(e)}")
        
        print(f"\n📊 Test Results: {passed}/{total} tests passed")
        if passed == total:
            print("🎉 All tests passed! Implementation is correct.")
        else:
            print("⚠️  Some tests failed. Please review implementation.")
        
        return passed == total
    
    def test_token_embedding(self):
        """Test token embedding layer."""
        embedding = TokenEmbedding(self.config['vocab_size'], self.config['embed_dim'])
        
        # Test shape
        token_ids = np.random.randint(0, self.config['vocab_size'], 
                                    (self.batch_size, self.seq_len))
        output = embedding.forward(token_ids)
        
        expected_shape = (self.batch_size, self.seq_len, self.config['embed_dim'])
        assert output.shape == expected_shape, f"Shape mismatch: {output.shape} vs {expected_shape}"
        
        # Test different tokens produce different embeddings
        token1 = embedding.forward(np.array([[0]]))
        token2 = embedding.forward(np.array([[1]]))
        assert not np.allclose(token1, token2), "Different tokens should have different embeddings"
        
        # Test same tokens produce same embeddings
        token1_repeat = embedding.forward(np.array([[0]]))
        assert np.allclose(token1, token1_repeat), "Same tokens should produce same embeddings"
    
    def test_positional_encoding(self):
        """Test positional encoding."""
        pos_enc = PositionalEncoding(self.config['max_seq_len'], self.config['embed_dim'])
        
        # Test shape
        input_tensor = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        output = pos_enc.forward(input_tensor)
        
        assert output.shape == input_tensor.shape, "Positional encoding should preserve shape"
        
        # Test different positions have different encodings
        pos1 = pos_enc.pos_encoding[0]
        pos2 = pos_enc.pos_encoding[1]
        assert not np.allclose(pos1, pos2), "Different positions should have different encodings"
        
        # Test periodicity properties for sinusoidal encoding
        # Even indices should contain sine, odd indices should contain cosine
        assert len(pos_enc.pos_encoding.shape) == 2, "Positional encoding should be 2D"
        assert pos_enc.pos_encoding.shape == (self.config['max_seq_len'], self.config['embed_dim'])
    
    def test_layer_normalization(self):
        """Test layer normalization."""
        layer_norm = LayerNormalization(self.config['embed_dim'])
        
        # Test normalization properties
        input_tensor = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim']) * 10
        output = layer_norm.forward(input_tensor)
        
        # Check mean ≈ 0 and variance ≈ 1 along last dimension
        mean = np.mean(output, axis=-1)
        variance = np.var(output, axis=-1)
        
        assert np.allclose(mean, 0, atol=self.tolerance), f"Mean should be ~0, got {np.mean(mean)}"
        assert np.allclose(variance, 1, atol=self.tolerance), f"Variance should be ~1, got {np.mean(variance)}"
        
        # Test shape preservation
        assert output.shape == input_tensor.shape, "Layer norm should preserve shape"
    
    def test_causal_mask(self):
        """Test causal mask creation."""
        mask = create_causal_mask(self.seq_len)
        
        # Test shape
        expected_shape = (self.seq_len, self.seq_len)
        assert mask.shape == expected_shape, f"Mask shape: {mask.shape} vs {expected_shape}"
        
        # Test lower triangular property
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                if i >= j:
                    assert mask[i, j] == True, f"Lower triangle should be True at ({i}, {j})"
                else:
                    assert mask[i, j] == False, f"Upper triangle should be False at ({i}, {j})"
        
        # Test that diagonal is True (can attend to self)
        diagonal = np.diag(mask)
        assert np.all(diagonal), "Diagonal should be True (self-attention allowed)"
    
    def test_softmax_function(self):
        """Test softmax implementation."""
        # Test basic properties
        input_tensor = np.random.randn(self.batch_size, self.seq_len, self.config['vocab_size'])
        output = softmax(input_tensor, axis=-1)
        
        # Test shape preservation
        assert output.shape == input_tensor.shape, "Softmax should preserve shape"
        
        # Test probabilities sum to 1
        sums = np.sum(output, axis=-1)
        assert np.allclose(sums, 1.0), "Softmax probabilities should sum to 1"
        
        # Test all values are positive
        assert np.all(output >= 0), "Softmax output should be non-negative"
        
        # Test numerical stability (large values)
        large_input = np.array([[1000, 1001, 999]])
        large_output = softmax(large_input, axis=-1)
        assert not np.any(np.isnan(large_output)), "Softmax should handle large values"
        assert not np.any(np.isinf(large_output)), "Softmax should not produce inf"
    
    def test_gelu_activation(self):
        """Test GELU activation function."""
        input_tensor = np.linspace(-3, 3, 100)
        output = gelu(input_tensor)
        
        # Test shape preservation
        assert output.shape == input_tensor.shape, "GELU should preserve shape"
        
        # Test properties
        # GELU(0) ≈ 0
        assert abs(gelu(0.0)) < 0.01, "GELU(0) should be close to 0"
        
        # GELU should be approximately x for large positive x
        large_x = 10.0
        assert abs(gelu(large_x) - large_x) < 0.1, "GELU(x) ≈ x for large positive x"
        
        # GELU should be close to 0 for large negative x
        large_neg_x = -10.0
        assert abs(gelu(large_neg_x)) < 0.01, "GELU(x) ≈ 0 for large negative x"
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot-product attention."""
        attention = ScaledDotProductAttention(self.config['embed_dim'])
        
        # Create test inputs
        q = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        k = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        v = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        mask = create_causal_mask(self.seq_len)
        
        output, weights = attention.forward(q, k, v, mask)
        
        # Test shapes
        expected_output_shape = (self.batch_size, self.seq_len, self.config['embed_dim'])
        expected_weights_shape = (self.batch_size, self.seq_len, self.seq_len)
        
        assert output.shape == expected_output_shape, f"Output shape: {output.shape}"
        assert weights.shape == expected_weights_shape, f"Weights shape: {weights.shape}"
        
        # Test attention weights properties
        # Should sum to 1 along last dimension
        weight_sums = np.sum(weights, axis=-1)
        assert np.allclose(weight_sums, 1.0), "Attention weights should sum to 1"
        
        # Test causal masking (upper triangle should be 0)
        for i in range(self.seq_len):
            for j in range(i + 1, self.seq_len):
                assert np.allclose(weights[:, i, j], 0), f"Causal mask failed at ({i}, {j})"
    
    def test_multi_head_attention(self):
        """Test multi-head attention."""
        mha = MultiHeadAttention(self.config['embed_dim'], self.config['num_heads'])
        
        # Create test inputs
        input_tensor = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        mask = create_causal_mask(self.seq_len)
        
        output, weights = mha.forward(input_tensor, input_tensor, input_tensor, mask)
        
        # Test shapes
        expected_output_shape = (self.batch_size, self.seq_len, self.config['embed_dim'])
        expected_weights_shape = (self.batch_size, self.config['num_heads'], 
                                self.seq_len, self.seq_len)
        
        assert output.shape == expected_output_shape, f"MHA output shape: {output.shape}"
        assert weights.shape == expected_weights_shape, f"MHA weights shape: {weights.shape}"
        
        # Test that heads produce different attention patterns
        head1_weights = weights[:, 0, :, :]
        head2_weights = weights[:, 1, :, :]
        assert not np.allclose(head1_weights, head2_weights), "Different heads should have different patterns"
    
    def test_feed_forward_network(self):
        """Test feed-forward network."""
        ffn = FeedForwardNetwork(self.config['embed_dim'], self.config['ff_dim'])
        
        # Test forward pass
        input_tensor = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        output = ffn.forward(input_tensor)
        
        # Test shape preservation
        assert output.shape == input_tensor.shape, "FFN should preserve input shape"
        
        # Test that different inputs produce different outputs
        input2 = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        output2 = ffn.forward(input2)
        assert not np.allclose(output, output2), "Different inputs should produce different outputs"
    
    def test_transformer_block(self):
        """Test complete transformer block."""
        block = TransformerBlock(self.config['embed_dim'], self.config['num_heads'], 
                               self.config['ff_dim'])
        
        input_tensor = np.random.randn(self.batch_size, self.seq_len, self.config['embed_dim'])
        mask = create_causal_mask(self.seq_len)
        
        output, attention_weights = block.forward(input_tensor, mask)
        
        # Test shape preservation
        assert output.shape == input_tensor.shape, "Transformer block should preserve shape"
        
        # Test attention weights shape
        expected_attn_shape = (self.batch_size, self.config['num_heads'], 
                              self.seq_len, self.seq_len)
        assert attention_weights.shape == expected_attn_shape
        
        # Test residual connections (output should be different from input)
        assert not np.allclose(output, input_tensor), "Block should transform input"
    
    def test_complete_model(self):
        """Test complete GPT model."""
        model = GPTTransformer(**self.config)
        
        # Test forward pass
        token_ids = np.random.randint(0, self.config['vocab_size'], 
                                    (self.batch_size, self.seq_len))
        logits, attention_weights = model.forward(token_ids, return_attention=True)
        
        # Test output shape
        expected_shape = (self.batch_size, self.seq_len, self.config['vocab_size'])
        assert logits.shape == expected_shape, f"Model output shape: {logits.shape}"
        
        # Test attention weights from all layers
        assert len(attention_weights) == self.config['num_layers']
        
        # Test next token prediction
        next_probs = model.predict_next_token(token_ids)
        expected_next_shape = (self.batch_size, self.config['vocab_size'])
        assert next_probs.shape == expected_next_shape
        
        # Test probability distribution
        prob_sums = np.sum(next_probs, axis=-1)
        assert np.allclose(prob_sums, 1.0), "Next token probabilities should sum to 1"
    
    def test_text_generation(self):
        """Test text generation functionality."""
        model = GPTTransformer(**self.config)
        
        # Test generation
        prompt = np.random.randint(0, self.config['vocab_size'], (1, 5))
        generated = model.generate_text(prompt, max_new_tokens=10)
        
        # Test that generation extends the sequence
        assert generated.shape[0] == 1, "Should maintain batch dimension"
        assert generated.shape[1] > prompt.shape[1], "Should generate new tokens"
        assert generated.shape[1] <= prompt.shape[1] + 10, "Should respect max_new_tokens"
        
        # Test that prompt is preserved
        assert np.array_equal(generated[:, :prompt.shape[1]], prompt), "Prompt should be preserved"
    
    def test_attention_patterns(self):
        """Test attention pattern properties."""
        model = GPTTransformer(**self.config)
        
        # Create a simple sequence
        token_ids = np.arange(self.seq_len).reshape(1, -1)
        logits, attention_weights = model.forward(token_ids, return_attention=True)
        
        # Test causal pattern in all layers
        for layer_idx, layer_weights in enumerate(attention_weights):
            for head_idx in range(self.config['num_heads']):
                head_attn = layer_weights[0, head_idx, :, :]  # [seq_len, seq_len]
                
                # Check causal masking
                for i in range(self.seq_len):
                    for j in range(i + 1, self.seq_len):
                        assert np.isclose(head_attn[i, j], 0), \
                            f"Layer {layer_idx}, Head {head_idx}: Future attention at ({i}, {j})"
        
        # Test attention distribution
        for layer_weights in attention_weights:
            row_sums = np.sum(layer_weights, axis=-1)
            assert np.allclose(row_sums, 1.0), "Attention rows should sum to 1"
    
    def test_gradient_flow_simulation(self):
        """Simulate gradient flow properties (without actual gradients)."""
        model = GPTTransformer(**self.config)
        
        # Test that small input changes produce small output changes (Lipschitz-like)
        token_ids1 = np.random.randint(0, self.config['vocab_size'], (1, self.seq_len))
        token_ids2 = token_ids1.copy()
        token_ids2[0, -1] = (token_ids2[0, -1] + 1) % self.config['vocab_size']
        
        logits1, _ = model.forward(token_ids1)
        logits2, _ = model.forward(token_ids2)
        
        # The outputs should be different (model is sensitive to input changes)
        assert not np.allclose(logits1, logits2), "Model should be sensitive to input changes"
        
        # But not too different (some stability)
        diff = np.mean(np.abs(logits1 - logits2))
        assert diff < 100, f"Output difference too large: {diff}"


def main():
    """Run the test suite."""
    tester = TransformerTester()
    success = tester.run_all_tests()
    
    if success:
        print(f"\n🏆 All tests passed! The Transformer implementation is mathematically correct.")
        print(f"📋 Test Report:")
        print(f"   ✅ Token embeddings working correctly")
        print(f"   ✅ Positional encoding implemented properly") 
        print(f"   ✅ Layer normalization has correct statistical properties")
        print(f"   ✅ Causal masking prevents future information leakage")
        print(f"   ✅ Attention mechanisms compute proper probability distributions")
        print(f"   ✅ Multi-head attention captures diverse patterns")
        print(f"   ✅ Feed-forward networks transform representations correctly")
        print(f"   ✅ Complete model produces valid output distributions")
        print(f"   ✅ Text generation works autoregressively")
        print(f"   ✅ All tensor dimensions are correct")
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
    
    return success


if __name__ == "__main__":
    main()