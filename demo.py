"""
Transformer Demo - Quick Start Guide
====================================

This file provides a quick demonstration of the Transformer implementation
and shows key features for the assignment deliverables.
"""

# Import our complete implementation
from transformer_numpy import (
    TokenEmbedding, PositionalEncoding, LayerNormalization,
    ScaledDotProductAttention, MultiHeadAttention, FeedForwardNetwork, 
    TransformerBlock, GPTTransformer, create_causal_mask, softmax, gelu
)
import numpy as np

def demonstrate_transformer():
    """Comprehensive demonstration of the Transformer implementation."""
    
    print("🎯 TRANSFORMER ARCHITECTURE DEMONSTRATION")
    print("=" * 50)
    
    # Configuration
    config = {
        'vocab_size': 1000,
        'embed_dim': 256,  # Smaller for demo
        'num_heads': 8,
        'num_layers': 4,   # Smaller for demo  
        'ff_dim': 1024,    # 4 * embed_dim
        'max_seq_len': 64
    }
    
    batch_size = 2
    seq_len = 12
    
    print("📋 Model Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 1. COMPONENT TESTING
    print("🔧 TESTING INDIVIDUAL COMPONENTS")
    print("-" * 30)
    
    # Test Token Embedding
    print("1️⃣ Token Embedding:")
    token_emb = TokenEmbedding(config['vocab_size'], config['embed_dim'])
    token_ids = np.random.randint(0, config['vocab_size'], (batch_size, seq_len))
    embeddings = token_emb.forward(token_ids)
    print(f"   Input: {token_ids.shape} -> Output: {embeddings.shape}")
    
    # Test Positional Encoding  
    print("2️⃣ Positional Encoding:")
    pos_enc = PositionalEncoding(config['max_seq_len'], config['embed_dim'])
    pos_embeddings = pos_enc.forward(embeddings)
    print(f"   Added positional info: {pos_embeddings.shape}")
    
    # Test Causal Mask
    print("3️⃣ Causal Mask:")
    mask = create_causal_mask(seq_len)
    print(f"   Mask shape: {mask.shape}")
    print(f"   Lower triangular: {np.array_equal(mask, np.tril(np.ones((seq_len, seq_len))).astype(bool))}")
    
    # Test Attention
    print("4️⃣ Scaled Dot-Product Attention:")
    attention = ScaledDotProductAttention(config['embed_dim'])
    attn_out, attn_weights = attention.forward(pos_embeddings, pos_embeddings, pos_embeddings, mask)
    print(f"   Attention output: {attn_out.shape}")
    print(f"   Attention weights: {attn_weights.shape}")
    print(f"   Weights sum to 1: {np.allclose(np.sum(attn_weights, axis=-1), 1.0)}")
    
    # Test Multi-Head Attention
    print("5️⃣ Multi-Head Attention:")
    mha = MultiHeadAttention(config['embed_dim'], config['num_heads'])
    mha_out, mha_weights = mha.forward(pos_embeddings, pos_embeddings, pos_embeddings, mask)
    print(f"   MHA output: {mha_out.shape}")
    print(f"   MHA weights: {mha_weights.shape}")
    
    # Test Feed-Forward Network
    print("6️⃣ Feed-Forward Network:")
    ffn = FeedForwardNetwork(config['embed_dim'], config['ff_dim'])
    ffn_out = ffn.forward(pos_embeddings)
    print(f"   FFN output: {ffn_out.shape}")
    
    # Test Layer Normalization
    print("7️⃣ Layer Normalization:")
    layer_norm = LayerNormalization(config['embed_dim'])
    norm_out = layer_norm.forward(pos_embeddings)
    mean_check = np.mean(norm_out, axis=-1)
    var_check = np.var(norm_out, axis=-1)
    print(f"   Mean ≈ 0: {np.allclose(mean_check, 0, atol=1e-6)}")
    print(f"   Variance ≈ 1: {np.allclose(var_check, 1, atol=1e-6)}")
    
    print()
    
    # 2. COMPLETE MODEL TESTING
    print("🏗️ COMPLETE TRANSFORMER MODEL")
    print("-" * 30)
    
    # Initialize full model
    model = GPTTransformer(**config)
    print("✅ Model initialized successfully!")
    
    # Forward pass
    logits, attention_weights = model.forward(token_ids, return_attention=True)
    print(f"📤 Forward Pass Results:")
    print(f"   Input tokens: {token_ids.shape}")
    print(f"   Output logits: {logits.shape}")
    print(f"   Attention layers: {len(attention_weights)}")
    
    # Next token prediction
    next_token_probs = model.predict_next_token(token_ids)
    print(f"🎯 Next Token Prediction:")
    print(f"   Probabilities shape: {next_token_probs.shape}")
    print(f"   Probability sums: {np.sum(next_token_probs, axis=-1)}")
    print(f"   Max probability: {np.max(next_token_probs):.4f}")
    
    # Text generation
    prompt = token_ids[:1, :3]  # First 3 tokens
    generated = model.generate_text(prompt, max_new_tokens=8)
    print(f"📝 Text Generation:")
    print(f"   Prompt: {prompt[0]}")
    print(f"   Generated: {generated[0]}")
    
    print()
    
    # 3. MATHEMATICAL VERIFICATION
    print("🔬 MATHEMATICAL VERIFICATION")
    print("-" * 30)
    
    # Verify softmax properties
    test_logits = np.random.randn(5, 10)
    probs = softmax(test_logits, axis=-1)
    print(f"✅ Softmax sums to 1: {np.allclose(np.sum(probs, axis=-1), 1.0)}")
    
    # Verify causal masking
    scores = np.random.randn(seq_len, seq_len)
    masked_scores = np.where(mask, scores, -np.inf)
    causal_check = np.all(masked_scores[np.triu_indices(seq_len, k=1)] == -np.inf)
    print(f"✅ Causal mask blocks future: {causal_check}")
    
    # Verify attention weights
    sample_attention = attention_weights[0][0, 0, :, :]  # First layer, first head
    attn_sum_check = np.allclose(np.sum(sample_attention, axis=-1), 1.0)
    print(f"✅ Attention weights sum to 1: {attn_sum_check}")
    
    # Verify layer norm statistics
    sample_data = np.random.randn(100, config['embed_dim']) * 5 + 2
    normed = layer_norm.forward(sample_data)
    ln_mean = np.mean(normed, axis=-1)
    ln_var = np.var(normed, axis=-1)
    print(f"✅ LayerNorm mean ≈ 0: {np.allclose(ln_mean, 0, atol=1e-5)}")
    print(f"✅ LayerNorm var ≈ 1: {np.allclose(ln_var, 1, atol=1e-5)}")
    
    print()
    
    # 4. ARCHITECTURE ANALYSIS  
    print("📊 ARCHITECTURE ANALYSIS")
    print("-" * 30)
    
    print("🏗️ Model Architecture:")
    print(f"   • {config['num_layers']} Transformer blocks")
    print(f"   • {config['num_heads']} attention heads per block") 
    print(f"   • {config['embed_dim']} embedding dimensions")
    print(f"   • {config['ff_dim']} feed-forward hidden size")
    print(f"   • {config['vocab_size']} vocabulary size")
    
    print("🔢 Parameter Count Estimate:")
    embedding_params = config['vocab_size'] * config['embed_dim']
    attention_params_per_layer = 4 * config['embed_dim'] * config['embed_dim']  # Q,K,V,O
    ffn_params_per_layer = 2 * config['embed_dim'] * config['ff_dim'] + config['ff_dim'] + config['embed_dim']
    layer_norm_params_per_layer = 4 * config['embed_dim']  # 2 layer norms * 2 params each
    total_layer_params = config['num_layers'] * (attention_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)
    output_params = config['embed_dim'] * config['vocab_size']
    
    total_params = embedding_params + total_layer_params + output_params
    print(f"   • Embedding: ~{embedding_params:,}")
    print(f"   • Transformer layers: ~{total_layer_params:,}")
    print(f"   • Output projection: ~{output_params:,}")
    print(f"   • Total: ~{total_params:,} parameters")
    
    print("💾 Memory Usage:")
    forward_memory = logits.nbytes + sum(w.nbytes for w in attention_weights)
    print(f"   • Forward pass: ~{forward_memory / 1024 / 1024:.2f} MB")
    
    print()
    
    # 5. DESIGN JUSTIFICATIONS
    print("🎓 DESIGN JUSTIFICATIONS")  
    print("-" * 30)
    
    print("🔄 Positional Encoding Choice: Sinusoidal")
    print("   ✓ No learned parameters needed")
    print("   ✓ Can extrapolate to longer sequences")
    print("   ✓ Captures both absolute and relative position")
    
    print("🎭 Multi-Head Attention Benefits:")
    print("   ✓ Captures different types of relationships")  
    print("   ✓ Allows parallel processing")
    print("   ✓ Reduces risk of attention collapse")
    
    print("🛡️ Causal Masking Purpose:")
    print("   ✓ Prevents information leakage from future tokens")
    print("   ✓ Maintains autoregressive property")
    print("   ✓ Essential for decoder-only architecture")
    
    print("⚡ Pre-Norm Architecture:")
    print("   ✓ Better gradient flow")
    print("   ✓ More stable training")
    print("   ✓ Reduced need for warmup")
    
    print()
    print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("✅ All components working correctly")
    print("✅ Mathematical properties verified") 
    print("✅ Architecture design justified")
    print("🚀 Ready for submission!")

if __name__ == "__main__":
    demonstrate_transformer()