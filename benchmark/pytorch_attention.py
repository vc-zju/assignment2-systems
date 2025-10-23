import torch
import torch.cuda
import math
import timeit
import itertools
from typing import Tuple
from jaxtyping import Float
from torch import Tensor
from einops import einsum
from cs336_basics.nn_utils import softmax


def scaled_dot_product_attention(
    Q: Float[Tensor, "batch seq_len d_model"],
    K: Float[Tensor, "batch seq_len d_model"], 
    V: Float[Tensor, "batch seq_len d_model"],
) -> Float[Tensor, "batch seq_len d_model"]:
    """Scaled dot-product attention without multihead (single head attention).
    
    Args:
        Q: Query tensor of shape (batch_size, seq_len, d_model)
        K: Key tensor of shape (batch_size, seq_len, d_model)
        V: Value tensor of shape (batch_size, seq_len, d_model)
    
    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores: Q @ K^T / sqrt(d_k)
    attention_scores = einsum(Q, K, "batch query d_k, batch key d_k -> batch query key") / math.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(attention_scores, dim=-1)
    
    # Apply attention weights to values
    output = einsum(attention_weights, V, "batch query key, batch key d_v -> batch query d_v")
    
    return output


def benchmark_attention_forward(Q: Tensor, K: Tensor, V: Tensor, num_warmup: int = 10, num_iters: int = 100) -> float:
    """Benchmark forward pass of attention."""
    # Warmup iterations
    for _ in range(num_warmup):
        _ = scaled_dot_product_attention(Q, K, V)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timed iterations
    times = []
    for _ in range(num_iters):
        start_time = timeit.default_timer()
        output = scaled_dot_product_attention(Q, K, V)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    
    # Return mean time
    return sum(times) / len(times)


def benchmark_attention_backward(Q: Tensor, K: Tensor, V: Tensor, num_warmup: int = 10, num_iters: int = 100) -> Tuple[float, float]:
    """Benchmark backward pass of attention and measure memory usage."""
    # Ensure tensors require gradients
    Q_grad = Q.clone().requires_grad_(True)
    K_grad = K.clone().requires_grad_(True) 
    V_grad = V.clone().requires_grad_(True)
    
    # Warmup iterations
    for _ in range(num_warmup):
        output = scaled_dot_product_attention(Q_grad, K_grad, V_grad)
        loss = output.sum()
        loss.backward()
        # Clear gradients
        Q_grad.grad = None
        K_grad.grad = None
        V_grad.grad = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Measure memory usage before backward pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Run one forward pass to measure memory before backward
    output = scaled_dot_product_attention(Q_grad, K_grad, V_grad)
    loss = output.sum()
    
    if torch.cuda.is_available():
        memory_before_backward = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    else:
        memory_before_backward = 0.0
    
    # Timed backward iterations
    times = []
    for _ in range(num_iters):
        # Forward pass (not timed)
        output = scaled_dot_product_attention(Q_grad, K_grad, V_grad)
        loss = output.sum()
        
        # Timed backward pass
        start_time = timeit.default_timer()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        
        # Clear gradients for next iteration
        Q_grad.grad = None
        K_grad.grad = None
        V_grad.grad = None
    
    # Return mean time and memory usage
    return sum(times) / len(times), memory_before_backward


def create_random_inputs(batch_size: int, seq_len: int, d_model: int, device: str = "cuda") -> Tuple[Tensor, Tensor, Tensor]:
    """Create random input tensors Q, K, V."""
    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    return Q, K, V


def main():
    """Main benchmarking function."""
    # Fixed parameters
    batch_size = 8
    d_model_options = [16, 32, 64, 128]
    seq_len_options = [256, 1024, 4096, 8192, 16384]
    num_warmup = 3
    num_iters = 10
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Results storage
    results = []
    
    print(f"Benchmarking attention with batch_size={batch_size}")
    print(f"d_model options: {d_model_options}")
    print(f"seq_len options: {seq_len_options}")
    print(f"Warmup iterations: {num_warmup}, Timed iterations: {num_iters}")
    print("-" * 80)
    
    # Iterate through all combinations
    for d_model, seq_len in itertools.product(d_model_options, seq_len_options):
        print(f"\nTesting d_model={d_model}, seq_len={seq_len}")
        
        try:
            # Create random inputs
            Q, K, V = create_random_inputs(batch_size, seq_len, d_model, device)
            
            # Benchmark forward pass
            forward_time = benchmark_attention_forward(Q, K, V, num_warmup, num_iters)
            
            # Benchmark backward pass and measure memory
            backward_time, memory_usage = benchmark_attention_backward(Q, K, V, num_warmup, num_iters)
            
            # Store results
            result = {
                'd_model': d_model,
                'seq_len': seq_len,
                'forward_time_ms': forward_time * 1000,
                'backward_time_ms': backward_time * 1000,
                'memory_mb': memory_usage
            }
            results.append(result)
            
            # Print results
            print(f"  Forward time:  {forward_time*1000:.3f} ms")
            print(f"  Backward time: {backward_time*1000:.3f} ms") 
            print(f"  Memory usage:  {memory_usage:.2f} MB")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            # Try to free memory and continue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Print summary table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"{'d_model':<8} {'seq_len':<8} {'Forward (ms)':<12} {'Backward (ms)':<13} {'Memory (MB)':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['d_model']:<8} {result['seq_len']:<8} {result['forward_time_ms']:<12.3f} "
              f"{result['backward_time_ms']:<13.3f} {result['memory_mb']:<12.2f}")


if __name__ == "__main__":
    main()
