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


def benchmark_attention_forward(attention_fn, Q: Tensor, K: Tensor, V: Tensor, num_warmup: int = 10, num_iters: int = 100) -> float:
    """Benchmark forward pass of attention."""
    # Warmup iterations
    for _ in range(num_warmup):
        _ = attention_fn(Q, K, V)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timed iterations
    times = []
    for _ in range(num_iters):
        start_time = timeit.default_timer()
        output = attention_fn(Q, K, V)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    
    # Return mean time
    return sum(times) / len(times)


def benchmark_attention_backward(attention_fn, Q: Tensor, K: Tensor, V: Tensor, num_warmup: int = 10, num_iters: int = 100) -> Tuple[float, float]:
    """Benchmark backward pass of attention and measure memory usage."""
    # Ensure tensors require gradients
    Q_grad = Q.clone().requires_grad_(True)
    K_grad = K.clone().requires_grad_(True) 
    V_grad = V.clone().requires_grad_(True)
    
    # Warmup iterations
    for _ in range(num_warmup):
        output = attention_fn(Q_grad, K_grad, V_grad)
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
    output = attention_fn(Q_grad, K_grad, V_grad)
    loss = output.sum()
    
    if torch.cuda.is_available():
        memory_before_backward = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    else:
        memory_before_backward = 0.0
    
    # Timed backward iterations
    times = []
    for _ in range(num_iters):
        # Forward pass (not timed)
        output = attention_fn(Q_grad, K_grad, V_grad)
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
    """Main benchmarking function comparing compiled vs uncompiled attention."""
    # Fixed parameters
    batch_size = 8
    d_model_options = [16, 32, 64, 128]
    seq_len_options = [256, 1024, 4096, 8192, 16384]
    num_warmup = 3
    num_iters = 10
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create compiled version of attention function
    compiled_attention = torch.compile(scaled_dot_product_attention)
    
    # Results storage
    results = []
    
    print(f"Benchmarking attention (compiled vs uncompiled) with batch_size={batch_size}")
    print(f"d_model options: {d_model_options}")
    print(f"seq_len options: {seq_len_options}")
    print(f"Warmup iterations: {num_warmup}, Timed iterations: {num_iters}")
    print("=" * 100)
    
    # Iterate through all combinations
    for d_model, seq_len in itertools.product(d_model_options, seq_len_options):
        print(f"\nTesting d_model={d_model}, seq_len={seq_len}")
        
        try:
            # Create random inputs
            Q, K, V = create_random_inputs(batch_size, seq_len, d_model, device)
            
            # Benchmark UNCOMPILED version
            print("  Running uncompiled version...")
            uncompiled_forward_time = benchmark_attention_forward(
                scaled_dot_product_attention, Q, K, V, num_warmup, num_iters
            )
            uncompiled_backward_time, uncompiled_memory = benchmark_attention_backward(
                scaled_dot_product_attention, Q, K, V, num_warmup, num_iters
            )
            
            # Benchmark COMPILED version  
            print("  Running compiled version...")
            compiled_forward_time = benchmark_attention_forward(
                compiled_attention, Q, K, V, num_warmup, num_iters
            )
            compiled_backward_time, compiled_memory = benchmark_attention_backward(
                compiled_attention, Q, K, V, num_warmup, num_iters
            )
            
            # Calculate speedups
            forward_speedup = uncompiled_forward_time / compiled_forward_time if compiled_forward_time > 0 else 0
            backward_speedup = uncompiled_backward_time / compiled_backward_time if compiled_backward_time > 0 else 0
            
            # Store results
            result = {
                'd_model': d_model,
                'seq_len': seq_len,
                'uncompiled_forward_ms': uncompiled_forward_time * 1000,
                'compiled_forward_ms': compiled_forward_time * 1000,
                'forward_speedup': forward_speedup,
                'uncompiled_backward_ms': uncompiled_backward_time * 1000,
                'compiled_backward_ms': compiled_backward_time * 1000,
                'backward_speedup': backward_speedup,
                'uncompiled_memory_mb': uncompiled_memory,
                'compiled_memory_mb': compiled_memory
            }
            results.append(result)
            
            # Print results
            print(f"    Uncompiled - Forward: {uncompiled_forward_time*1000:.3f} ms, Backward: {uncompiled_backward_time*1000:.3f} ms")
            print(f"    Compiled   - Forward: {compiled_forward_time*1000:.3f} ms, Backward: {compiled_backward_time*1000:.3f} ms")
            print(f"    Speedup    - Forward: {forward_speedup:.2f}x, Backward: {backward_speedup:.2f}x")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            # Try to free memory and continue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Print comprehensive comparison table
    print("\n" + "=" * 120)
    print("ATTENTION COMPILATION COMPARISON RESULTS")
    print("=" * 120)
    print(f"{'Config':<15} {'Uncompiled Forward':<18} {'Compiled Forward':<16} {'Forward':<8} "
          f"{'Uncompiled Backward':<19} {'Compiled Backward':<17} {'Backward':<8}")
    print(f"{'(d_model, seq_len)':<15} {'(ms)':<18} {'(ms)':<16} {'Speedup':<8} "
          f"{'(ms)':<19} {'(ms)':<17} {'Speedup':<8}")
    print("-" * 120)
    
    for result in results:
        config = f"({result['d_model']}, {result['seq_len']})"
        print(f"{config:<15} {result['uncompiled_forward_ms']:<18.3f} {result['compiled_forward_ms']:<16.3f} "
              f"{result['forward_speedup']:<8.2f} {result['uncompiled_backward_ms']:<19.3f} "
              f"{result['compiled_backward_ms']:<17.3f} {result['backward_speedup']:<8.2f}")
    
    # Print summary statistics
    if results:
        avg_forward_speedup = sum(r['forward_speedup'] for r in results) / len(results)
        avg_backward_speedup = sum(r['backward_speedup'] for r in results) / len(results)
        print(f"\nAverage Speedups:")
        print(f"  Forward pass:  {avg_forward_speedup:.2f}x")
        print(f"  Backward pass: {avg_backward_speedup:.2f}x")


if __name__ == "__main__":
    main()
