import argparse
import math
import cs336_basics.model
import timeit
import torch
import torch.cuda.nvtx as nvtx

from typing import Callable

def benchmark_model(description: str, num_warmup_iters: int, num_iters: int, function: Callable, *args, **kwargs) -> tuple[float, float]:
        for i in range(num_warmup_iters):
            with nvtx.range(f"Warmup {description} {i}"):
                function(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        times: list[float] = []
        for i in range(num_iters):
            start_time = timeit.default_timer()
            with nvtx.range(f"Iteration {description} {i}"):
                function(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
        mean_time: float = sum(times) / len(times)
        std_dev: float = math.sqrt(sum((time - mean_time) ** 2 for time in times) / len(times))
        print(f"{description} took {mean_time:.6f} seconds per iteration ± {std_dev:.6f} seconds")

def benchmark_backward_only(description: str, num_warmup_iters: int, num_iters: int, model, input_data) -> tuple[float, float]:
        for i in range(num_warmup_iters):
            # Execute forward pass first (not included in timing)
            output = model(input_data)
            loss = output.sum()
            # Only benchmark the backward pass
            with nvtx.range(f"Warmup {description} {i}"):
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        times: list[float] = []
        for i in range(num_iters):
            # Execute forward pass first (not included in timing)
            output = model(input_data)
            loss = output.sum()
            # Only measure backward pass time
            start_time = timeit.default_timer()
            with nvtx.range(f"Iteration {description} {i}"):
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
        
        mean_time: float = sum(times) / len(times)
        std_dev: float = math.sqrt(sum((time - mean_time) ** 2 for time in times) / len(times))
        print(f"{description} took {mean_time:.6f} seconds per iteration ± {std_dev:.6f} seconds")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="Benchmarking model")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-warmup-iters", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()
    model: cs336_basics.model.BasicsTransformerLM = cs336_basics.model.BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    )
    if torch.cuda.is_available():
        model.to(torch.cuda.current_device())
    input_data: torch.Tensor = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))
    if torch.cuda.is_available():
        input_data = input_data.to(torch.cuda.current_device())
    benchmark_model("Forward pass", args.num_warmup_iters, args.num_iters, model, input_data)
    
    benchmark_backward_only("Backward pass", args.num_warmup_iters, args.num_iters, model, input_data)

if __name__ == "__main__":
    main()