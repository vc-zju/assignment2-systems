import argparse
import math
import cs336_basics.model
import cs336_basics.optimizer
import timeit
import torch
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext
from jaxtyping import Float, Bool
from einops import einsum
from torch import Tensor
from cs336_basics.nn_utils import softmax

from typing import Callable

def benchmark_model(description: str, num_warmup_iters: int, num_iters: int, context_manager, profile_memory: bool, function: Callable, *args, **kwargs) -> tuple[float, float]:
        for i in range(num_warmup_iters):
            with nvtx.range(f"Warmup {description} {i}"):
                with context_manager:
                    function(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        times: list[float] = []
        if profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        for i in range(num_iters):
            start_time = timeit.default_timer()
            with nvtx.range(f"Iteration {description} {i}"):
                with context_manager:
                    function(*args, **kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
        if profile_memory:
            torch.cuda.memory._dump_snapshot(f"{description}.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
        mean_time: float = sum(times) / len(times)
        std_dev: float = math.sqrt(sum((time - mean_time) ** 2 for time in times) / len(times))
        print(f"{description} took {mean_time:.6f} seconds per iteration ± {std_dev:.6f} seconds")

def benchmark_train_step_nvtx(description: str, num_warmup_iters: int, num_iters: int, model, input_data, optimizer):
        # Warmup iterations
        for i in range(1, num_warmup_iters + 1):
            with nvtx.range(f"{description} warmup iter {i}"):
                # Execute forward pass first (not included in timing)
                output = model(input_data)
                loss = output.sum()
                optimizer.zero_grad()
                loss.backward()
                lr = cs336_basics.optimizer.get_cosine_lr(i, 1e-2, 1e-3, 1000, 5000)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
        # Formal Iteration
        for i in range(1, num_iters + 1):
            with nvtx.range(f"{description} formal iter {i} forward"):
            # Execute forward pass first (not included in timing)
                output = model(input_data)
                loss = output.sum()

            with nvtx.range(f"{description} formal iter {i} backward"):
                optimizer.zero_grad()
                loss.backward()

            with nvtx.range(f"{description} formal iter {i} optimizer"):
                lr = cs336_basics.optimizer.get_cosine_lr(i, 1e-2, 1e-3, 1000, 5000)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()

def benchmark_train_step(description: str, num_warmup_iters: int, num_iters: int, model, input_data, optimizer, context_manager, profile_memory: bool):
        # Warmup iterations
        for i in range(1, num_warmup_iters + 1):
            with context_manager:
                output = model(input_data)
                loss = output.sum()
                optimizer.zero_grad()
                loss.backward()
                lr = cs336_basics.optimizer.get_cosine_lr(i, 1e-2, 1e-3, 1000, 5000)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
        # Formal Iteration
        forward_times: list[float] = []
        backward_times: list[float] = []
        optimizer_times: list[float] = []
        if profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        for i in range(1, num_iters + 1):
            with context_manager:
                forward_start_time = timeit.default_timer()
                output = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_end_time = timeit.default_timer()
                forward_times.append(forward_end_time - forward_start_time)
                loss = output.sum()
                backward_start_time = timeit.default_timer()
                optimizer.zero_grad()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                backward_end_time = timeit.default_timer()
                backward_times.append(backward_end_time - backward_start_time)
                optimizer_start_time = timeit.default_timer()
                lr = cs336_basics.optimizer.get_cosine_lr(i, 1e-2, 1e-3, 1000, 5000)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_end_time = timeit.default_timer()
                optimizer_times.append(optimizer_end_time - optimizer_start_time)
        if profile_memory:
            torch.cuda.memory._dump_snapshot(f"{description}_train_step.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
        mean_forward_time: float = sum(forward_times) / len(forward_times)
        std_dev_forward: float = math.sqrt(sum((time - mean_forward_time) ** 2 for time in forward_times) / len(forward_times))
        mean_backward_time: float = sum(backward_times) / len(backward_times)
        std_dev_backward: float = math.sqrt(sum((time - mean_backward_time) ** 2 for time in backward_times) / len(backward_times))
        mean_optimizer_time: float = sum(optimizer_times) / len(optimizer_times)
        std_dev_optimizer: float = math.sqrt(sum((time - mean_optimizer_time) ** 2 for time in optimizer_times) / len(optimizer_times))
        print(f"{description} took {mean_forward_time:.6f} seconds per iteration ± {std_dev_forward:.6f} seconds for forward pass")
        print(f"{description} took {mean_backward_time:.6f} seconds per iteration ± {std_dev_backward:.6f} seconds for backward pass")
        print(f"{description} took {mean_optimizer_time:.6f} seconds per iteration ± {std_dev_optimizer:.6f} seconds for optimizer step")

@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("attention_scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    with nvtx.range("attention_scores_mask"):
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("computing output"):
        result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    return result


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
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
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
    optimizer: cs336_basics.optimizer.AdamW = cs336_basics.optimizer.AdamW(model.parameters())
    context_manager = torch.autocast(device_type="cuda") if args.use_mixed_precision else nullcontext()
    benchmark_model(f"{args.description}_forward_pass", args.num_warmup_iters, args.num_iters, context_manager, args.profile_memory, model, input_data)
    benchmark_train_step(args.description, args.num_warmup_iters, args.num_iters, model, input_data, optimizer, context_manager, args.profile_memory)


if __name__ == "__main__":
    main()
