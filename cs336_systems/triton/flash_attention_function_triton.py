import math
import torch 

from .flash_fwd_kernel import flash_fwd_kernel

class FlashAttentionFunctionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        q_tile_size: int = 16
        k_tile_size: int = 16
        
        # Support both 2D (N, D) and 3D (B, N, D) tensors
        if Q.dim() == 2:
            Q = Q.unsqueeze(0)
            K = K.unsqueeze(0) 
            V = V.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        # Now assume tensors have shape (B, N, D)
        assert Q.shape[2] == K.shape[2] and K.shape[2] == V.shape[2]
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()

        batch_size: int = Q.shape[0]
        n_q: int = Q.shape[1]
        d: int = Q.shape[2]
        n_k: int = K.shape[1]

        O: torch.Tensor = torch.empty_like(Q)
        L: torch.Tensor = torch.empty((Q.shape[0], Q.shape[1]), device=Q.device, dtype=Q.dtype)

        scale: float = math.sqrt(d)

        flash_fwd_kernel[(math.ceil(n_q / q_tile_size), batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_q, n_k, scale, is_causal,
            d, q_tile_size, k_tile_size
        )

        # If input was 2D, squeeze the batch dimension from output
        if squeeze_output:
            O = O.squeeze(0)
            L = L.squeeze(0)
        
        ctx.save_for_backward(Q, K, L)
        ctx.is_causal = is_causal
            
        return O

    @staticmethod
    def backward(ctx, dout):
        raise NotImplementedError

def main():
    torch.manual_seed(123)
    Q = torch.randn(1, 16, 32)
    K = torch.randn(1, 16, 32)
    V = torch.randn(1, 16, 32)
    Q = Q.to("cuda:0")
    K = K.to("cuda:0")
    V = V.to("cuda:0")
    # Test both regular and causal attention
    O_regular = FlashAttentionFunctionTriton.apply(Q, K, V, False)
    O_causal = FlashAttentionFunctionTriton.apply(Q, K, V, True)
    print(f"Regular output shape: {O_regular.shape}")
    print(f"Causal output shape: {O_causal.shape}")

if __name__ == "__main__":
    main()
