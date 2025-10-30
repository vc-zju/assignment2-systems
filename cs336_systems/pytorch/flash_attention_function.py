import math
import torch 

class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
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
        batch_size: int = Q.shape[0]
        n_q: int = Q.shape[1]
        n_k: int = K.shape[1]
        d: int = Q.shape[2]
        t_q: int = math.ceil(n_q / q_tile_size)
        t_k: int = math.ceil(n_k / k_tile_size)
        
        O: torch.Tensor = torch.zeros_like(Q)
        L: torch.Tensor = torch.empty((batch_size, n_q), device=Q.device, dtype=Q.dtype)
        
        # Split along sequence dimension (dim=1)
        q_chunks: tuple[torch.Tensor] = torch.split(Q, q_tile_size, dim=1)
        o_chunks: tuple[torch.Tensor] = torch.split(O, q_tile_size, dim=1)
        l_chunks: tuple[torch.Tensor] = torch.split(L, q_tile_size, dim=1)
        k_chunks: tuple[torch.Tensor] = torch.split(K, k_tile_size, dim=1)
        v_chunks: tuple[torch.Tensor] = torch.split(V, k_tile_size, dim=1)
        
        for i in range(t_q):
            actual_q_size = q_chunks[i].shape[1]  # Actual chunk size (may be smaller than q_tile_size)
            
            l_i: torch.Tensor = torch.zeros((batch_size, actual_q_size), device=Q.device, dtype=Q.dtype)
            m_i: torch.Tensor = torch.full((batch_size, actual_q_size), float('-inf'), device=Q.device, dtype=Q.dtype)
            o_i: torch.Tensor = torch.zeros((batch_size, actual_q_size, d), device=Q.device, dtype=Q.dtype)
            
            for j in range(t_k):
                # Batch matrix multiplication: (B, N_q, D) @ (B, D, N_k) -> (B, N_q, N_k)
                s: torch.Tensor = torch.bmm(q_chunks[i], k_chunks[j].transpose(-2, -1)) / math.sqrt(d)
                last_m_i = m_i.clone()
                # Get max along last dimension for each batch
                m_i = torch.maximum(m_i, torch.max(s, dim=-1)[0])
                # Subtract max for numerical stability
                p = torch.exp(s - m_i.unsqueeze(-1))
                m_exp_compensate: torch.Tensor = torch.exp(last_m_i - m_i)
                l_i = m_exp_compensate * l_i + torch.sum(p, dim=-1)
                
                # Update output: use batch matrix multiplication
                o_i = torch.bmm(torch.diag_embed(m_exp_compensate), o_i) + torch.bmm(p, v_chunks[j])
            # Store results back to chunks
            o_chunks[i][:] = torch.bmm(torch.diag_embed(1 / l_i), o_i)
            l_chunks[i][:] = m_i + torch.log(l_i)
            
        # If input was 2D, squeeze the batch dimension from output
        if squeeze_output:
            O = O.squeeze(0)
            L = L.squeeze(0)
        
        ctx.save_for_backward(Q, K, L)
            
        return O

    @staticmethod
    def backward(ctx, dout):
        raise NotImplementedError


def main():
    torch.manual_seed(123)
    Q = torch.randn(1, 16, 32)
    K = torch.randn(1, 16, 32)
    V = torch.randn(1, 16, 32)
    O = FlashAttentionFunction.apply(Q, K, V, False)

if __name__ == "__main__":
    main()