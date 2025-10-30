from torch.nn.modules import padding
import triton
import triton.language as tl

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    
    # Track the current key tile start position for causal masking
    key_tile_start = 0
    
    for _ in tl.range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0,))
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        s = tl.dot(q, tl.trans(k)) / scale
        # Create indices for the current query and key tiles
        query_indices = tl.arange(0, Q_TILE_SIZE) + query_tile_index * Q_TILE_SIZE
        key_indices = tl.arange(0, K_TILE_SIZE) + key_tile_start
        mask = (query_indices[:,None] < N_QUERIES) & (key_indices[None,:] < N_KEYS)
        s += tl.where(mask, 0, float("-inf"))
        
        # Apply causal masking if enabled
        if is_causal:
            # Create causal mask: mask out positions where key_idx > query_idx
            causal_mask = query_indices[:, None] >= key_indices[None, :]
            s = tl.where(causal_mask, s, float("-inf"))
        
        new_m = tl.maximum(m, tl.max(s, axis=-1))
        p = tl.exp(s - new_m[:, None])
        m_exp_compensate = tl.exp(m - new_m)
        l = m_exp_compensate * l + tl.sum(p, axis=-1)
        o = m_exp_compensate[:, None] * o + tl.dot(p.to(v.dtype), v)
        m = new_m
        
        # Advance key tile position
        key_tile_start += K_TILE_SIZE
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(O_block_ptr, (1/l[:, None] * o).to(O_block_ptr.type.element_ty), boundary_check=(0,))
    tl.store(L_block_ptr, m + tl.log(l), boundary_check=(0,))
        
    