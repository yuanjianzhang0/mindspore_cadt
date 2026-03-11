import warnings
warnings.filterwarnings('ignore')

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class CrossAttention(nn.Cell):
    """
    Core Multi-Head Cross-Modal Attention Mechanism.
    
    Computes attention weights by projecting a query from the anchor modality (e.g., Vision) 
    against key-value pairs from an auxiliary modality (e.g., Sonar/Physiological).
    This establishes temporal correlation across heterogeneous sensory streams.
    """
    def __init__(self, d_model: int, n_heads: int):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Query, Key, and Value formulation
        self.W_q = nn.Dense(d_model, d_model)
        self.W_k = nn.Dense(d_model, d_model)
        self.W_v = nn.Dense(d_model, d_model)
        
        # Output projection back to the latent model dimension
        self.fc_out = nn.Dense(d_model, d_model)
        
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()
        self.transpose = ops.Transpose()
        self.sqrt_d_k = ops.scalar_to_tensor(np.sqrt(self.d_k), ms.float32)

    def construct(self, query: ms.Tensor, key: ms.Tensor, value: ms.Tensor) -> ms.Tensor:
        batch_size = query.shape[0]
        
        # 1. Linear projection and reshaping into multiple attention heads
        # Shape transition: [Batch, Seq_len, d_model] -> [Batch, Heads, Seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k)
        
        Q = self.transpose(Q, (0, 2, 1, 3))
        K = self.transpose(K, (0, 2, 1, 3))
        V = self.transpose(V, (0, 2, 1, 3))
        
        # 2. Scaled Dot-Product Attention
        K_T = self.transpose(K, (0, 1, 3, 2))
        scores = self.batch_matmul(Q, K_T) / self.sqrt_d_k
        attention_weights = self.softmax(scores)
        
        # 3. Aggregation of Value representations based on attention mapping
        context = self.batch_matmul(attention_weights, V)
        
        # 4. Concatenation of independent heads
        context = self.transpose(context, (0, 2, 1, 3))
        context = context.view(batch_size, -1, self.n_heads * self.d_k)
        
        # Final linear projection
        return self.fc_out(context)