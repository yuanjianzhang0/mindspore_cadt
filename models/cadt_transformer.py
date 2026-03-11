import warnings
warnings.filterwarnings('ignore')

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np

class CrossAttention(nn.Cell):
    """
    Core Cross-Modal Attention Mechanism.
    
    This module computes attention weights by projecting a query from the anchor 
    modality (e.g., visual) against key-value pairs from an auxiliary modality 
    (e.g., sonar or physiological signals). This enables the network to focus on 
    temporally correlated features across heterogeneous sensory inputs.
    """
    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Query, Key, and Value
        self.W_q = nn.Dense(d_model, d_model)
        self.W_k = nn.Dense(d_model, d_model)
        self.W_v = nn.Dense(d_model, d_model)
        self.fc_out = nn.Dense(d_model, d_model)
        
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = ops.BatchMatMul()
        self.transpose = ops.Transpose()
        self.sqrt_d_k = ops.scalar_to_tensor(np.sqrt(self.d_k), ms.float32)

    def construct(self, query, key, value):
        batch_size = query.shape[0]
        
        # 1. Linear projections and reshaping for multi-head attention formulation
        # Shape: [Batch, Heads, Seq_len, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k)
        
        Q = self.transpose(Q, (0, 2, 1, 3))
        K = self.transpose(K, (0, 2, 1, 3))
        V = self.transpose(V, (0, 2, 1, 3))
        
        # 2. Scaled dot-product attention computation
        K_T = self.transpose(K, (0, 1, 3, 2))
        scores = self.batch_matmul(Q, K_T) / self.sqrt_d_k
        attention_weights = self.softmax(scores)
        
        # 3. Context vector aggregation
        context = self.batch_matmul(attention_weights, V)
        
        # 4. Concatenation of attention heads and final linear projection
        context = self.transpose(context, (0, 2, 1, 3))
        context = context.view(batch_size, -1, self.d_model if hasattr(self, 'd_model') else self.n_heads * self.d_k)
        
        return self.fc_out(context)


class CADTFusionBlock(nn.Cell):
    """
    Cross-modal Aquatic Distress Transformer (CADT) Fusion Block.
    
    Facilitates deep interaction between visual, sonar, and physiological modalities 
    via asymmetric cross-attention. It establishes a mutual supervision framework 
    to mitigate modality collapse in challenging underwater environments.
    """
    def __init__(self, d_model, n_heads):
        super(CADTFusionBlock, self).__init__()
        # Asymmetric cross-attention: Vision querying Sonar and Physiological data
        self.vis_sonar_attn = CrossAttention(d_model, n_heads)
        self.vis_physio_attn = CrossAttention(d_model, n_heads)
        
        self.layer_norm1 = nn.LayerNorm([d_model])
        self.layer_norm2 = nn.LayerNorm([d_model])
        self.layer_norm3 = nn.LayerNorm([d_model])
        
        # Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.SequentialCell(
            nn.Dense(d_model, d_model * 4),
            nn.GELU(),
            nn.Dense(d_model * 4, d_model)
        )

    def construct(self, vis_feat, sonar_feat, physio_feat):
        # 1. Cross-modal mutual supervision: utilizing visual features as the 
        # primary anchor to query sonar representations.
        fused_sonar = self.vis_sonar_attn(vis_feat, sonar_feat, sonar_feat)
        vis_feat = self.layer_norm1(vis_feat + fused_sonar) 
        
        # 2. Secondary querying for physiological alignment
        fused_physio = self.vis_physio_attn(vis_feat, physio_feat, physio_feat)
        vis_feat = self.layer_norm2(vis_feat + fused_physio)
        
        # 3. Feature refinement via FFN and residual connection
        ffn_out = self.ffn(vis_feat)
        out = self.layer_norm3(vis_feat + ffn_out)
        
        return out


class CADT(nn.Cell):
    """
    Cross-modal Aquatic Distress Transformer (CADT) Architecture.
    
    The primary network designed for intelligent drowning behavior recognition.
    It integrates heterogeneous sensory inputs into a unified latent space and 
    employs stacked fusion blocks for robust decision-making.
    """
    def __init__(self, d_model=256, n_heads=8, num_layers=3, num_classes=2):
        super(CADT, self).__init__()
        self.d_model = d_model
        
        # Modality-specific projection layers for spatial dimension unification
        self.vis_proj = nn.Dense(512, d_model)    
        self.sonar_proj = nn.Dense(128, d_model)  
        self.physio_proj = nn.Dense(64, d_model)  
        
        # Stacked multi-modal fusion layers
        self.layers = nn.CellList([CADTFusionBlock(d_model, n_heads) for _ in range(num_layers)])
        
        # Classification head for distress probability estimation
        self.classifier = nn.SequentialCell(
            nn.Dense(d_model, 128),
            nn.ReLU(),
            nn.Dense(128, num_classes)
        )

    def construct(self, vis_x, sonar_x, physio_x):
        # 1. Project heterogeneous inputs into a shared latent semantic space
        vis_feat = self.vis_proj(vis_x)
        sonar_feat = self.sonar_proj(sonar_x)
        physio_feat = self.physio_proj(physio_x)
        
        # 2. Deep cross-modal feature fusion
        for layer in self.layers:
            vis_feat = layer(vis_feat, sonar_feat, physio_feat)
            
        # 3. Temporal aggregation via Global Average Pooling (GAP)
        pooled_feat = vis_feat.mean(axis=1)
        
        # 4. Distress state classification
        logits = self.classifier(pooled_feat) 
        
        return logits


# ==========================================
# Validation & Profiling Module
# ==========================================
if __name__ == '__main__':
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("CPU")
    
    print("[INFO] Initializing CADT Architecture Validation...")
    
    # Hyperparameters simulation
    batch_size = 4
    seq_len = 16 
    num_classes = 2 
    
    model = CADT(d_model=256, n_heads=8, num_layers=2, num_classes=num_classes)
    
    # Synthetic tensors representing heterogeneous sensory streams
    dummy_vis = ms.Tensor(np.random.randn(batch_size, seq_len, 512), ms.float32)
    dummy_sonar = ms.Tensor(np.random.randn(batch_size, seq_len, 128), ms.float32)
    dummy_physio = ms.Tensor(np.random.randn(batch_size, seq_len, 64), ms.float32)
    
    # Forward pass
    output_logits = model(dummy_vis, dummy_sonar, dummy_physio)
    probabilities = ops.softmax(output_logits, axis=-1)
    
    print(f"[INFO] Forward Pass Completed.")
    print(f"[INFO] Input Shapes -> Vis: {dummy_vis.shape}, Sonar: {dummy_sonar.shape}, Physio: {dummy_physio.shape}")
    print(f"[INFO] Output Shape -> {output_logits.shape}")
    print(f"[INFO] Sample Prediction (Normal, Distress) -> {probabilities[0].asnumpy()}")