import warnings
warnings.filterwarnings('ignore')

import mindspore.nn as nn
import mindspore as ms

# Explicit relative import for the decoupled attention module
from .attention import CrossAttention

class CADTFusionBlock(nn.Cell):
    """
    Cross-modal Aquatic Distress Transformer (CADT) Fusion Block.
    
    Implements a robust mutual supervision mechanism. It leverages asymmetric 
    cross-attention where visual features actively query both acoustic (Sonar) 
    and biomedical (Physiological) streams, mitigating modality collapse.
    """
    def __init__(self, d_model: int, n_heads: int):
        super(CADTFusionBlock, self).__init__()
        
        # Asymmetric multi-modal attention pathways
        self.vis_sonar_attn = CrossAttention(d_model, n_heads)
        self.vis_physio_attn = CrossAttention(d_model, n_heads)
        
        # Stabilization layers
        self.layer_norm1 = nn.LayerNorm([d_model])
        self.layer_norm2 = nn.LayerNorm([d_model])
        self.layer_norm3 = nn.LayerNorm([d_model])
        
        # Position-wise Feed-Forward Network (FFN) for feature refinement
        self.ffn = nn.SequentialCell(
            nn.Dense(d_model, d_model * 4),
            nn.GELU(),
            nn.Dense(d_model * 4, d_model)
        )

    def construct(self, vis_feat: ms.Tensor, sonar_feat: ms.Tensor, physio_feat: ms.Tensor) -> ms.Tensor:
        # 1. Structural Interaction 1: Visual querying Sonar
        fused_sonar = self.vis_sonar_attn(vis_feat, sonar_feat, sonar_feat)
        vis_feat = self.layer_norm1(vis_feat + fused_sonar) # Residual connection
        
        # 2. Structural Interaction 2: Visual querying Physiological status
        fused_physio = self.vis_physio_attn(vis_feat, physio_feat, physio_feat)
        vis_feat = self.layer_norm2(vis_feat + fused_physio)
        
        # 3. Non-linear mapping via FFN
        ffn_out = self.ffn(vis_feat)
        out = self.layer_norm3(vis_feat + ffn_out)
        
        return out