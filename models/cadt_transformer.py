import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class CrossAttention(nn.Cell):
    """
    非对称跨模态注意力机制 (Asymmetric Cross-Attention)
    """
    def __init__(self, d_model, n_heads):
        super(CrossAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
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
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k)
        
        Q = self.transpose(Q, (0, 2, 1, 3))
        K = self.transpose(K, (0, 2, 1, 3))
        V = self.transpose(V, (0, 2, 1, 3))
        
        K_T = self.transpose(K, (0, 1, 3, 2))
        scores = self.batch_matmul(Q, K_T) / self.sqrt_d_k
        attention_weights = self.softmax(scores)
        
        context = self.batch_matmul(attention_weights, V)
        context = self.transpose(context, (0, 2, 1, 3))
        context = context.view(batch_size, -1, self.n_heads * self.d_k)
        
        return self.fc_out(context)


class CADTFusionBlock(nn.Cell):
    """
    堆叠融合模块 (Stacked CADTFusionBlocks)
    """
    def __init__(self, d_model, n_heads):
        super(CADTFusionBlock, self).__init__()
        self.vis_sonar_attn = CrossAttention(d_model, n_heads)
        self.vis_physio_attn = CrossAttention(d_model, n_heads)
        
        self.layer_norm1 = nn.LayerNorm([d_model])
        self.layer_norm2 = nn.LayerNorm([d_model])
        self.layer_norm3 = nn.LayerNorm([d_model])
        
        self.ffn = nn.SequentialCell(
            nn.Dense(d_model, d_model * 4),
            nn.GELU(),
            nn.Dense(d_model * 4, d_model)
        )

    def construct(self, vis_feat, sonar_feat, physio_feat):
        # 视觉作为 Query，主动融合声呐和生理特征
        fused_sonar = self.vis_sonar_attn(vis_feat, sonar_feat, sonar_feat)
        vis_feat = self.layer_norm1(vis_feat + fused_sonar) 
        
        fused_physio = self.vis_physio_attn(vis_feat, physio_feat, physio_feat)
        vis_feat = self.layer_norm2(vis_feat + fused_physio)
        
        ffn_out = self.ffn(vis_feat)
        out = self.layer_norm3(vis_feat + ffn_out)
        return out


class CADT(nn.Cell):
    """
    核心网络：基于前端 YOLOv12 与 AutoFormer 提取的特征进行联合推理
    """
    def __init__(self, d_model=256, n_heads=8, num_layers=3, num_classes=2):
        super(CADT, self).__init__()
        self.d_model = d_model
        
        self.vis_proj = nn.Dense(512, d_model)    # 对应水上图像特征
        self.sonar_proj = nn.Dense(128, d_model)  # 对应水下声呐特征
        self.physio_proj = nn.Dense(64, d_model)  # 对应 AutoFormer 生理特征
        
        self.layers = nn.CellList([CADTFusionBlock(d_model, n_heads) for _ in range(num_layers)])
        
        self.regression_head = nn.SequentialCell(
            nn.Dense(d_model, 128),
            nn.ReLU(),
            nn.Dense(128, num_classes)
        )

    def construct(self, vis_x, sonar_x, physio_x):
        vis_feat = self.vis_proj(vis_x)
        sonar_feat = self.sonar_proj(sonar_x)
        physio_feat = self.physio_proj(physio_x)
        
        for layer in self.layers:
            vis_feat = layer(vis_feat, sonar_feat, physio_feat)
            
        pooled_feat = vis_feat.mean(axis=1)
        
        logits = self.regression_head(pooled_feat) 
        
        return logits, pooled_feat
