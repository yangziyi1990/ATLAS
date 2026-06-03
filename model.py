"""
SAGE v3.1 Model Architecture (ESM Regression MLM)
===================================================
核心升级 (相对于 v3):
1. MLM 目标: 从 hash classification (65542类) 改为 ESM Feature Regression
   - 删除 mlm_head (Linear 480→65542)
   - 新增 esm_regression_head (3层MLP: 480→960→480→480)
   - Loss: Cosine Similarity + Smooth L1 (回归还原被mask位置的ESM蛋白质特征)
2. Strain Embedding: 从 contrastive_head projected 192d 改为 attention_pooling 480d
   - 下游直接使用 480d 表征, contrastive_head 仅用于训练时对比loss
3. 保留的 v3 特性:
   - ProgressiveGatedFusion: 渐进式门控融合
   - Soft Hierarchical Attention: per-layer 距离偏置门控 + soft cross-contig bias
   - DistanceAwareAttentionV2: Flash Attention 兼容
   - SwiGLU FFN
   - AttentionPooling (复用于 strain embedding)
   - <END> token 支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ==============================================================================
# 1. Gaussian RBF Distance Encoding
# ==============================================================================

class GaussianRBF(nn.Module):
    """Gaussian RBF for continuous intergenic distance encoding."""
    def __init__(self, d_model, num_centers=64, v_min=-50.0, v_max=5000.0):
        super().__init__()
        centers = torch.linspace(v_min, v_max, num_centers)
        self.register_buffer("centers", centers)
        width = (v_max - v_min) / num_centers
        self.register_buffer("width", torch.tensor(width))
        self.linear = nn.Linear(num_centers, d_model)

    def forward(self, distances):
        distances = distances.unsqueeze(-1)
        rbf = torch.exp(-((distances - self.centers) ** 2) / (self.width ** 2))
        return self.linear(rbf)

# ==============================================================================
# 2. Progressive Gated Fusion (替代 AttentionFeatureFusion)
# ==============================================================================

class ProgressiveGatedFusion(nn.Module):
    """
    渐进式门控融合: 支持三种模式融合多路特征.
    
    模式:
    1. 全局权重 (默认): 仅 num_features 个可学习标量, 适合小数据
    2. Token-level gating: 每个 token 独立计算融合权重 (MLP), 适合大数据
    3. Segment-level gating: 按 contig/replicon 分段, 段内共享融合权重 (低秩 MLP)
       - 生物学动机: 同一 contig 内基因受相同调控环境, 融合权重应一致
       - 相比 token-level: 参数更少 (低秩), 段内共享避免过拟合
       - 相比全局: 允许不同基因组区域有不同的特征优先级
    
    消融实验启示:
    - 简单相加 (wo_attention_fusion) PPL=38.03
    - Cross-attention fusion PPL=46.34
    - Token-level gating: PPL=23.44 但 ARI 下降 (过拟合)
    → 需要 segment-level 的中间粒度方案.
    
    生物学动机:
    - gene_emb (ESM) 是核心语义 → 权重最大
    - strand/replicon/contig 是结构先验 → 较小权重
    - mutation_emb 对耐药性关键 → 可学习提升
    - distance_emb 对操纵子建模关键 → 可学习提升
    """
    def __init__(self, d_model, num_features=7, dropout=0.1,
                 use_token_level_gating=False, use_segment_level_gating=False):
        super().__init__()
        assert not (use_token_level_gating and use_segment_level_gating), \
            "use_token_level_gating and use_segment_level_gating are mutually exclusive."
        
        self.num_features = num_features
        self.d_model = d_model
        self.use_token_level_gating = use_token_level_gating
        self.use_segment_level_gating = use_segment_level_gating
        
        if use_token_level_gating:
            # Token-level: 每个 token 独立计算融合权重
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model * num_features, d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_features),
            )
        elif use_segment_level_gating:
            # Segment-level: 低秩 MLP, 段内 mean-pool 后计算一组权重
            # 比 token-level 参数量减少 ~4x (bottleneck=64 vs d_model=256)
            bottleneck = max(32, d_model // 4)
            self.segment_gate_proj = nn.Sequential(
                nn.Linear(d_model * num_features, bottleneck),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, num_features),
            )
            # 全局权重作为 fallback (当 segment_ids 不可用时)
            init_weights = torch.zeros(num_features)
            init_weights[0] = 2.0
            self.feature_weights_fallback = nn.Parameter(init_weights)
        else:
            # 全局可学习权重: 仅 num_features 个标量参数
            init_weights = torch.zeros(num_features)
            init_weights[0] = 2.0  # gene_emb 初始权重最大
            self.feature_weights = nn.Parameter(init_weights)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features_list, segment_ids=None):
        """
        Args:
            features_list: list of [B, L, D], length = num_features
            segment_ids: [B, L] contig IDs for segment-level gating (可选)
        Returns:
            fused: [B, L, D]
        """
        stacked = torch.stack(features_list, dim=2)  # [B, L, N, D]
        
        if self.use_token_level_gating:
            B, L, N, D = stacked.shape
            concat = stacked.view(B, L, N * D)
            gates = self.gate_proj(concat)  # [B, L, N]
            gates = F.softmax(gates, dim=-1).unsqueeze(-1)  # [B, L, N, 1]
            fused = (stacked * gates).sum(dim=2)
        elif self.use_segment_level_gating:
            if segment_ids is not None:
                B, L, N, D = stacked.shape
                fused = torch.zeros(B, L, D, device=stacked.device, dtype=stacked.dtype)
                
                for b in range(B):
                    unique_segs = segment_ids[b].unique()
                    for seg_id in unique_segs:
                        if seg_id.item() == 0:
                            continue  # 跳过 PAD segment (contig_id=0 为 padding)
                        seg_mask = (segment_ids[b] == seg_id)  # [L]
                        if seg_mask.sum() == 0:
                            continue
                        # 段内 mean-pool 所有特征, 然后计算该段的融合权重
                        seg_features = stacked[b, seg_mask]  # [S, N, D]
                        seg_mean = seg_features.mean(dim=0)  # [N, D]
                        seg_concat = seg_mean.view(1, N * D)  # [1, N*D]
                        seg_gates = self.segment_gate_proj(seg_concat)  # [1, N]
                        seg_gates = F.softmax(seg_gates, dim=-1).unsqueeze(-1)  # [1, N, 1]
                        # 段内所有 token 共享同一组权重
                        fused[b, seg_mask] = (seg_features * seg_gates).sum(dim=1)  # [S, D]
                    # PAD 位置 fused 保持零向量 (embedding 也是零), 不影响后续计算
            else:
                # Fallback: segment_ids 不可用, 使用全局权重
                weights = F.softmax(self.feature_weights_fallback, dim=0)
                weights = weights.view(1, 1, -1, 1)
                fused = (stacked * weights).sum(dim=2)
        else:
            weights = F.softmax(self.feature_weights, dim=0)  # [N]
            weights = weights.view(1, 1, -1, 1)  # [1, 1, N, 1]
            fused = (stacked * weights).sum(dim=2)
        
        return self.layer_norm(self.dropout(fused))
    
    def get_fusion_weights(self):
        """返回当前融合权重 (用于可解释性分析)"""
        if self.use_token_level_gating or self.use_segment_level_gating:
            return None
        return F.softmax(self.feature_weights, dim=0).detach().cpu()

# ==============================================================================
# 3. Rotary Positional Embedding (RoPE with 1.5x extrapolation)
# ==============================================================================

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=4096, extrapolation_factor=1.5):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        # 预计算 1.5x 外推的缓存 (借鉴 Bacformer)
        self.max_seq_len = int(max_seq_len * extrapolation_factor)
        self._build_cache(self.max_seq_len)
        
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)
        
    def forward(self, q, k, position_ids=None):
        seq_len = q.size(2)
        if position_ids is None:
            if seq_len > self.max_seq_len:
                self._build_cache(seq_len)
                self.max_seq_len = seq_len
            cos = self.cos_cached[:, :, :seq_len, ...]
            sin = self.sin_cached[:, :, :seq_len, ...]
        else:
            max_pos = position_ids.max().item() + 1
            if max_pos > self.max_seq_len:
                self._build_cache(max_pos)
                self.max_seq_len = max_pos
            cos = self.cos_cached[0, 0, position_ids].unsqueeze(1)
            sin = self.sin_cached[0, 0, position_ids].unsqueeze(1)
            
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
        
    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

# ==============================================================================
# 4. Distance-Aware Attention V2 (Flash Attention 兼容 + 稀疏距离偏置)
# ==============================================================================

class DistanceAwareAttentionV2(nn.Module):
    """
    Distance-Aware Attention v2:
    
    优化策略:
    1. 无 distance bias 时 → 使用 PyTorch SDPA (自动选择 Flash Attention)
    2. 有 distance bias 时 → 手动计算, 但距离偏置限制在 ±distance_window 范围
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1, distance_window=128, max_seq_len=4096):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.distance_window = distance_window
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # 每个 head 学习独立的距离衰减斜率
        self.distance_slope = nn.Parameter(torch.zeros(num_heads))
        nn.init.uniform_(self.distance_slope, -0.1, -0.01)
        
    def forward(self, x, mask=None, position_ids=None, distance_bias=None, cross_contig_bias=None):
        B, L, E = x.size()
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope(q, k, position_ids)
        
        if distance_bias is None and cross_contig_bias is None:
            # 无距离偏置且无跨 contig 偏置 → 使用 SDPA (支持 Flash Attention)
            attn_mask = None
            if mask is not None:
                # mask: [B, 1, L, L] (1=attend, 0=block)
                attn_mask = mask.expand(B, self.num_heads, L, L).to(dtype=torch.bool)
                # SDPA 使用 additive mask: 0 = attend, -inf = block
                attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(dtype=q.dtype)
            
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # 有偏置 → 手动计算 attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
            
            if distance_bias is not None:
                # 距离偏置: log1p 压缩 + 可学习斜率
                slopes = self.distance_slope.view(1, self.num_heads, 1, 1)
                dist_bias = torch.log1p(distance_bias.abs().unsqueeze(1)) * slopes  # [B, H, L, L]
                
                # 稀疏化: 超出 ±window 范围的距离偏置设为 0 (仅靠 RoPE 提供位置信息)
                if self.distance_window < L:
                    # 构建窗口 mask: 只在 ±window 范围内保留 distance bias
                    row_idx = torch.arange(L, device=x.device).unsqueeze(1)
                    col_idx = torch.arange(L, device=x.device).unsqueeze(0)
                    outside_window = (row_idx - col_idx).abs() > self.distance_window
                    dist_bias = dist_bias.masked_fill(outside_window.unsqueeze(0).unsqueeze(0), 0.0)
                
                attn_scores = attn_scores + dist_bias
            
            if cross_contig_bias is not None:
                # 跨 contig soft 偏置: [B, 1, L, L] 广播到所有 heads
                attn_scores = attn_scores + cross_contig_bias
            
            if mask is not None:
                attn_mask = (mask == 0)
                attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
        
        out = out.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(out)

# ==============================================================================
# 5. SwiGLU FFN (保留, 消融实验验证有效)
# ==============================================================================

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# ==============================================================================
# 6. Strain-Aware Transformer Layer V3
# ==============================================================================

class StrainAwareTransformerLayerV3(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1,
                 use_swiglu=True, distance_window=128, max_seq_len=4096):
        super().__init__()
        self.self_attn = DistanceAwareAttentionV2(
            d_model, num_heads, dropout=dropout,
            distance_window=distance_window, max_seq_len=max_seq_len
        )
        
        if use_swiglu:
            swiglu_dim = int(dim_feedforward * 2 / 3)
            self.ffn = SwiGLUFFN(d_model, swiglu_dim, dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, position_ids=None, distance_bias=None, cross_contig_bias=None):
        # Pre-LN Transformer Block
        # Note: all args must be positional-compatible for torch.utils.checkpoint
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, mask=src_mask, position_ids=position_ids,
                              distance_bias=distance_bias, cross_contig_bias=cross_contig_bias)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.ffn(src2)
        src = src + self.dropout2(src2)
        return src

# ==============================================================================
# 7. Attention Pooling (保留, 用于可解释性)
# ==============================================================================

class AttentionPooling(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, pad_mask=None):
        B = x.size(0)
        query = self.query.expand(B, -1, -1)
        pooled, attn_weights = self.cross_attn(query, x, x, key_padding_mask=pad_mask)
        pooled = self.norm(pooled.squeeze(1))
        return pooled, attn_weights

# ==============================================================================
# 8. GenomicLanguageModelV3 (SAGE v3 主模型)
# ==============================================================================

class GenomicLanguageModelV3(nn.Module):
    """
    SAGE v3: Strain-Aware Genomic Language Model
    
    核心升级:
    1. ProgressiveGatedFusion: 替代 AttentionFeatureFusion, 小数据更稳定
    2. Soft Hierarchical Attention: per-layer 可学习距离偏置门控 + soft cross-contig bias
       (替代原硬切换三阶段, 解决表征坍缩问题)
    3. DistanceAwareAttentionV2: Flash Attention 兼容, 稀疏距离偏置, cross-contig bias
    4. 新增 <END> token (vocab[5]), 区分序列结束和 contig 分隔
    5. RoPE 1.5x 外推裕度
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, feature_dim=None,
                 esm_features=None, use_swiglu=True, use_hierarchical_attention=True,
                 use_gated_fusion=True, use_distance_bias=True, exclude_cog=False,
                 max_seq_len=2048, distance_window=128, 
                 use_token_level_gating=False, use_segment_level_gating=False,
                 gradient_checkpointing=False, contrastive_proj_dim=128):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_hierarchical_attention = use_hierarchical_attention
        self.use_gated_fusion = use_gated_fusion
        self.use_distance_bias = use_distance_bias
        self.use_segment_level_gating = use_segment_level_gating
        self.exclude_cog = exclude_cog
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        
        # ---- Embedding Layers ----
        self.use_continuous_features = (feature_dim is not None)
        
        if self.use_continuous_features:
            self.gene_embedding = nn.Linear(feature_dim, d_model)
            if esm_features is not None:
                self.register_buffer("esm_features", esm_features)
        else:
            self.gene_embedding = nn.Embedding(vocab_size, d_model)
        
        self.strand_embedding = nn.Embedding(3, d_model)       # 0=PAD, 1=+, 2=-
        self.replicon_embedding = nn.Embedding(4, d_model)      # 0=PAD, 1=Chr, 2=Plasmid, 3=Unk
        self.cog_embedding = nn.Embedding(29, d_model)          # 27 COG + PAD + UNK
        self.contig_embedding = nn.Embedding(1005, d_model)     # 0=PAD, 1~999 contig ids
        self.mutation_embedding = nn.Embedding(4, d_model)      # 0=PAD, 1=WT, 2=Single, 3=Multi
        self.distance_embedding = GaussianRBF(d_model)
        
        # ---- Progressive Gated Fusion ----
        num_fusion_features = 6 if self.exclude_cog else 7
        if self.use_gated_fusion:
            self.feature_fusion = ProgressiveGatedFusion(
                d_model, num_features=num_fusion_features,
                dropout=dropout, use_token_level_gating=use_token_level_gating,
                use_segment_level_gating=use_segment_level_gating
            )
        else:
            self.simple_fusion_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # ---- Soft Hierarchical Encoder ----
        # 确保 num_layers >= 3 以支持 per-layer 分级初始化
        if use_hierarchical_attention:
            assert num_layers >= 3, (
                f"num_layers={num_layers} must be >= 3 for three-stage hierarchical attention. "
                f"Use --no_hierarchical_attention for fewer layers."
            )
        # 三阶段分界点: 各 1/3
        self.stage1_end = num_layers // 3       # Local
        self.stage2_end = 2 * num_layers // 3   # Bridge
        # stage3: [stage2_end, num_layers)      # Global
        
        # Soft Hierarchical: per-layer 可学习参数
        if use_hierarchical_attention:
            # 距离偏置门控: sigmoid(gate) ∈ [0,1], 初始全 1 (底层保持满距离偏置)
            # 底层初始化为正值(≈sigmoid→1), 高层初始化为 0(≈sigmoid→0.5)
            dist_gate_init = torch.zeros(num_layers)
            dist_gate_init[:self.stage2_end] = 2.0   # 底层 sigmoid(2)≈0.88
            dist_gate_init[self.stage2_end:] = 0.0    # 高层 sigmoid(0)=0.5, 可学习
            self.dist_bias_gate = nn.Parameter(dist_gate_init)
            
            # 跨 contig 注意力软化系数: sigmoid(scale) ∈ [0,1]
            # 底层初始化为负值(接近0, 强 blocking), 高层初始化为正值(接近1, 弱 blocking)
            cross_scale_init = torch.zeros(num_layers)
            cross_scale_init[:self.stage1_end] = -3.0   # sigmoid(-3)≈0.05, 强 contig blocking
            cross_scale_init[self.stage1_end:self.stage2_end] = 0.0  # sigmoid(0)=0.5, 中等
            cross_scale_init[self.stage2_end:] = 2.0    # sigmoid(2)≈0.88, 弱 blocking
            self.cross_contig_scale = nn.Parameter(cross_scale_init)
        
        self.layers = nn.ModuleList([
            StrainAwareTransformerLayerV3(
                d_model, num_heads, dim_feedforward, dropout,
                use_swiglu=use_swiglu, distance_window=distance_window,
                max_seq_len=max_seq_len
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # ---- Attention Pooling ----
        self.attention_pooling = AttentionPooling(
            d_model, num_heads=min(4, num_heads), dropout=dropout
        )
        
        # ---- Task Heads ----
        # ESM Regression MLM Head: 从 hidden_state 还原被 mask 位置的 ESM 蛋白质特征
        # 替代原 mlm_head = Linear(d_model, vocab_size) 的 hash classification
        self.esm_regression_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, feature_dim if feature_dim else d_model),
        )
        self.cog_head = nn.Linear(d_model, 29)
        self.strand_head = nn.Linear(d_model, 3)
        
        # ---- Contrastive Projection Head (for genus-level SupCon, training only) ----
        self.contrastive_proj_dim = contrastive_proj_dim
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, contrastive_proj_dim),
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _compute_pairwise_distance(self, distance_ids, contig_ids):
        """构建 pairwise 距离矩阵 [B, L, L]"""
        B, L = distance_ids.shape
        abs_positions = torch.cumsum(distance_ids.abs(), dim=1)
        pair_dist = (abs_positions.unsqueeze(2) - abs_positions.unsqueeze(1)).abs()
        same_contig = (contig_ids.unsqueeze(1) == contig_ids.unsqueeze(2))
        pair_dist = pair_dist.masked_fill(~same_contig, 1e6)
        return pair_dist
    
    def _build_bridge_mask(self, local_mask, global_mask, replicon_ids):
        """
        [DEPRECATED] 旧三阶段方案的 Bridge Mask, 已被 _build_soft_cross_contig_bias 替代.
        保留此方法仅用于向后兼容参考.
        
        Bridge Mask: contig 内全连接 + 同 replicon 跨 contig 弱连接
        
        逻辑:
        - 同 contig: 全连接 (weight=1)
        - 同 replicon 不同 contig: 弱连接 (允许 attend)
        - 不同 replicon: 隔断
        """
        # local_mask: [B, 1, L, L]
        # 构建同 replicon mask
        same_replicon = (replicon_ids.unsqueeze(1) == replicon_ids.unsqueeze(2))
        same_replicon = same_replicon.unsqueeze(1)  # [B, 1, L, L]
        
        # Bridge = local (同 contig) OR (同 replicon) — 但仍需要 global 的 pad mask
        bridge_mask = (local_mask.bool() | same_replicon) & global_mask.bool()
        
        return bridge_mask.long()
    
    def _build_soft_cross_contig_bias(self, contig_ids, replicon_ids):
        """
        构建 soft 跨 contig 注意力偏置 [B, 1, L, L]
        
        替代硬 bridge/global mask 的三值 soft penalty:
        - 同 contig: 0 (无惩罚)
        - 跨 contig 同 replicon: -2.0 (允许 attend 但有中等惩罚)
        - 跨 replicon: -4.0 (允许 attend 但有较大惩罚)
        
        配合 per-layer cross_contig_scale 使用:
        effective_bias = cross_bias * (1 - scale_i)
        底层 scale_i≈0 → 满惩罚(近似 contig blocking)
        高层 scale_i≈1 → 弱惩罚(近似 global attention)
        """
        same_contig = (contig_ids.unsqueeze(1) == contig_ids.unsqueeze(2))    # [B, L, L]
        same_replicon = (replicon_ids.unsqueeze(1) == replicon_ids.unsqueeze(2))  # [B, L, L]
        
        cross_contig_same_rep = (~same_contig) & same_replicon
        cross_replicon = (~same_contig) & (~same_replicon)
        
        bias = torch.zeros_like(same_contig, dtype=torch.float)
        bias[cross_contig_same_rep] = -2.0
        bias[cross_replicon] = -4.0
        
        return bias.unsqueeze(1)  # [B, 1, L, L]
    
    def forward(self, gene_seqs=None, gene_features=None, strand_ids=None,
                replicon_ids=None, cog_ids=None, contig_ids=None, mutation_ids=None,
                distance_ids=None, position_ids=None, mask=None,
                global_mask=None, extract_features=False, return_pooled=False):
        
        # ---- Step 1: Compute all feature embeddings ----
        if self.use_continuous_features:
            if gene_features is None and hasattr(self, 'esm_features') and gene_seqs is not None:
                gene_features = self.esm_features[gene_seqs]
            if gene_features is not None:
                gene_emb = self.gene_embedding(gene_features)
            else:
                raise ValueError("use_continuous_features=True but no gene_features provided.")
        else:
            gene_emb = self.gene_embedding(gene_seqs)
        
        strand_emb = self.strand_embedding(strand_ids) if strand_ids is not None else torch.zeros_like(gene_emb)
        replicon_emb = self.replicon_embedding(replicon_ids) if replicon_ids is not None else torch.zeros_like(gene_emb)
        cog_emb = self.cog_embedding(cog_ids) if cog_ids is not None else torch.zeros_like(gene_emb)
        contig_emb = self.contig_embedding(contig_ids) if contig_ids is not None else torch.zeros_like(gene_emb)
        mutation_emb = self.mutation_embedding(mutation_ids) if mutation_ids is not None else torch.zeros_like(gene_emb)
        distance_emb = self.distance_embedding(distance_ids) if distance_ids is not None else torch.zeros_like(gene_emb)
        
        # ---- Step 2: Feature Fusion ----
        if self.exclude_cog:
            features_list = [gene_emb, strand_emb, replicon_emb, contig_emb, mutation_emb, distance_emb]
        else:
            features_list = [gene_emb, strand_emb, replicon_emb, cog_emb, contig_emb, mutation_emb, distance_emb]
        
        if self.use_gated_fusion:
            x = self.feature_fusion(features_list, segment_ids=contig_ids)
        else:
            x = features_list[0]
            for feat in features_list[1:]:
                x = x + feat
            x = self.simple_fusion_norm(x)
        x = self.dropout(x)
        
        # ---- Step 3: Compute pairwise distance bias ----
        distance_bias = None
        need_distance = self.use_distance_bias and distance_ids is not None and contig_ids is not None
        
        # ---- Step 4: Construct masks ----
        # Global mask: only block PAD
        if global_mask is None and mask is not None:
            row_valid = mask.any(dim=-1, keepdim=True)
            col_valid = mask.any(dim=-2, keepdim=True)
            global_mask = (row_valid & col_valid).long()
        
        # Soft cross-contig bias (替代硬 bridge/global mask 切换)
        cross_contig_bias = None
        if self.use_hierarchical_attention and contig_ids is not None and replicon_ids is not None:
            cross_contig_bias = self._build_soft_cross_contig_bias(contig_ids, replicon_ids)
        
        # ---- Step 5: Soft Hierarchical Attention ----
        for i, layer in enumerate(self.layers):
            if not self.use_hierarchical_attention:
                # 非分层模式: 与原来行为完全一致
                layer_mask = mask
                use_dist = need_distance
                
                if use_dist and distance_bias is None:
                    distance_bias = self._compute_pairwise_distance(distance_ids, contig_ids)
                layer_dist = distance_bias if use_dist else None
                layer_cross_bias = None
            else:
                # Soft Hierarchical: 所有层共用 local_mask, 通过 soft bias 控制跨 contig 交互
                layer_mask = mask  # 始终以 contig-blocked local mask 为基础
                
                # 距离偏置: 所有层都可用, 但强度由 per-layer gate 控制
                if need_distance and distance_bias is None:
                    distance_bias = self._compute_pairwise_distance(distance_ids, contig_ids)
                
                if need_distance:
                    gate_i = torch.sigmoid(self.dist_bias_gate[i])
                    layer_dist = distance_bias * gate_i
                else:
                    layer_dist = None
                
                # 跨 contig 注意力: per-layer scale 控制 blocking 强度
                if cross_contig_bias is not None:
                    scale_i = torch.sigmoid(self.cross_contig_scale[i])
                    # scale_i≈0 → bias 保持原值(强 blocking); scale_i≈1 → bias→0(弱 blocking)
                    layer_cross_bias = cross_contig_bias * (1.0 - scale_i)
                else:
                    layer_cross_bias = None
            
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    layer, x, layer_mask, position_ids, layer_dist, layer_cross_bias,
                    use_reentrant=False
                )
            else:
                x = layer(x, src_mask=layer_mask, position_ids=position_ids,
                          distance_bias=layer_dist, cross_contig_bias=layer_cross_bias)
        
        x = self.norm(x)
        
        # ---- Step 6: Output ----
        if extract_features:
            if return_pooled:
                pad_mask = None
                if gene_seqs is not None:
                    pad_mask = (gene_seqs == 0)
                pooled, attn_weights = self.attention_pooling(x, pad_mask=pad_mask)
                return x, pooled, attn_weights
            return x
        
        outputs = {
            "hidden_states": x,
            "cog_logits": self.cog_head(x),
            "strand_logits": self.strand_head(x)
        }
        
        # ---- Strain Embedding: Attention Pooling → 480d L2 normalized ----
        # 下游直接使用此 480d 表征 (与 gene_embedding 同维度空间)
        if gene_seqs is not None:
            pad_mask = (gene_seqs == 0)  # PAD positions
            pooled, _ = self.attention_pooling(x, pad_mask=pad_mask)  # [B, d_model]
            outputs["strain_embedding"] = F.normalize(pooled, dim=-1)  # [B, 480]
            
            # Contrastive projection: 仅用于训练时 SupCon loss 计算
            projected = self.contrastive_head(pooled)  # [B, proj_dim]
            outputs["contrastive_projected"] = F.normalize(projected, dim=-1)  # [B, 192]
        
        return outputs
