"""
SAGE v3 Dataset + Dynamic Padding Collator + Targeted Masking V2
================================================================
核心升级:
1. 动态 Padding: batch 级 pad 到最长序列 (借鉴 Bacformer), 节省 30-50% FLOPs
2. 靶向掩码 v2: 操纵子感知 + 耐药基因簇掩码 + 质粒靶向
3. 新增 <END> token 保护 (不被掩码)
4. Contig-blocked Local Mask + Bridge Mask + Global Mask 三级构建
"""

import os
import torch
from torch.utils.data import Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GenomicSentenceDatasetV3(Dataset):
    """
    v3 数据集: 支持变长序列 (不在 __getitem__ 中 padding).
    Padding 延迟到 collate_fn 中动态执行.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        
        logging.info(f"Loading v3 pre-processed transformer inputs from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cannot find data at {data_path}. Run generate_input.py first.")
        
        data = torch.load(data_path, weights_only=False)
        
        self.sentences = data["sentences"]
        self.strands = data["strands"]
        self.replicons = data["replicons"]
        self.cogs = data["cogs"]
        self.contigs = data.get("contigs", None)
        self.mutations = data.get("mutations", None)
        self.distances = data.get("distances", None)
        self.esm_features = data.get("esm_features", None)
        self.genome_ids = data.get("genome_ids", None)
        
        self.vocab = data["vocabs"]["gene"]
        self.strand_vocab = data["vocabs"]["strand"]
        self.replicon_vocab = data["vocabs"]["replicon"]
        self.cog_vocab = data["vocabs"]["cog"]
        self.mutation_vocab = data["vocabs"].get("mutation", {"<PAD>": 0})
        
        logging.info(f"Loaded {len(self.sentences)} sentences. Vocab Size: {len(self.vocab)}")
        if self.genome_ids is not None:
            logging.info(f"  Genome IDs available: {len(set(self.genome_ids))} unique strains.")
        
        # genus_ids: 由外部 (train.py) 注入, 用于对比学习
        self.genus_ids = None
        
        # 序列长度统计
        if self.sentences:
            seq_lens = [len(s) for s in self.sentences]
            logging.info(f"  Seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, "
                         f"mean={sum(seq_lens)/len(seq_lens):.1f}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        seq = self.sentences[idx]
        strand_seq = self.strands[idx]
        replicon_seq = self.replicons[idx]
        cog_seq = self.cogs[idx]
        contig_seq = self.contigs[idx] if self.contigs is not None else [0] * len(seq)
        mut_seq = self.mutations[idx] if self.mutations is not None else [0] * len(seq)
        dist_seq = self.distances[idx] if self.distances is not None else [0] * len(seq)
        
        # 构建 per-contig 位置编码
        pos_seq = []
        pos = 0
        if len(seq) > 0:
            prev_c = contig_seq[0]
            for c in contig_seq:
                if c == prev_c:
                    pos_seq.append(pos)
                    pos += 1
                else:
                    pos = 0
                    pos_seq.append(pos)
                    pos += 1
                    prev_c = c
        
        # 不在此处 padding! 返回变长 tensor
        sample = {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "strand_ids": torch.tensor(strand_seq, dtype=torch.long),
            "replicon_ids": torch.tensor(replicon_seq, dtype=torch.long),
            "cog_ids": torch.tensor(cog_seq, dtype=torch.long),
            "contig_ids": torch.tensor(contig_seq, dtype=torch.long),
            "mutation_ids": torch.tensor(mut_seq, dtype=torch.long),
            "distance_ids": torch.tensor(dist_seq, dtype=torch.float),
            "position_ids": torch.tensor(pos_seq, dtype=torch.long)
        }
        if self.genus_ids is not None:
            sample["genus_id"] = self.genus_ids[idx]
        return sample


def get_dynamic_mlm_collator(vocab, mask_prob=0.15, span_length=3,
                              use_targeted_masking=True,
                              operon_mask_prob=0.05,
                              max_operon_distance=300,
                              min_operon_genes=3):
    """
    v3 Dynamic MLM Collator:
    1. 动态 padding 到 batch 内最长序列
    2. 靶向掩码 v2: 突变基因 + 耐药功能(COG V) + 质粒基因 + 操纵子整体掩码
    3. 三级 attention mask 构建
    """
    pad_id = vocab["<PAD>"]
    cls_id = vocab["<CLS>"]
    sep_id = vocab["<SEP>"]
    end_id = vocab["<END>"]
    mask_id = vocab["<MASK>"]
    
    def collate_fn(batch):
        # 1. 动态 padding 到 batch 内最长序列
        max_len = max(b['input_ids'].size(0) for b in batch)
        B = len(batch)
        
        seqs = torch.full((B, max_len), pad_id, dtype=torch.long)
        strands = torch.zeros(B, max_len, dtype=torch.long)
        replicons = torch.zeros(B, max_len, dtype=torch.long)
        cogs = torch.zeros(B, max_len, dtype=torch.long)
        contigs = torch.zeros(B, max_len, dtype=torch.long)
        mutations = torch.zeros(B, max_len, dtype=torch.long)
        distances = torch.zeros(B, max_len, dtype=torch.float)
        position_ids = torch.zeros(B, max_len, dtype=torch.long)
        
        for i, b in enumerate(batch):
            L = b['input_ids'].size(0)
            seqs[i, :L] = b['input_ids']
            strands[i, :L] = b['strand_ids']
            replicons[i, :L] = b['replicon_ids']
            cogs[i, :L] = b['cog_ids']
            contigs[i, :L] = b['contig_ids']
            mutations[i, :L] = b['mutation_ids']
            distances[i, :L] = b['distance_ids']
            position_ids[i, :L] = b['position_ids']
        
        labels = seqs.clone()
        labels_strand = strands.clone()
        labels_cog = cogs.clone()
        
        # 2. 靶向掩码 v2
        span_prob = mask_prob / span_length
        probability_matrix = torch.full(labels.shape, span_prob)
        
        # 特殊 token 保护 (PAD, CLS, SEP, END)
        special_tokens_mask = (
            (seqs == pad_id) | (seqs == cls_id) | 
            (seqs == sep_id) | (seqs == end_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        
        if use_targeted_masking:
            # (a) 突变基因靶向: mutation > 1, 3x 概率
            mutated = mutations > 1
            probability_matrix[mutated] = min(span_prob * 3.0, 0.5)
            
            # (b) 耐药功能靶向: COG='V' (Defense mechanisms), 2x 概率
            # V 在 cog_vocab 中的索引 = 2 + ord('V') - ord('A') = 2 + 21 = 23
            defense_cog = (cogs == 23)
            probability_matrix[defense_cog] = min(span_prob * 2.0, 0.5)
            
            # (c) 质粒基因靶向: replicon=plasmid(2), 1.5x 概率
            on_plasmid = (replicons == 2)
            probability_matrix[on_plasmid] = torch.max(
                probability_matrix[on_plasmid],
                torch.tensor(min(span_prob * 1.5, 0.5))
            )
        
        # 再次确保特殊 token 不被掩码
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        
        # 3. Span masking (contig-aware)
        span_starts = torch.bernoulli(probability_matrix).bool()
        masked_indices = span_starts.clone()
        current_starts = span_starts.clone()
        
        for _ in range(1, span_length):
            shifted = torch.roll(current_starts, shifts=1, dims=1)
            shifted[:, 0] = False
            same_contig = (contigs == torch.roll(contigs, shifts=1, dims=1))
            same_contig[:, 0] = False
            current_starts = shifted & same_contig
            masked_indices |= current_starts
        
        # 4. 操纵子整体掩码 (新增)
        if use_targeted_masking and operon_mask_prob > 0:
            operon_mask = _detect_and_mask_operons(
                distances, contigs, special_tokens_mask,
                min_genes=min_operon_genes,
                max_distance=max_operon_distance,
                mask_prob=operon_mask_prob
            )
            masked_indices = masked_indices | operon_mask
        
        # 确保特殊 token 不被掩码
        masked_indices.masked_fill_(special_tokens_mask, False)
        
        labels[~masked_indices] = -100
        labels_strand[~masked_indices] = -100
        labels_cog[~masked_indices] = -100
        
        # 5. BERT-style masking: 80% MASK, 10% random, 10% unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        seqs[indices_replaced] = mask_id
        
        # 特征泄露防护
        strands[indices_replaced] = 0
        replicons[indices_replaced] = 0
        cogs[indices_replaced] = 0
        mutations[indices_replaced] = 0
        distances[indices_replaced] = 0
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(vocab), labels.shape, dtype=torch.long)
        seqs[indices_random] = random_words[indices_random]
        
        strands[indices_random] = 0
        replicons[indices_random] = 0
        cogs[indices_random] = 0
        mutations[indices_random] = 0
        distances[indices_random] = 0
        
        # 6. 构建 Attention Masks
        # Local mask: contig-blocked + pad blocking
        same_contig_mask = (contigs.unsqueeze(1) == contigs.unsqueeze(2))
        pad_mask = (seqs != pad_id).unsqueeze(1)
        local_mask = (same_contig_mask & pad_mask).long()
        # 确保非 PAD token 至少能 attend 自己（即使跨 contig mask 全为 0 的边界情况）
        # 副作用: PAD token 也会 attend 自己，但 PAD embedding 为零不影响结果
        local_mask.diagonal(dim1=-2, dim2=-1).fill_(1)
        
        # 7. Genus IDs (用于对比学习)
        genus_ids = torch.tensor(
            [b.get("genus_id", -1) for b in batch], dtype=torch.long
        )
        
        return {
            "input_ids": seqs,
            "strand_ids": strands,
            "replicon_ids": replicons,
            "cog_ids": cogs,
            "contig_ids": contigs,
            "mutation_ids": mutations,
            "distance_ids": distances,
            "position_ids": position_ids,
            "attention_mask": local_mask.unsqueeze(1),  # [B, 1, L, L]
            "labels": labels,
            "labels_strand": labels_strand,
            "labels_cog": labels_cog,
            "genus_ids": genus_ids
        }
    
    return collate_fn


def _detect_and_mask_operons(distances, contigs, special_mask,
                              min_genes=3, max_distance=300, mask_prob=0.05):
    """
    检测密集基因区 (操纵子) 并以一定概率整体掩码.
    
    操纵子检测: 连续 >=min_genes 个基因, 距离 < max_distance bp, 同一 contig
    """
    B, L = distances.shape
    operon_mask = torch.zeros(B, L, dtype=torch.bool)
    
    for b in range(B):
        i = 0
        while i < L:
            # 跳过特殊 token
            if special_mask[b, i]:
                i += 1
                continue
            
            # 开始检测一个潜在操纵子
            j = i + 1
            while j < L and not special_mask[b, j]:
                if contigs[b, j] != contigs[b, i]:
                    break
                dist_val = distances[b, j].abs().item()
                # BUG4 fix: distance=0 表示紧邻基因或 contig 首个基因
                # 仅在超出 max_distance 时中断; distance=0 (紧邻) 是操纵子典型模式
                if dist_val > max_distance:
                    break
                j += 1
            
            operon_len = j - i
            if operon_len >= min_genes:
                if torch.rand(1).item() < mask_prob:
                    operon_mask[b, i:j] = True
            
            i = j
    
    return operon_mask
