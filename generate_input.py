"""
SAGE v3: Generate Transformer Inputs
=====================================
核心升级:
1. 全基因组建模: max_seq_len 提升到 2048, 大多数细菌基因组不需切分
2. Contig-aware 重叠滑动窗口: 超长基因组在 contig 边界处优先切割
3. 增强 SEP 分隔符: contig 边界处插入 SEP, 显式编码基因组拓扑
4. 保留所有 v2 特征: ESM + strand + replicon + COG + contig + mutation + distance
"""

import os
import glob
import re
import torch
import numpy as np
import logging
import random
from tqdm import tqdm
import h5py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_mutation_annotations(dataset_dir):
    """加载突变注释文件"""
    mut_dir = os.path.join(dataset_dir, "annotations", "mutations")
    gene_to_mut = {}
    if not os.path.exists(mut_dir):
        logging.warning(f"Mutation annotations not found at {mut_dir}, defaulting to WildType (1)")
        return gene_to_mut
    mut_files = glob.glob(os.path.join(mut_dir, "*.tsv"))
    for fpath in mut_files:
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip('\n').split('\t')
                    if len(parts) >= 2:
                        gene_to_mut[parts[0]] = int(parts[1])
        except Exception:
            pass
    return gene_to_mut


def load_eggnog_annotations(dataset_dir):
    """加载 eggNOG 注释文件"""
    eggnog_dir = os.path.join(dataset_dir, "annotations", "eggnog")
    gene_to_cog = {}
    if not os.path.exists(eggnog_dir):
        logging.warning(f"eggNOG annotations not found at {eggnog_dir}")
        return gene_to_cog
    anno_files = glob.glob(os.path.join(eggnog_dir, "*.annotations"))
    for fpath in anno_files:
        try:
            with open(fpath, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip('\n').split('\t')
                    if len(parts) > 6:
                        raw_id = parts[0]
                        gene_id = raw_id.split("@@", 1)[-1] if "@@" in raw_id else raw_id
                        cog = parts[6]
                        cog_primary = cog[0] if len(cog) > 0 else "-"
                        gene_to_cog[gene_id] = cog_primary
        except Exception:
            pass
    return gene_to_cog


class AdaptiveGenomeSentenceBuilder:
    """
    三级序列长度策略:
    
    Level 1 - 全基因组模式 (genes <= max_seq_len - 2):
        直接将整个基因组作为一个 sample
    
    Level 2 - Contig-aware 重叠滑动窗口 (genes > max_seq_len - 2):
        使用 overlap_ratio 重叠的滑动窗口
        窗口边界优先在 contig 分界处切割
    """
    
    def __init__(self, max_seq_len=2048, overlap_ratio=0.25):
        self.max_seq_len = max_seq_len
        # 有效容量 = max_seq_len - 2 (CLS + END)
        self.effective_len = max_seq_len - 2
        self.stride = int(self.effective_len * (1 - overlap_ratio))
    
    def _find_contig_boundaries(self, contig_ids):
        """找到所有 contig 切换点的索引"""
        boundaries = []
        for i in range(1, len(contig_ids)):
            if contig_ids[i] != contig_ids[i - 1]:
                boundaries.append(i)
        return boundaries
    
    def _find_nearest_boundary(self, target, boundaries, max_deviation):
        """找到离目标位置最近的 contig 边界 (在允许偏差范围内)"""
        if not boundaries:
            return None
        candidates = [b for b in boundaries if abs(b - target) <= max_deviation]
        if not candidates:
            return None
        return min(candidates, key=lambda b: abs(b - target))
    
    def build_sentences(self, genome_genes, genome_strands, genome_replicons,
                        genome_cogs, genome_contig_ids, genome_mutations,
                        genome_distances, vocab):
        """
        将一个基因组的完整基因序列切分为 sentences.
        
        Returns:
            list of dict, 每个 dict 包含一个 sentence 的所有特征
        """
        total_genes = len(genome_genes)
        
        if total_genes == 0:
            return []
        
        # Level 1: 全基因组模式 - 不需要切分
        if total_genes <= self.effective_len:
            return [self._wrap_sentence(
                genome_genes, genome_strands, genome_replicons,
                genome_cogs, genome_contig_ids, genome_mutations,
                genome_distances, vocab
            )]
        
        # Level 2: Contig-aware 重叠滑动窗口
        sentences = []
        contig_boundaries = self._find_contig_boundaries(genome_contig_ids)
        max_deviation = self.effective_len // 4  # 允许最多 25% 的偏移去对齐 contig 边界
        
        i = 0
        while i < total_genes:
            end = min(i + self.effective_len, total_genes)
            
            # 如果不是最后一个窗口, 尝试在 contig 边界处切割
            if end < total_genes:
                best_cut = self._find_nearest_boundary(end, contig_boundaries, max_deviation)
                if best_cut is not None and best_cut > i:
                    end = best_cut
            
            sentences.append(self._wrap_sentence(
                genome_genes[i:end], genome_strands[i:end],
                genome_replicons[i:end], genome_cogs[i:end],
                genome_contig_ids[i:end], genome_mutations[i:end],
                genome_distances[i:end], vocab
            ))
            
            # 步进
            if end >= total_genes:
                break
            i += self.stride
            # 确保最后一个窗口不会太短
            if total_genes - i < self.effective_len // 4:
                # 剩余太少, 将最后一段合并到当前窗口
                break
        
        return sentences
    
    def _wrap_sentence(self, genes, strand_seq, replicon_seq, cog_seq,
                       contig_seq, mut_seq, dist_seq, vocab):
        """给 sentence 添加 CLS 和 END token"""
        # CLS at beginning, END at end (替代 v2 的 SEP)
        chunk = [vocab["<CLS>"]] + list(genes) + [vocab["<END>"]]
        s_full = [0] + list(strand_seq) + [0]  # PAD=0
        r_full = [0] + list(replicon_seq) + [0]
        c_full = [0] + list(cog_seq) + [0]
        cid_full = [contig_seq[0] if contig_seq else 0] + list(contig_seq) + [contig_seq[-1] if contig_seq else 0]
        m_full = [0] + list(mut_seq) + [0]
        d_full = [0] + list(dist_seq) + [0]
        
        return {
            "sentence": chunk,
            "strands": s_full,
            "replicons": r_full,
            "cogs": c_full,
            "contigs": cid_full,
            "mutations": m_full,
            "distances": d_full,
        }


def build_transformer_inputs(
    dataset_dir: str = "/opt/ai4g_chriszyyang/buddy2/SAGE/dataset",
    output_dir: str = "/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/features/transformer_v3",
    max_seq_len: int = 2048,
    overlap_ratio: float = 0.25
):
    """
    SAGE v3 数据预处理: 生成 Transformer Genomic Sentences.
    
    核心改变:
    - max_seq_len 提升到 2048 (覆盖 80%+ 完整细菌基因组)
    - Contig-aware 切分 (在 contig 边界处优先切割)
    - 新增 <END> token (区分序列结束和 contig 分隔)
    - 保留 <SEP> 作为 contig 内部分隔符
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 初始化 Vocab (新增 <END> token)
    vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "<CLS>": 3, "<SEP>": 4, "<END>": 5}
    strand_vocab = {"<PAD>": 0, "+": 1, "-": 2}
    replicon_vocab = {"<PAD>": 0, "chromosome": 1, "plasmid": 2, "unknown": 3}
    mutation_vocab = {"<PAD>": 0, "wildtype": 1, "single_mut": 2, "multi_mut": 3}
    
    cog_vocab = {"<PAD>": 0, "<UNK>": 1}
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ-":
        cog_vocab[char] = len(cog_vocab)
    
    def _get_or_add_vocab(gene_id):
        if gene_id not in vocab:
            vocab[gene_id] = len(vocab)
        return vocab[gene_id]
    
    # 2. 加载注释
    logging.info("Loading eggNOG COG annotations...")
    gene_to_cog = load_eggnog_annotations(dataset_dir)
    logging.info("Loading Mutation annotations...")
    gene_to_mut = load_mutation_annotations(dataset_dir)
    
    genomes_dir = os.path.join(dataset_dir, "genomes")
    gff_files = glob.glob(os.path.join(genomes_dir, "**", "*.gff"), recursive=True)
    logging.info(f"Found {len(gff_files)} .gff files to parse physical order.")
    
    # 初始化 sentence builder
    builder = AdaptiveGenomeSentenceBuilder(max_seq_len=max_seq_len, overlap_ratio=overlap_ratio)
    
    all_sentences = []
    all_strands = []
    all_replicons = []
    all_cogs = []
    all_contig_ids = []
    all_mutations = []
    all_distances = []
    genome_ids = []
    total_genes = 0
    query_to_gene = {}
    
    # 统计
    full_genome_count = 0
    windowed_count = 0
    
    # 3. 解析 GFF 提取物理 Synteny
    for gff_file in tqdm(gff_files, desc="Parsing physical gene order"):
        if '.ipynb_checkpoints' in gff_file:
            continue
        
        # 提取 genome_id
        gff_parent = os.path.basename(os.path.dirname(gff_file))
        if gff_parent == os.path.basename(genomes_dir):
            genome_id = os.path.splitext(os.path.basename(gff_file))[0]
        else:
            genome_id = gff_parent
        
        contigs = {}
        seq_to_replicon = {}
        
        try:
            with open(gff_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 9:
                        continue
                    
                    seq_id = parts[0]
                    feature_type = parts[2]
                    attributes = parts[8]
                    
                    if feature_type == 'region':
                        if 'plasmid' in attributes.lower():
                            seq_to_replicon[seq_id] = 'plasmid'
                        elif 'chromosome' in attributes.lower():
                            seq_to_replicon[seq_id] = 'chromosome'
                        continue
                    
                    # BUG5 fix: 只保留 CDS，避免同一基因因 gene+CDS 行被重复解析
                    if feature_type != 'CDS':
                        continue
                    
                    start_pos = int(parts[3])
                    end_pos = int(parts[4])
                    strand_val = parts[6] if parts[6] in ['+', '-'] else '+'
                    
                    locus_tag_match = re.search(r'locus_tag=([^;]+)', attributes)
                    id_match = re.search(r'ID=([^;]+)', attributes)
                    protein_match = re.search(r'protein_id=([^;]+)', attributes)
                    
                    gene_id_local = None
                    if locus_tag_match:
                        gene_id_local = locus_tag_match.group(1)
                    elif id_match:
                        gene_id_local = id_match.group(1)
                    
                    query_id = gene_id_local
                    if protein_match:
                        query_id = protein_match.group(1)
                    
                    if not gene_id_local:
                        continue
                    
                    query_to_gene[query_id] = gene_id_local
                    
                    if seq_id not in contigs:
                        contigs[seq_id] = []
                    
                    cog_cat = gene_to_cog.get(query_id, "-")
                    if cog_cat not in cog_vocab:
                        cog_cat = "-"
                    
                    replicon_type = seq_to_replicon.get(seq_id, "unknown")
                    mut_type = gene_to_mut.get(query_id, 1)
                    
                    contigs[seq_id].append({
                        'id': gene_id_local,
                        'start': start_pos,
                        'end': end_pos,
                        'strand': strand_val,
                        'replicon': replicon_type,
                        'cog': cog_cat,
                        'mutation': mut_type
                    })
                    total_genes += 1
                    
        except Exception as e:
            logging.error(f"Error parsing {gff_file}: {e}")
            continue
        
        # 将所有 contigs 拼接成长序列, 在 contig 边界插入 SEP
        genome_genes = []
        genome_strands = []
        genome_replicons = []
        genome_cogs = []
        genome_contig_ids_local = []
        genome_mutations = []
        genome_distances = []
        
        contig_id_counter = 1
        
        # 数据增强: 随机打乱 contig 顺序
        seq_ids = list(contigs.keys())
        random.shuffle(seq_ids)
        
        for seq_idx, seq_id in enumerate(seq_ids):
            genes = contigs[seq_id]
            genes.sort(key=lambda x: x['start'])
            
            # 在非首个 contig 之前插入 SEP 分隔符
            if seq_idx > 0 and len(genome_genes) > 0:
                genome_genes.append(vocab["<SEP>"])
                genome_strands.append(strand_vocab["<PAD>"])
                genome_replicons.append(replicon_vocab["<PAD>"])
                genome_cogs.append(cog_vocab["<PAD>"])
                genome_contig_ids_local.append(min(contig_id_counter, 999))
                genome_mutations.append(0)
                genome_distances.append(0)
            
            for i, g in enumerate(genes):
                genome_genes.append(_get_or_add_vocab(g['id']))
                genome_strands.append(strand_vocab[g['strand']])
                genome_replicons.append(replicon_vocab[g['replicon']])
                genome_cogs.append(cog_vocab[g['cog']])
                genome_contig_ids_local.append(min(contig_id_counter, 999))
                genome_mutations.append(g['mutation'])
                
                if i == 0:
                    distance = 0
                else:
                    distance = g['start'] - genes[i - 1]['end']
                genome_distances.append(distance)
            
            contig_id_counter += 1
        
        # 使用 adaptive builder 切分
        n_genes = len(genome_genes)
        sentence_dicts = builder.build_sentences(
            genome_genes, genome_strands, genome_replicons,
            genome_cogs, genome_contig_ids_local, genome_mutations,
            genome_distances, vocab
        )
        
        if n_genes <= builder.effective_len:
            full_genome_count += 1
        else:
            windowed_count += 1
        
        for sd in sentence_dicts:
            all_sentences.append(sd["sentence"])
            all_strands.append(sd["strands"])
            all_replicons.append(sd["replicons"])
            all_cogs.append(sd["cogs"])
            all_contig_ids.append(sd["contigs"])
            all_mutations.append(sd["mutations"])
            all_distances.append(sd["distances"])
            genome_ids.append(genome_id)
    
    logging.info(f"Parsed {total_genes} genes into {len(all_sentences)} sentences.")
    logging.info(f"  Full-genome sentences (no splitting): {full_genome_count} genomes")
    logging.info(f"  Windowed sentences (needed splitting): {windowed_count} genomes")
    logging.info(f"From {len(set(genome_ids))} unique genomes (strains).")
    logging.info(f"Final Vocab Size: {len(vocab)}")
    
    # 序列长度统计
    if all_sentences:
        seq_lens = [len(s) for s in all_sentences]
        logging.info(f"Sentence lengths: min={min(seq_lens)}, max={max(seq_lens)}, "
                     f"mean={np.mean(seq_lens):.1f}, median={np.median(seq_lens):.1f}")
    else:
        logging.warning("No sentences generated! Check GFF files and genomes directory.")
    
    # 4. 加载 ESM 特征
    esm_files = glob.glob(os.path.join(dataset_dir, "features", "esm_embeddings*.h5"))
    
    esm_dim = 480
    if esm_files:
        try:
            with h5py.File(esm_files[0], 'r') as h5f:
                for grp_key in h5f.keys():
                    if isinstance(h5f[grp_key], h5py.Group):
                        for seq_key in h5f[grp_key].keys():
                            esm_dim = h5f[grp_key][seq_key].shape[0]
                            break
                        break
        except Exception as e:
            logging.warning(f"Could not probe ESM dimension: {e}")
    
    logging.info(f"Detected ESM feature dimension: {esm_dim}")
    esm_features = np.zeros((len(vocab), esm_dim), dtype=np.float32)
    
    found_esm = 0
    if esm_files:
        for esm_path in esm_files:
            logging.info(f"Loading ESM features from {esm_path}...")
            try:
                with h5py.File(esm_path, 'r') as h5f:
                    for group_key in h5f.keys():
                        grp = h5f[group_key]
                        if isinstance(grp, h5py.Dataset):
                            continue
                        for seq_id in grp.keys():
                            gene_id_mapped = query_to_gene.get(seq_id, seq_id)
                            if gene_id_mapped in vocab:
                                idx = vocab[gene_id_mapped]
                                esm_features[idx, :esm_dim] = grp[seq_id][:]
                                found_esm += 1
            except Exception as e:
                logging.error(f"Error loading ESM features from {esm_path}: {e}")
        logging.info(f"Successfully loaded ESM features for {found_esm}/{len(vocab)} genes.")
    else:
        logging.warning("No ESM feature files found. Using zero vectors.")
    
    # 5. 保存
    output_data = {
        "sentences": all_sentences,
        "strands": all_strands,
        "replicons": all_replicons,
        "cogs": all_cogs,
        "contigs": all_contig_ids,
        "mutations": all_mutations,
        "distances": all_distances,
        "genome_ids": genome_ids,
        "esm_features": torch.tensor(esm_features, dtype=torch.float),
        "vocabs": {
            "gene": vocab,
            "strand": strand_vocab,
            "replicon": replicon_vocab,
            "cog": cog_vocab,
            "mutation": mutation_vocab
        }
    }
    
    output_path = os.path.join(output_dir, "transformer_inputs_v3.pt")
    torch.save(output_data, output_path)
    logging.info(f"Successfully saved transformer inputs to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAGE v3: Generate Transformer inputs")
    parser.add_argument("--dataset_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset")
    parser.add_argument("--output_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/features/transformer_v3")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--overlap_ratio", type=float, default=0.25,
                        help="Overlap ratio for sliding window (default: 0.25)")
    
    args = parser.parse_args()
    
    build_transformer_inputs(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        overlap_ratio=args.overlap_ratio
    )
