"""
SAGE v3 Downstream: Strain Clustering Evaluation
==================================================
与 v2 的区别:
1. 使用 GenomicSentenceDatasetV3 (变长序列)
2. 使用 GenomicLanguageModelV3 (三阶段分层 + Progressive Fusion)
3. 评估时手动构建 attention masks (因为变长序列需要动态 padding)
"""

import os
import sys
import json
import glob
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, Counter
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GenomicLanguageModelV3
from dataset import GenomicSentenceDatasetV3
from model_configs import add_config_argument, get_model_config, apply_config_to_args


def detect_cog_degenerate(data):
    """检测 COG 数据是否退化"""
    cogs = data.get('cogs', None)
    if cogs is None:
        return True  # BUG7 fix: cogs 不存在时应视为退化，与 eval_pretrain_v3.py 一致
    all_cog_values = set()
    for sample_cogs in cogs:
        if isinstance(sample_cogs, torch.Tensor):
            all_cog_values.update(sample_cogs.tolist())
        else:
            all_cog_values.update(sample_cogs)
        if len(all_cog_values) > 2:
            return False
    if len(all_cog_values) <= 2:
        logging.info(f"COG degenerate (unique: {all_cog_values}). exclude_cog=True")
        return True
    return False


def eval_collate_fn(batch):
    """评估用 collate: 动态 padding, 不做 masking"""
    max_len = max(b['input_ids'].size(0) for b in batch)
    B = len(batch)
    
    padded = {}
    for key in batch[0].keys():
        if key == 'distance_ids':
            pad_val, dtype = 0.0, torch.float
        else:
            pad_val, dtype = 0, torch.long
        
        t = torch.full((B, max_len), pad_val, dtype=dtype)
        for i, b in enumerate(batch):
            L = b[key].size(0)
            t[i, :L] = b[key]
        padded[key] = t
    
    return padded


class StrainEmbeddingExtractorV3:
    """从预训练 SAGE v3 模型提取 strain-level embedding"""
    
    def __init__(self, model_path, data_path, device='cuda',
                 d_model=256, num_heads=8, num_layers=6, dim_feedforward=1024,
                 max_seq_len=2048, distance_window=128,
                 use_swiglu=True, use_hierarchical_attention=True,
                 use_gated_fusion=True, use_distance_bias=True,
                 use_token_level_gating=False, use_segment_level_gating=False,
                 contrastive_proj_dim=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logging.info(f"Loading data from {data_path}...")
        data = torch.load(data_path, map_location='cpu', weights_only=False)
        self.vocabs = data['vocabs']
        vocab_size = len(self.vocabs['gene'])
        
        esm_features = data.get('esm_features', None)
        feature_dim = esm_features.shape[1] if esm_features is not None else None
        exclude_cog = detect_cog_degenerate(data)
        
        logging.info(f"Initializing v3 model (d={d_model}, L={num_layers}, H={num_heads})...")
        self.model = GenomicLanguageModelV3(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            feature_dim=feature_dim,
            esm_features=esm_features,
            use_swiglu=use_swiglu,
            use_hierarchical_attention=use_hierarchical_attention,
            use_gated_fusion=use_gated_fusion,
            use_distance_bias=use_distance_bias,
            exclude_cog=exclude_cog,
            max_seq_len=max_seq_len,
            distance_window=distance_window,
            use_token_level_gating=use_token_level_gating,
            use_segment_level_gating=use_segment_level_gating,
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            logging.info(f"Loading weights from {model_path}...")
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                logging.warning(f"Missing keys: {len(missing)}")
            if unexpected:
                logging.warning(f"Unexpected keys: {len(unexpected)}")
            logging.info("Model weights loaded.")
        else:
            logging.warning(f"No checkpoint at {model_path}. Using random weights.")
        
        self.model.eval()
        self.pad_id = self.vocabs['gene'].get('<PAD>', 0)
    
    def _build_masks(self, input_ids, contig_ids):
        """构建 local + global masks"""
        same_contig = (contig_ids.unsqueeze(1) == contig_ids.unsqueeze(2))
        pad_mask = (input_ids != self.pad_id).unsqueeze(1)
        local_mask = (same_contig & pad_mask).long()
        local_mask.diagonal(dim1=-2, dim2=-1).fill_(1)
        
        row_valid = local_mask.any(dim=-1, keepdim=True)
        col_valid = local_mask.any(dim=-2, keepdim=True)
        global_mask = (row_valid & col_valid).long()
        
        return local_mask.unsqueeze(1), global_mask.unsqueeze(1)
    
    @torch.no_grad()
    def extract_embeddings(self, dataset, batch_size=16, use_attention_pooling=True):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=eval_collate_fn, num_workers=2)
        
        all_embeddings = []
        all_attn_weights = []
        
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(self.device)
            strand_ids = batch['strand_ids'].to(self.device)
            replicon_ids = batch['replicon_ids'].to(self.device)
            cog_ids = batch['cog_ids'].to(self.device)
            contig_ids = batch['contig_ids'].to(self.device)
            mutation_ids = batch['mutation_ids'].to(self.device)
            distance_ids = batch['distance_ids'].to(self.device)
            position_ids = batch['position_ids'].to(self.device)
            
            local_mask, global_mask = self._build_masks(input_ids, contig_ids)
            
            result = self.model(
                gene_seqs=input_ids,
                strand_ids=strand_ids,
                replicon_ids=replicon_ids,
                cog_ids=cog_ids,
                contig_ids=contig_ids,
                mutation_ids=mutation_ids,
                distance_ids=distance_ids,
                position_ids=position_ids,
                mask=local_mask,
                global_mask=global_mask,
                extract_features=True,
                return_pooled=use_attention_pooling,
            )
            
            if use_attention_pooling:
                gene_emb, strain_emb, attn_w = result
                all_embeddings.append(strain_emb.cpu().numpy())
                # 存储每个样本的 attention weights (变长, 后续统一处理)
                attn_np = attn_w.squeeze(1).cpu().numpy()  # [B, L]
                for row in attn_np:
                    all_attn_weights.append(row)  # 逐样本存储, 避免长度不一致
            else:
                gene_emb = result
                pad_mask = (input_ids != self.pad_id).float().unsqueeze(-1)
                sum_emb = (gene_emb * pad_mask).sum(dim=1)
                count = pad_mask.sum(dim=1).clamp(min=1)
                mean_emb = sum_emb / count
                all_embeddings.append(mean_emb.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logging.info(f"  Extracted {(batch_idx + 1) * batch_size}/{len(dataset)} samples")
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Pad attention weights 到统一最大长度
        if all_attn_weights:
            max_attn_len = max(w.shape[0] for w in all_attn_weights)
            padded_attn = np.zeros((len(all_attn_weights), max_attn_len), dtype=np.float32)
            for i, w in enumerate(all_attn_weights):
                padded_attn[i, :w.shape[0]] = w
            attn_weights = padded_attn
        else:
            attn_weights = None
        
        logging.info(f"Extracted embeddings: shape={embeddings.shape}")
        return embeddings, attn_weights


def assign_genus_labels(dataset, genomes_dir, data_path):
    """
    为每个 sentence 分配属名标签.
    
    策略:
    1. 优先从 genome_ids + 目录结构推断 (快速且可靠)
    2. Fallback: 通过 GFF 基因集合匹配
    """
    logging.info("Assigning genus labels...")
    
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    sentences = data['sentences']
    genome_ids = data.get('genome_ids', None)
    
    # 策略 1: 从 genome_ids + 目录结构推断
    # genome_id 通常是 GCA_xxx, 目录结构: genomes/<Genus>/<GCA_xxx>/genomic.gff
    if genome_ids is not None and os.path.isdir(genomes_dir):
        # 建立 GCA → Genus 映射
        gca_to_genus = {}
        for genus_dir in os.listdir(genomes_dir):
            genus_path = os.path.join(genomes_dir, genus_dir)
            if not os.path.isdir(genus_path):
                continue
            for gca_dir in os.listdir(genus_path):
                if gca_dir.startswith("GCA_"):
                    gca_to_genus[gca_dir] = genus_dir
        
        if gca_to_genus:
            labels = []
            gca_labels = []
            for gid in genome_ids:
                genus = gca_to_genus.get(gid, 'Unknown')
                labels.append(genus)
                gca_labels.append(gid)
            
            found = sum(1 for l in labels if l != 'Unknown')
            logging.info(f"  Strategy 1 (genome_ids → genus): {found}/{len(labels)} assigned")
            
            if found > 0:
                label_counts = Counter(labels)
                logging.info(f"Genus distribution ({len(label_counts)} genera):")
                for genus, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
                    logging.info(f"  {genus}: {count}")
                return labels, gca_labels
            else:
                logging.warning("  Strategy 1 found no matches, trying fallback...")
    
    # 策略 2 (Fallback): 通过 GFF 基因集合匹配
    logging.info("  Using fallback: GFF gene-set matching...")
    vocab = data['vocabs']['gene']
    
    gff_files = sorted(glob.glob(os.path.join(genomes_dir, "**", "*.gff"), recursive=True))
    gff_files = [f for f in gff_files if '.ipynb_checkpoints' not in f]
    
    gff_to_genus = {}
    gff_to_gca = {}
    for gff in gff_files:
        parts = gff.split(os.sep)
        genus = None
        gca = None
        for i, p in enumerate(parts):
            if p.startswith("GCA_"):
                gca = p
            if p == 'genomes' and i + 1 < len(parts):
                genus = parts[i + 1]
        gff_to_genus[gff] = genus or 'Unknown'
        gff_to_gca[gff] = gca or 'Unknown'
    
    gff_gene_sets = {}
    for gff in gff_files:
        genes = set()
        try:
            import re
            with open(gff, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts_line = line.strip().split('\t')
                    if len(parts_line) < 9:
                        continue
                    if parts_line[2] not in ['CDS', 'gene']:
                        continue
                    locus = re.search(r'locus_tag=([^;]+)', parts_line[8])
                    if locus:
                        gene_id = locus.group(1)
                        if gene_id in vocab:
                            genes.add(vocab[gene_id])
        except Exception:
            pass
        gff_gene_sets[gff] = genes
    
    special_ids = {vocab.get('<PAD>', 0), vocab.get('<CLS>', 3), vocab.get('<SEP>', 4),
                   vocab.get('<MASK>', 2), vocab.get('<UNK>', 1), vocab.get('<END>', 5)}
    
    labels = []
    gca_labels = []
    for sent in sentences:
        sent_genes = set(sent) - special_ids
        best_gff = None
        best_overlap = 0
        for gff, gene_set in gff_gene_sets.items():
            overlap = len(sent_genes & gene_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_gff = gff
        
        labels.append(gff_to_genus.get(best_gff, 'Unknown') if best_gff else 'Unknown')
        gca_labels.append(gff_to_gca.get(best_gff, 'Unknown') if best_gff else 'Unknown')
    
    label_counts = Counter(labels)
    logging.info(f"Genus distribution ({len(label_counts)} genera):")
    for genus, count in sorted(label_counts.items(), key=lambda x: -x[1])[:10]:
        logging.info(f"  {genus}: {count}")
    
    return labels, gca_labels


def evaluate_clustering(embeddings, labels):
    """评估聚类质量: ARI, NMI, Silhouette"""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    
    le = LabelEncoder()
    true_labels = le.fit_transform(labels)
    n_clusters = len(set(true_labels))
    
    logging.info(f"Evaluating: {len(embeddings)} samples, {n_clusters} clusters")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings)
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    sil = silhouette_score(embeddings, true_labels, metric='cosine') if len(embeddings) > n_clusters > 1 else 0.0
    
    results = {
        'ARI': round(ari, 4), 'NMI': round(nmi, 4), 'Silhouette': round(sil, 4),
        'n_samples': len(embeddings), 'n_clusters': n_clusters,
    }
    logging.info(f"  ARI={ari:.4f}, NMI={nmi:.4f}, Silhouette={sil:.4f}")
    return results


def visualize_embeddings(embeddings, labels, output_path, method='tsne', title=''):
    """t-SNE / UMAP 可视化"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import LabelEncoder
        
        le = LabelEncoder()
        numeric_labels = le.fit_transform(labels)
        
        if method == 'umap':
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                coords = reducer.fit_transform(embeddings)
            except ImportError:
                logging.warning("UMAP not installed, using t-SNE")
                method = 'tsne'
        
        if method == 'tsne':
            from sklearn.manifold import TSNE
            perplexity = min(30, len(embeddings) - 1)
            reducer = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
            coords = reducer.fit_transform(embeddings)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap('tab20', len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], label=label, s=30, alpha=0.7)
        
        ax.set_title(f'SAGE v3 Strain Embeddings ({method.upper()}) {title}', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved: {output_path}")
    except ImportError as e:
        logging.warning(f"Visualization skipped: {e}")


def get_args():
    parser = argparse.ArgumentParser(description="SAGE v3 Downstream: Clustering")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/features/transformer_v3/transformer_inputs_v3.pt")
    parser.add_argument("--genomes_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/genomes")
    parser.add_argument("--output_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/results/downstream_v3")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--distance_window", type=int, default=128)
    parser.add_argument("--use_swiglu", action="store_true", default=True)
    parser.add_argument("--no_swiglu", dest="use_swiglu", action="store_false")
    parser.add_argument("--use_hierarchical_attention", action="store_true", default=True)
    parser.add_argument("--no_hierarchical_attention", dest="use_hierarchical_attention", action="store_false")
    parser.add_argument("--use_gated_fusion", action="store_true", default=True)
    parser.add_argument("--no_gated_fusion", dest="use_gated_fusion", action="store_false")
    parser.add_argument("--use_distance_bias", action="store_true", default=True)
    parser.add_argument("--no_distance_bias", dest="use_distance_bias", action="store_false")
    parser.add_argument("--vis_method", type=str, default="tsne", choices=["umap", "tsne"])

    # 以下参数仅训练时使用, 评估时忽略, 但需要声明以兼容消融实验的 EXTRA_ARGS 透传
    parser.add_argument("--cog_loss_weight", type=float, default=0.5,
                        help="(ignored in eval)")
    parser.add_argument("--strand_loss_weight", type=float, default=0.5,
                        help="(ignored in eval)")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.0,
                        help="(ignored in eval)")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="(ignored in eval)")
    parser.add_argument("--contrastive_proj_dim", type=int, default=128,
                        help="(ignored in eval) Contrastive projection head dimension")
    parser.add_argument("--use_targeted_masking", action="store_true", default=True,
                        help="(ignored in eval)")
    parser.add_argument("--no_targeted_masking", dest="use_targeted_masking",
                        action="store_false", help="(ignored in eval)")
    parser.add_argument("--mask_prob", type=float, default=0.15,
                        help="(ignored in eval)")
    parser.add_argument("--span_length", type=int, default=3,
                        help="(ignored in eval)")
    parser.add_argument("--operon_mask_prob", type=float, default=0.05,
                        help="(ignored in eval)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="(ignored in eval)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false", help="(ignored in eval)")
    parser.add_argument("--use_token_level_gating", action="store_true", default=False,
                        help="(ignored in eval)")
    parser.add_argument("--use_segment_level_gating", action="store_true", default=False,
                        help="(ignored in eval)")

    # 模型配置预设
    add_config_argument(parser)

    args = parser.parse_args()

    if args.model_config:
        config = get_model_config(args.model_config)
        apply_config_to_args(args, config)

    return args


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("SAGE v3 Downstream: Strain Clustering Evaluation")
    logging.info("=" * 60)
    
    extractor = StrainEmbeddingExtractorV3(
        model_path=args.model_path, data_path=args.data_path,
        d_model=args.d_model, num_heads=args.num_heads,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        max_seq_len=args.max_seq_len, distance_window=args.distance_window,
        use_swiglu=args.use_swiglu,
        use_hierarchical_attention=args.use_hierarchical_attention,
        use_gated_fusion=args.use_gated_fusion,
        use_distance_bias=args.use_distance_bias,
        use_token_level_gating=args.use_token_level_gating,
        use_segment_level_gating=args.use_segment_level_gating,
        contrastive_proj_dim=args.contrastive_proj_dim,
    )
    
    dataset = GenomicSentenceDatasetV3(args.data_path)
    genus_labels, gca_labels = assign_genus_labels(dataset, args.genomes_dir, args.data_path)
    
    logging.info("\n--- Extracting SAGE v3 embeddings (Attention Pooling) ---")
    emb_attn, attn_weights = extractor.extract_embeddings(
        dataset, batch_size=args.batch_size, use_attention_pooling=True
    )
    
    logging.info("\n--- Extracting SAGE v3 embeddings (Mean Pooling) ---")
    emb_mean, _ = extractor.extract_embeddings(
        dataset, batch_size=args.batch_size, use_attention_pooling=False
    )
    
    logging.info("\n--- Random baseline ---")
    emb_random = np.random.randn(*emb_attn.shape).astype(np.float32)
    
    logging.info("\n" + "=" * 60)
    results = {}
    
    logging.info("[1] SAGE v3 + Attention Pooling:")
    results['sage_v3_attn_pool'] = evaluate_clustering(emb_attn, genus_labels)
    
    logging.info("[2] SAGE v3 + Mean Pooling:")
    results['sage_v3_mean_pool'] = evaluate_clustering(emb_mean, genus_labels)
    
    logging.info("[3] Random Baseline:")
    results['random'] = evaluate_clustering(emb_random, genus_labels)
    
    results_path = os.path.join(args.output_dir, "clustering_results_v3.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("\n--- Visualizations ---")
    visualize_embeddings(emb_attn, genus_labels,
                        os.path.join(args.output_dir, f"clustering_{args.vis_method}_attn_v3.png"),
                        method=args.vis_method, title='(Attn Pool)')
    visualize_embeddings(emb_mean, genus_labels,
                        os.path.join(args.output_dir, f"clustering_{args.vis_method}_mean_v3.png"),
                        method=args.vis_method, title='(Mean Pool)')
    
    np.save(os.path.join(args.output_dir, "strain_embeddings_attn_v3.npy"), emb_attn)
    np.save(os.path.join(args.output_dir, "strain_embeddings_mean_v3.npy"), emb_mean)
    if attn_weights is not None:
        np.save(os.path.join(args.output_dir, "attention_weights_v3.npy"), attn_weights)
    
    with open(os.path.join(args.output_dir, "embedding_labels_v3.json"), 'w') as f:
        json.dump({'genus': genus_labels, 'gca': gca_labels}, f)
    
    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info(f"{'Method':<30} {'ARI':>8} {'NMI':>8} {'Silhouette':>12}")
    logging.info("-" * 60)
    for name, res in results.items():
        logging.info(f"{name:<30} {res['ARI']:>8.4f} {res['NMI']:>8.4f} {res['Silhouette']:>12.4f}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
