"""
SAGE v4: Streaming Dataset for Large-Scale Genome Pre-training
================================================================
Replaces generate_input.py + dataset.py with a fully streaming pipeline.

Key features:
  1. IterableDataset: No memory explosion — reads shard files on-the-fly
  2. Dynamic contig shuffling: Data augmentation at load time (like Bacformer)
  3. On-the-fly ESM feature lookup: Loads per-shard .pt files lazily
  4. DDP-aware shard partitioning: Each rank processes disjoint shard subsets
  5. Max sequence length 6000: Covers 98%+ complete bacterial genomes
  6. Contig-aware sentence construction with [CLS] contig1 [SEP] contig2 [END]

Usage:
    from dataset_streaming import SAGEStreamingDataset, get_streaming_collator
    dataset = SAGEStreamingDataset(
        metadata_dir="/path/to/annotations_sharded",
        esm_dir="/path/to/esm_sharded",
        max_seq_len=6000,
    )
    collate_fn = get_streaming_collator(max_seq_len=6000)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=8)
"""

import os
import json
import random
import logging
import hashlib
import torch
from torch.utils.data import IterableDataset, get_worker_info
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Vocab definitions (compatible with v3)
# ============================================================================
GENE_SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "<CLS>": 3, "<SEP>": 4, "<END>": 5}
NUM_SPECIAL = len(GENE_SPECIAL_TOKENS)  # 6
HASH_VOCAB_SIZE = 2**16  # hash bucket size for gene IDs
TOTAL_VOCAB_SIZE = HASH_VOCAB_SIZE + NUM_SPECIAL  # 65542, must match model's vocab_size

STRAND_VOCAB = {"<PAD>": 0, "+": 1, "-": 2}
REPLICON_VOCAB = {"<PAD>": 0, "chromosome": 1, "plasmid": 2, "unknown": 3}
MUTATION_VOCAB = {"<PAD>": 0, "wildtype": 1, "single_mut": 2, "multi_mut": 3}

COG_VOCAB = {"<PAD>": 0, "<UNK>": 1}
for _c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ-":
    COG_VOCAB[_c] = len(COG_VOCAB)


class ESMFeatureCache:
    """Lazy-loading cache for per-shard ESM features."""

    def __init__(self, esm_dir):
        self.esm_dir = esm_dir
        self._cache = {}      # shard_id → dict
        self._current_shard = None

    def get(self, shard_id, query_id):
        """Get ESM embedding for a protein. Returns FP16 tensor or None."""
        if shard_id != self._current_shard:
            self._load_shard(shard_id)
        return self._cache.get(query_id)

    def _load_shard(self, shard_id):
        """Load a shard's ESM features, evicting the previous one."""
        self._cache.clear()
        self._current_shard = shard_id

        pt_path = os.path.join(self.esm_dir, f"esm_shard_{shard_id:04d}.pt")
        if os.path.exists(pt_path):
            try:
                data = torch.load(pt_path, map_location='cpu', weights_only=False)
                self._cache = data
            except Exception as e:
                logging.warning(f"Failed to load ESM shard {shard_id}: {e}")

    def get_esm_dim(self):
        """Probe ESM dimension from first available shard."""
        pt_files = sorted(glob.glob(os.path.join(self.esm_dir, "esm_shard_*.pt")))
        for pt_file in pt_files:
            try:
                data = torch.load(pt_file, map_location='cpu', weights_only=False)
                for key in data:
                    return data[key].shape[0]
            except Exception:
                continue
        return 480  # default ESM-2 t12 dimension


def build_genome_sample(genome_data, shard_id, esm_cache, max_seq_len=6000, esm_dim=480):
    """
    Convert a single genome's metadata into a training sample.
    
    Implements:
      - Random contig order shuffling (data augmentation)
      - [CLS] contig1_genes [SEP] contig2_genes ... [END] formatting
      - Truncation to max_seq_len
      - ESM feature lookup per gene
    
    Returns:
        dict with all feature tensors, or None if genome is empty.
    """
    contigs = genome_data.get('contigs', [])
    if not contigs:
        return None

    # Data augmentation: shuffle contig order
    contig_order = list(range(len(contigs)))
    random.shuffle(contig_order)

    # Build flattened gene sequence with separators
    gene_ids = []       # string IDs for ESM lookup
    query_ids = []      # protein IDs for ESM lookup
    strands = []
    replicons = []
    cogs = []
    contig_ids = []
    mutations = []
    distances = []

    # Start with CLS
    effective_len = max_seq_len - 2  # reserve CLS + END

    contig_counter = 1
    for idx, contig_idx in enumerate(contig_order):
        contig = contigs[contig_idx]
        genes = contig['genes']

        if not genes:
            continue

        # Insert SEP between contigs (not before the first one)
        if idx > 0 and len(gene_ids) > 0:
            gene_ids.append(None)  # SEP marker
            query_ids.append(None)
            strands.append(STRAND_VOCAB["<PAD>"])
            replicons.append(REPLICON_VOCAB["<PAD>"])
            cogs.append(COG_VOCAB["<PAD>"])
            contig_ids.append(min(contig_counter, 999))
            mutations.append(0)
            distances.append(0)

        for g in genes:
            gene_ids.append(g['gene_id'])
            query_ids.append(g.get('query_id', g['gene_id']))
            strands.append(STRAND_VOCAB.get(g['strand'], 1))
            replicons.append(REPLICON_VOCAB.get(g['replicon'], 3))
            cog_char = g.get('cog', '-')
            cogs.append(COG_VOCAB.get(cog_char, COG_VOCAB.get('-', 28)))
            contig_ids.append(min(contig_counter, 999))
            mutations.append(g.get('mutation', 1))
            distances.append(g.get('distance', 0))

        contig_counter += 1

        # Early termination if we're already past max length
        if len(gene_ids) >= effective_len:
            gene_ids = gene_ids[:effective_len]
            query_ids = query_ids[:effective_len]
            strands = strands[:effective_len]
            replicons = replicons[:effective_len]
            cogs = cogs[:effective_len]
            contig_ids = contig_ids[:effective_len]
            mutations = mutations[:effective_len]
            distances = distances[:effective_len]
            break

    if not gene_ids:
        return None

    # Build ESM features for each gene position
    esm_features = []
    gene_token_ids = []  # integer IDs (hash-based, for MLM prediction target)

    for i, (gid, qid) in enumerate(zip(gene_ids, query_ids)):
        if gid is None:
            # SEP token
            gene_token_ids.append(GENE_SPECIAL_TOKENS["<SEP>"])
            esm_features.append(torch.zeros(esm_dim, dtype=torch.float16))
        else:
            # Use hash-based token ID (deterministic, no global vocab needed)
            # Range: [NUM_SPECIAL, NUM_SPECIAL + HASH_VOCAB_SIZE)
            token_id = int(hashlib.md5(gid.encode()).hexdigest(), 16) % HASH_VOCAB_SIZE + NUM_SPECIAL
            gene_token_ids.append(token_id)

            # Look up ESM feature
            esm_feat = esm_cache.get(shard_id, qid) if esm_cache else None
            if esm_feat is None:
                esm_feat = torch.zeros(esm_dim, dtype=torch.float16)
            esm_features.append(esm_feat)

    # Wrap with CLS and END
    gene_token_ids = [GENE_SPECIAL_TOKENS["<CLS>"]] + gene_token_ids + [GENE_SPECIAL_TOKENS["<END>"]]
    strands = [0] + strands + [0]
    replicons = [0] + replicons + [0]
    cogs = [0] + cogs + [0]
    contig_ids = [contig_ids[0] if contig_ids else 0] + contig_ids + [contig_ids[-1] if contig_ids else 0]
    mutations = [0] + mutations + [0]
    distances = [0] + distances + [0]

    # ESM features: pad CLS and END with zeros
    zero_feat = torch.zeros(esm_dim, dtype=torch.float16)
    esm_features = [zero_feat] + esm_features + [zero_feat]
    esm_tensor = torch.stack(esm_features)  # [L, esm_dim]

    # Build per-contig position IDs
    pos_ids = []
    pos = 0
    prev_c = contig_ids[0]
    for c in contig_ids:
        if c == prev_c:
            pos_ids.append(pos)
            pos += 1
        else:
            pos = 0
            pos_ids.append(pos)
            pos += 1
            prev_c = c

    return {
        "input_ids": torch.tensor(gene_token_ids, dtype=torch.long),
        "esm_features": esm_tensor,
        "strand_ids": torch.tensor(strands, dtype=torch.long),
        "replicon_ids": torch.tensor(replicons, dtype=torch.long),
        "cog_ids": torch.tensor(cogs, dtype=torch.long),
        "contig_ids": torch.tensor(contig_ids, dtype=torch.long),
        "mutation_ids": torch.tensor(mutations, dtype=torch.long),
        "distance_ids": torch.tensor(distances, dtype=torch.float),
        "position_ids": torch.tensor(pos_ids, dtype=torch.long),
        "genome_id": genome_data['genome_id'],
        "genus": genome_data.get('genus', 'Unknown'),
    }


class SAGEStreamingDataset(IterableDataset):
    """
    Streaming dataset for SAGE v4 large-scale pre-training.
    
    Reads metadata JSONL files and ESM .pt files per shard,
    yielding individual genome samples on-the-fly.
    
    Supports:
      - Multi-worker data loading (each worker gets disjoint shards)
      - DDP rank-aware partitioning
      - Epoch-based shard shuffling
    """

    def __init__(self, metadata_dir, esm_dir=None, max_seq_len=6000,
                 shuffle_shards=True, seed=42):
        super().__init__()
        self.metadata_dir = metadata_dir
        self.esm_dir = esm_dir
        self.max_seq_len = max_seq_len
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.epoch = 0

        # Discover shard files
        self.shard_files = sorted(glob.glob(os.path.join(metadata_dir, "shard_*_metadata.jsonl")))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard metadata files found in {metadata_dir}")

        logging.info(f"SAGEStreamingDataset: {len(self.shard_files)} shards in {metadata_dir}")

        # Probe ESM dimension
        self.esm_dim = 480
        if esm_dir:
            cache = ESMFeatureCache(esm_dir)
            self.esm_dim = cache.get_esm_dim()
            logging.info(f"  ESM feature dim: {self.esm_dim}")

    def set_epoch(self, epoch):
        """Set epoch for reproducible shard shuffling (call before each epoch)."""
        self.epoch = epoch

    def _get_worker_shards(self):
        """Partition shards across workers and DDP ranks."""
        shard_indices = list(range(len(self.shard_files)))

        if self.shuffle_shards:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(shard_indices)

        # DDP partitioning
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Split across DDP ranks
        rank_shards = shard_indices[rank::world_size]

        # Split across DataLoader workers
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_shards = rank_shards[worker_id::num_workers]
        else:
            worker_shards = rank_shards

        return worker_shards

    def __iter__(self):
        worker_shards = self._get_worker_shards()

        esm_cache = ESMFeatureCache(self.esm_dir) if self.esm_dir else None

        for shard_idx in worker_shards:
            shard_file = self.shard_files[shard_idx]

            # Extract shard_id from filename: shard_XXXX_metadata.jsonl → XXXX
            basename = os.path.basename(shard_file)
            try:
                # "shard_0001_metadata.jsonl" → split('_') → ['shard', '0001', 'metadata.jsonl']
                parts = basename.split('_')
                shard_id = int(parts[1])
            except (IndexError, ValueError):
                shard_id = shard_idx

            # Read and process each genome in the shard
            genomes_in_shard = []
            try:
                with open(shard_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            genomes_in_shard.append(json.loads(line))
            except Exception as e:
                logging.warning(f"Error reading shard {shard_file}: {e}")
                continue

            # Shuffle genomes within shard
            if self.shuffle_shards:
                rng = random.Random(self.seed + self.epoch + shard_idx)
                rng.shuffle(genomes_in_shard)

            for genome_data in genomes_in_shard:
                sample = build_genome_sample(
                    genome_data, shard_id, esm_cache,
                    max_seq_len=self.max_seq_len, esm_dim=self.esm_dim
                )
                if sample is not None:
                    yield sample


def get_streaming_collator(max_seq_len=6000, mask_prob=0.15, span_length=3,
                           use_targeted_masking=True, operon_mask_prob=0.05):
    """
    Dynamic MLM collator for streaming dataset.
    Performs:
      1. Dynamic padding to batch-max length
      2. Span masking with contig awareness
      3. Targeted masking (mutation, defense COG, plasmid)
      4. Contig-blocked attention mask construction
    """
    pad_id = GENE_SPECIAL_TOKENS["<PAD>"]
    cls_id = GENE_SPECIAL_TOKENS["<CLS>"]
    sep_id = GENE_SPECIAL_TOKENS["<SEP>"]
    end_id = GENE_SPECIAL_TOKENS["<END>"]
    mask_id = GENE_SPECIAL_TOKENS["<MASK>"]

    def collate_fn(batch):
        max_len = min(max(b['input_ids'].size(0) for b in batch), max_seq_len)
        B = len(batch)

        # Probe ESM dim from first sample
        esm_dim = batch[0]['esm_features'].size(1)

        seqs = torch.full((B, max_len), pad_id, dtype=torch.long)
        esm_feats = torch.zeros(B, max_len, esm_dim, dtype=torch.float16)
        strands = torch.zeros(B, max_len, dtype=torch.long)
        replicons = torch.zeros(B, max_len, dtype=torch.long)
        cogs = torch.zeros(B, max_len, dtype=torch.long)
        contigs = torch.zeros(B, max_len, dtype=torch.long)
        mutations = torch.zeros(B, max_len, dtype=torch.long)
        distances = torch.zeros(B, max_len, dtype=torch.float)
        position_ids = torch.zeros(B, max_len, dtype=torch.long)
        genome_ids = []
        genus_names = []

        for i, b in enumerate(batch):
            L = min(b['input_ids'].size(0), max_len)
            seqs[i, :L] = b['input_ids'][:L]
            esm_feats[i, :L] = b['esm_features'][:L]
            strands[i, :L] = b['strand_ids'][:L]
            replicons[i, :L] = b['replicon_ids'][:L]
            cogs[i, :L] = b['cog_ids'][:L]
            contigs[i, :L] = b['contig_ids'][:L]
            mutations[i, :L] = b['mutation_ids'][:L]
            distances[i, :L] = b['distance_ids'][:L]
            position_ids[i, :L] = b['position_ids'][:L]
            genome_ids.append(b.get('genome_id', ''))
            genus_names.append(b.get('genus', 'Unknown'))

        # Labels (before masking)
        labels = seqs.clone()
        labels_strand = strands.clone()
        labels_cog = cogs.clone()

        # ★ ESM Regression Target: 保存原始 ESM 特征 (mask 之前)
        # 作为 regression MLM 的预测目标
        esm_targets = esm_feats.clone()  # [B, L, esm_dim] fp16

        # Masking
        span_prob = mask_prob / span_length
        probability_matrix = torch.full(labels.shape, span_prob)

        special_tokens_mask = (
            (seqs == pad_id) | (seqs == cls_id) |
            (seqs == sep_id) | (seqs == end_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        if use_targeted_masking:
            mutated = mutations > 1
            probability_matrix[mutated] = min(span_prob * 3.0, 0.5)
            defense_cog = (cogs == 23)  # V = Defense mechanisms
            probability_matrix[defense_cog] = min(span_prob * 2.0, 0.5)
            on_plasmid = (replicons == 2)
            probability_matrix[on_plasmid] = torch.max(
                probability_matrix[on_plasmid],
                torch.tensor(min(span_prob * 1.5, 0.5))
            )

        probability_matrix.masked_fill_(special_tokens_mask, 0.0)

        # Span masking
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

        # Operon masking
        if use_targeted_masking and operon_mask_prob > 0:
            operon_mask = _detect_operons(distances, contigs, special_tokens_mask,
                                          min_genes=3, max_distance=300,
                                          mask_prob=operon_mask_prob)
            masked_indices = masked_indices | operon_mask

        masked_indices.masked_fill_(special_tokens_mask, False)

        labels[~masked_indices] = -100
        labels_strand[~masked_indices] = -100
        labels_cog[~masked_indices] = -100

        # BERT-style masking for input_ids: 80% MASK, 10% random, 10% unchanged
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        seqs[indices_replaced] = mask_id
        strands[indices_replaced] = 0
        replicons[indices_replaced] = 0
        cogs[indices_replaced] = 0
        mutations[indices_replaced] = 0
        distances[indices_replaced] = 0

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(NUM_SPECIAL, TOTAL_VOCAB_SIZE, labels.shape, dtype=torch.long)
        seqs[indices_random] = random_words[indices_random]
        strands[indices_random] = 0
        replicons[indices_random] = 0
        cogs[indices_random] = 0
        mutations[indices_random] = 0
        distances[indices_random] = 0

        # ★ ESM 输入清零: 对所有 masked_indices 位置统一清零
        # (包括 10% unchanged, 避免 ESM regression 信息泄漏)
        esm_feats[masked_indices] = 0

        # Build contig-blocked attention mask
        same_contig_mask = (contigs.unsqueeze(1) == contigs.unsqueeze(2))
        pad_mask = (seqs != pad_id).unsqueeze(1)
        local_mask = (same_contig_mask & pad_mask).long()
        local_mask.diagonal(dim1=-2, dim2=-1).fill_(1)

        # Genus labels for contrastive learning
        # Use deterministic hash-based mapping so the same genus always gets
        # the same ID regardless of batch composition or GPU rank (critical for DDP).
        genus_ids = torch.tensor(
            [int(hashlib.md5(g.encode()).hexdigest(), 16) % (2**31) for g in genus_names],
            dtype=torch.long,
        )

        return {
            "input_ids": seqs,
            "esm_features": esm_feats.float(),  # convert to FP32 for model input
            "esm_targets": esm_targets.float(),  # ★ ESM regression target (FP32)
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
            "genus_ids": genus_ids,
            "genome_ids": genome_ids,
        }

    return collate_fn


def _detect_operons(distances, contigs, special_mask,
                    min_genes=3, max_distance=300, mask_prob=0.05):
    """Detect dense gene clusters (operons) and mask with given probability."""
    B, L = distances.shape
    operon_mask = torch.zeros(B, L, dtype=torch.bool)

    for b in range(B):
        i = 0
        while i < L:
            if special_mask[b, i]:
                i += 1
                continue
            j = i + 1
            while j < L and not special_mask[b, j]:
                if contigs[b, j] != contigs[b, i]:
                    break
                if distances[b, j].abs().item() > max_distance:
                    break
                j += 1
            if j - i >= min_genes:
                if random.random() < mask_prob:
                    operon_mask[b, i:j] = True
            i = j

    return operon_mask
