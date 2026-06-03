"""
SAGE v4: Generate Shard Index for Large-Scale Genome Processing
================================================================
Step 1 of the scalable pipeline:
  - Scan genome directory using `os.scandir` (fast, avoids glob)
  - Build index of all .gff and .faa file paths
  - Divide into shards of N genomes each
  - Save shard assignments as JSON for downstream parallel processing

Usage:
    python generate_shard_index.py \
        --genomes_dir /path/to/by_pathogen \
        --output_dir /path/to/shard_index \
        --shard_size 1000
"""

import os
import json
import argparse
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _get_gca_entries(genus_path, genus_name):
    """Collect all GCA_ directory paths under a single genus directory."""
    entries = []
    try:
        for entry in os.scandir(genus_path):
            if entry.is_dir(follow_symlinks=True) and entry.name.startswith("GCA_"):
                entries.append((entry.path, entry.name, genus_name))
    except OSError:
        pass
    return entries


def _scan_gca_batch(gca_batch):
    """Scan a batch of GCA directories for gff/faa/fna files (thread-safe)."""
    genomes = []
    skipped = 0
    for gca_path, gca_name, genus_name in gca_batch:
        gff_path = faa_path = fna_path = None
        try:
            for f in os.scandir(gca_path):
                if f.name.endswith('.gff'):
                    gff_path = f.path
                elif f.name.endswith('.faa'):
                    faa_path = f.path
                elif f.name.endswith('.fna'):
                    fna_path = f.path
        except OSError:
            skipped += 1
            continue

        if gff_path and faa_path:
            genomes.append({
                "genome_id": gca_name,
                "genus": genus_name,
                "gff": gff_path,
                "faa": faa_path,
                "fna": fna_path,
            })
        else:
            skipped += 1
    return genomes, skipped


def scan_genomes(genomes_dir):
    """
    Two-stage high-concurrency genome scanner.
    Stage 1: Collect all GCA_ directories across all genera (fast, shallow scan).
    Stage 2: Batch-scan GCA directories at genome level with 256 threads,
             eliminating the long-tail bottleneck from large genera.

    Expected layout: genomes_dir/<Genus>/<GCA_xxx>/{*.gff, *.faa, *.fna}
    """
    genomes = []
    skipped = 0

    logging.info(f"Scanning genome directory (two-stage parallel): {genomes_dir}")

    try:
        genus_entries = sorted(os.scandir(genomes_dir), key=lambda e: e.name)
    except OSError as e:
        logging.error(f"Cannot scan {genomes_dir}: {e}")
        return genomes

    valid_genera = [
        e for e in genus_entries
        if e.is_dir(follow_symlinks=True) and not e.name.startswith('.')
    ]
    logging.info(f"Found {len(valid_genera)} genus directories, collecting GCA directories...")

    # Stage 1: Multi-threaded collection of all GCA directories
    all_gca_entries = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(_get_gca_entries, e.path, e.name) for e in valid_genera]
        for future in as_completed(futures):
            all_gca_entries.extend(future.result())

    total_gca = len(all_gca_entries)
    logging.info(f"Found {total_gca} GCA directories, launching batched parallel scan...")

    # Stage 2: Batch scan at genome level with high concurrency
    batch_size = 200
    batches = [all_gca_entries[i:i + batch_size] for i in range(0, total_gca, batch_size)]
    total_batches = len(batches)
    done_batches = 0

    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(_scan_gca_batch, batch) for batch in batches]

        for future in as_completed(futures):
            g_list, s_count = future.result()
            genomes.extend(g_list)
            skipped += s_count
            done_batches += 1
            if done_batches % 50 == 0 or done_batches == total_batches:
                logging.info(f"Progress: {done_batches}/{total_batches} batches scanned, "
                             f"{len(genomes)} valid genomes so far...")

    logging.info(f"Found {len(genomes)} complete genomes (skipped {skipped} incomplete)")
    return genomes


def build_shards(genomes, shard_size):
    """
    Divide genome list into shards of shard_size.
    
    Returns:
        list of list: each sub-list is a shard of genome dicts
    """
    shards = []
    for i in range(0, len(genomes), shard_size):
        shards.append(genomes[i:i + shard_size])
    return shards


def main():
    parser = argparse.ArgumentParser(description="SAGE v4: Generate shard index for large-scale genome processing")
    parser.add_argument("--genomes_dir", type=str, required=True,
                        help="Root directory containing Genus/GCA_xxx/ structure")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for shard index files")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Number of genomes per shard (default: 1000)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Scan all genomes
    genomes = scan_genomes(args.genomes_dir)

    if not genomes:
        logging.error("No genomes found. Exiting.")
        return

    # Step 2: Genus statistics
    genus_counts = defaultdict(int)
    for g in genomes:
        genus_counts[g["genus"]] += 1

    logging.info(f"Genus distribution ({len(genus_counts)} genera):")
    for genus, count in sorted(genus_counts.items(), key=lambda x: -x[1])[:20]:
        logging.info(f"  {genus}: {count}")

    # Step 3: Build shards
    shards = build_shards(genomes, args.shard_size)
    n_shards = len(shards)
    logging.info(f"Created {n_shards} shards (shard_size={args.shard_size})")

    # Step 4: Save master index
    master_index = {
        "total_genomes": len(genomes),
        "shard_size": args.shard_size,
        "num_shards": n_shards,
        "genomes_dir": os.path.abspath(args.genomes_dir),
        "genus_counts": dict(genus_counts),
    }
    master_path = os.path.join(args.output_dir, "master_index.json")
    with open(master_path, 'w') as f:
        json.dump(master_index, f, indent=2)
    logging.info(f"Master index saved: {master_path}")

    # Step 5: Save per-shard index files
    for shard_id, shard_genomes in enumerate(shards):
        shard_path = os.path.join(args.output_dir, f"shard_{shard_id:04d}.json")
        shard_data = {
            "shard_id": shard_id,
            "num_genomes": len(shard_genomes),
            "genomes": shard_genomes,
        }
        with open(shard_path, 'w') as f:
            json.dump(shard_data, f)

    logging.info(f"Shard index files saved to {args.output_dir}/shard_XXXX.json")

    # Step 6: Save flat genome list (for quick lookups)
    genome_list_path = os.path.join(args.output_dir, "all_genomes.tsv")
    with open(genome_list_path, 'w') as f:
        f.write("genome_id\tgenus\tgff\tfaa\tfna\tshard_id\n")
        for shard_id, shard_genomes in enumerate(shards):
            for g in shard_genomes:
                f.write(f"{g['genome_id']}\t{g['genus']}\t{g['gff']}\t{g['faa']}\t{g.get('fna', '')}\t{shard_id}\n")
    logging.info(f"Flat genome list saved: {genome_list_path}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
