"""
SAGE v4: Shard-based COG & Mutation Annotation Extraction
==========================================================
Processes a single shard of genomes, extracting:
  1. COG functional categories (from GFF product fields)
  2. Mutation/resistance annotations (from GFF product fields)
  3. Gene physical order metadata (contig, strand, replicon, distance)

Output per shard:
  - shard_XXXX_metadata.jsonl  (one JSON line per genome)

Can run in parallel: one process per shard.

Usage:
    python extract_annotations_shard.py \
        --shard_file /path/to/shard_index/shard_0000.json \
        --output_dir /path/to/annotations_sharded
"""

import os
import re
import json
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# COG keyword rules (reused from v3 extract_cog_from_gff.py)
# ============================================================================
COG_KEYWORD_RULES = [
    (r'beta-lactamase|carbapenemase|metallo-beta-lactamase|MBL fold metallo-hydrolase', 'V'),
    (r'efflux|multidrug|drug resistance|antimicrobial', 'V'),
    (r'restriction endonuclease|restriction-modification|methylase.*restriction', 'V'),
    (r'CRISPR|Cas[0-9]|toxin-antitoxin|abortive infection', 'V'),
    (r'chloramphenicol acetyltransferase|aminoglycoside.*transferase', 'V'),
    (r'tetracycline resistance|macrolide.*resistance|vancomycin resistance', 'V'),
    (r'ribosom|tRNA.*ligase|tRNA.*synthetase|translation.*factor|elongation factor|initiation factor', 'J'),
    (r'30S ribosomal|50S ribosomal|16S rRNA|23S rRNA|rRNA methyltransferase', 'J'),
    (r'transcriptional regulator|transcription factor|sigma factor|RNA polymerase', 'K'),
    (r'LysR family|TetR.*family|AraC family|MarR family|GntR family|IclR family', 'K'),
    (r'helix-turn-helix.*regulator|HTH.*regulator|DNA-binding.*regulator', 'K'),
    (r'DNA polymerase|DNA helicase|DNA ligase|DNA gyrase|topoisomerase', 'L'),
    (r'recombinase|integrase|transposase|resolvase|reverse transcriptase', 'L'),
    (r'DNA repair|DNA mismatch|uvrA|uvrB|uvrC|mutS|mutL|recA|recB', 'L'),
    (r'IS[0-9]|insertion sequence|IS element', 'L'),
    (r'cell division|FtsZ|FtsA|FtsI|FtsW|FtsQ|FtsK|FtsB|FtsL|FtsN|MinC|MinD|MinE', 'D'),
    (r'chromosome partitioning|chromosome segregation|ParA|ParB|SMC', 'D'),
    (r'signal transduction|two-component|sensor histidine kinase|response regulator', 'T'),
    (r'diguanylate cyclase|phosphodiesterase|c-di-GMP|GGDEF|EAL domain', 'T'),
    (r'chemotaxis|CheA|CheB|CheR|CheW|CheY|CheZ', 'T'),
    (r'lipopolysaccharide|peptidoglycan|murein|MurA|MurB|MurC|MurD|MurE|MurF|MurG', 'M'),
    (r'outer membrane protein|porin|OmpA|OmpC|OmpF|OmpW', 'M'),
    (r'lipid A|LPS|O-antigen|capsule|capsular|polysaccharide biosynthesis', 'M'),
    (r'penicillin-binding protein|PBP|transpeptidase', 'M'),
    (r'flagell|pilin|pilus|fimbr|type IV pil|motility|chemotaxis protein', 'N'),
    (r'type I secretion|type II secretion|type III secretion|type IV secretion|type VI secretion', 'U'),
    (r'Sec-dependent|Tat pathway|signal peptidase|preprotein translocase', 'U'),
    (r'Sec[ABDEFY]|SecG|TatA|TatB|TatC', 'U'),
    (r'chaperone|GroEL|GroES|DnaK|DnaJ|GrpE|ClpA|ClpB|ClpP|ClpX|HslU|HslV', 'O'),
    (r'protease|peptidase|proteasome|Lon protease|FtsH', 'O'),
    (r'thioredoxin|glutaredoxin|peroxidase|catalase|superoxide dismutase', 'O'),
    (r'cytochrome|NADH.*dehydrogenase|NADH.*oxidoreductase|ATP synthase|electron transfer', 'C'),
    (r'succinate dehydrogenase|fumarate reductase|nitrate reductase|nitrite reductase', 'C'),
    (r'ferredoxin|flavodoxin|hydrogenase|oxidoreductase.*NAD|dehydrogenase.*NAD', 'C'),
    (r'citrate synthase|aconitase|isocitrate dehydrogenase|oxoglutarate dehydrogenase', 'C'),
    (r'glycosyl|glucosidase|galactosidase|mannosidase|xylosidase|amylase', 'G'),
    (r'phosphotransferase system|PTS|sugar.*transport|sugar.*permease', 'G'),
    (r'glycolysis|gluconeogenesis|pentose phosphate|fructose.*bisphosphate', 'G'),
    (r'amino acid.*transport|amino acid.*permease|amino acid.*dehydrogenase', 'E'),
    (r'aminotransferase|transaminase|deaminase|amino acid.*biosynthesis', 'E'),
    (r'glutamate|glutamine|aspartate|asparagine|lysine.*biosynthesis', 'E'),
    (r'tryptophan|tyrosine|phenylalanine|histidine.*biosynthesis', 'E'),
    (r'nucleotide|purine|pyrimidine|nucleoside|thymidylate|dihydrofolate reductase', 'F'),
    (r'ribonucleotide reductase|nucleoside kinase|adenylate kinase', 'F'),
    (r'coenzyme|cofactor|biotin|thiamin|riboflavin|folate|pantothenate|pyridoxal', 'H'),
    (r'NAD biosynthesis|FAD biosynthesis|CoA biosynthesis|heme biosynthesis', 'H'),
    (r'porphyrin|corrin|menaquinone|ubiquinone', 'H'),
    (r'fatty acid|lipase|acyl-CoA|acyltransferase|phospholipase|lipid.*biosynthesis', 'I'),
    (r'beta-oxidation|enoyl.*reductase|3-ketoacyl|acetyl-CoA carboxylase', 'I'),
    (r'iron.*transport|ferric|ferrous|siderophore|TonB-dependent receptor', 'P'),
    (r'zinc.*transport|copper.*transport|manganese.*transport|magnesium.*transport', 'P'),
    (r'sulfate.*transport|phosphate.*transport|potassium.*transport|sodium.*transport', 'P'),
    (r'ABC transporter.*metal|cation.*transport|anion.*transport', 'P'),
    (r'polyketide|non-ribosomal peptide|NRPS|PKS|siderophore biosynthesis', 'Q'),
    (r'terpene|phenazine|phenol|aromatic.*degradation|catechol', 'Q'),
    (r'MFS transporter|ABC transporter|MATE.*transporter|RND.*transporter', 'G'),
    (r'transporter|permease|symporter|antiporter', 'R'),
    (r'methyltransferase|acetyltransferase|acyltransferase', 'R'),
    (r'hydrolase|alpha/beta.*hydrolase|esterase', 'R'),
    (r'oxidoreductase|dehydrogenase|reductase|oxygenase|monooxygenase|dioxygenase', 'C'),
    (r'kinase|phosphatase', 'T'),
    (r'synthase|synthetase|ligase', 'R'),
    (r'hypothetical protein|uncharacterized protein|DUF[0-9]+|unknown function', 'S'),
    (r'domain.*containing protein|domain of unknown function', 'S'),
]

COG_RULES_COMPILED = [(re.compile(pattern, re.IGNORECASE), cog) for pattern, cog in COG_KEYWORD_RULES]

# Mutation patterns
HIGH_CONF_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'resistance protein', r'resistant\b', r'antibiotic resistance',
        r'drug resistance', r'multidrug resistance', r'tetracycline resistance',
        r'chloramphenicol resistance', r'macrolide resistance', r'vancomycin resistance',
        r'beta-lactam resistance', r'aminoglycoside resistance', r'quinolone resistance',
    ]
]

MEDIUM_CONF_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'beta-lactamase', r'carbapenemase', r'metallo-beta-lactamase',
        r'ESBL', r'OXA-\d+', r'KPC-\d+', r'NDM-\d+', r'VIM-\d+', r'IMP-\d+', r'CTX-M',
        r'efflux pump', r'efflux transporter', r'multidrug efflux',
        r'MFS.*efflux', r'RND.*efflux', r'AcrAB', r'MexAB',
        r'aminoglycoside.*phosphotransferase', r'aminoglycoside.*acetyltransferase',
        r'chloramphenicol acetyltransferase',
        r'tetracycline.*efflux', r'tet\([A-Z]\)', r'ribosomal protection protein',
        r'16S rRNA methyltransferase', r'23S rRNA methyltransferase',
        r'penicillin-binding protein', r'PBP\d',
        r'Qnr\w+', r'MCR-\d+', r'phosphoethanolamine.*transferase',
        r'VanA|VanB|VanC|VanD', r'dihydropteroate synthase',
        r'fosfomycin.*thiol transferase', r'FosA|FosB|FosC|FosX',
    ]
]


def product_to_cog(product_str):
    if not product_str:
        return '-'
    for regex, cog in COG_RULES_COMPILED:
        if regex.search(product_str):
            return cog
    return '-'


def classify_resistance(product_str):
    if not product_str:
        return 1
    for pattern in HIGH_CONF_PATTERNS:
        if pattern.search(product_str):
            return 3
    for pattern in MEDIUM_CONF_PATTERNS:
        if pattern.search(product_str):
            return 2
    return 1


def parse_single_genome(gff_path, genome_id, genus):
    """
    Parse a single GFF file and extract all gene-level metadata.
    
    Returns:
        dict with genome_id, genus, and list of contigs.
        Each contig contains sorted genes with all annotations.
    """
    contigs = {}
    seq_to_replicon = {}

    try:
        with open(gff_path, 'r') as f:
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

                if feature_type != 'CDS':
                    continue

                start_pos = int(parts[3])
                end_pos = int(parts[4])
                strand_val = parts[6] if parts[6] in ['+', '-'] else '+'

                # Extract gene IDs
                locus_match = re.search(r'locus_tag=([^;]+)', attributes)
                id_match = re.search(r'ID=([^;]+)', attributes)
                protein_match = re.search(r'protein_id=([^;]+)', attributes)

                gene_id_local = None
                if locus_match:
                    gene_id_local = locus_match.group(1)
                elif id_match:
                    gene_id_local = id_match.group(1)

                query_id = protein_match.group(1) if protein_match else gene_id_local

                if not gene_id_local:
                    continue

                # Extract product
                product_match = re.search(r'product=([^;]+)', attributes)
                product = product_match.group(1) if product_match else ''
                product = product.replace('%3B', ';').replace('%2C', ',').replace('%25', '%')

                # COG and mutation classification
                cog = product_to_cog(product)
                mutation = classify_resistance(product)
                replicon_type = seq_to_replicon.get(seq_id, 'unknown')

                if seq_id not in contigs:
                    contigs[seq_id] = []

                contigs[seq_id].append({
                    'gene_id': gene_id_local,
                    'query_id': query_id or gene_id_local,
                    'start': start_pos,
                    'end': end_pos,
                    'strand': strand_val,
                    'replicon': replicon_type,
                    'cog': cog,
                    'mutation': mutation,
                })

    except Exception as e:
        logging.error(f"Error parsing {gff_path}: {e}")
        return None

    # Sort genes within each contig by start position
    contig_list = []
    for seq_id, genes in contigs.items():
        genes.sort(key=lambda x: x['start'])
        # Compute inter-gene distances
        for i, g in enumerate(genes):
            if i == 0:
                g['distance'] = 0
            else:
                g['distance'] = g['start'] - genes[i - 1]['end']
        contig_list.append({
            'seq_id': seq_id,
            'replicon': seq_to_replicon.get(seq_id, 'unknown'),
            'genes': genes,
        })

    return {
        'genome_id': genome_id,
        'genus': genus,
        'num_contigs': len(contig_list),
        'num_genes': sum(len(c['genes']) for c in contig_list),
        'contigs': contig_list,
    }


def process_shard(shard_file, output_dir):
    """Process all genomes in a single shard."""
    os.makedirs(output_dir, exist_ok=True)

    with open(shard_file, 'r') as f:
        shard_data = json.load(f)

    shard_id = shard_data['shard_id']
    genomes = shard_data['genomes']

    logging.info(f"Processing shard {shard_id}: {len(genomes)} genomes")

    output_path = os.path.join(output_dir, f"shard_{shard_id:04d}_metadata.jsonl")
    total_genes = 0
    processed = 0

    with open(output_path, 'w') as out_f:
        for genome_info in genomes:
            genome_id = genome_info['genome_id']
            genus = genome_info['genus']
            gff_path = genome_info['gff']

            result = parse_single_genome(gff_path, genome_id, genus)
            if result is None:
                continue

            out_f.write(json.dumps(result) + '\n')
            total_genes += result['num_genes']
            processed += 1

    logging.info(f"Shard {shard_id}: processed {processed}/{len(genomes)} genomes, "
                 f"{total_genes} total genes → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAGE v4: Shard-based annotation extraction")
    parser.add_argument("--shard_file", type=str, required=True,
                        help="Path to shard JSON file (from generate_shard_index.py)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for annotation JSONL files")
    args = parser.parse_args()

    process_shard(args.shard_file, args.output_dir)


if __name__ == "__main__":
    main()
