import os
from scar.logger import setup_logger
from scar.config import load_config
import pandas as pd
from scipy.stats import hypergeom
from datetime import datetime

def normalize_pairs(pairs):
    return {tuple(sorted(pair)) for pair in pairs}

def main():


    cfg = load_config("scar/config.yaml")

    input_dir = cfg.get("ppi_input_dir", "../results")  # This is a directory, not a single file
    output_dir =  cfg.get("ppi_output_dir", "../ppi_results") 
    os.makedirs(output_dir, exist_ok=True)

    GENE_FILE = cfg.get("ppi_GENE_FILE","data/liver/genes.tsv")
    PPI_FILE = cfg.get("PPI_FILE", "interaction_data/ppi/PP-Decagon_ppi.csv")
    PPI_NAME_FILE = cfg.get("PPI_NAME_FILE", "interaction_data/ppi/G-SynMiner_miner-geneHUGO.tsv")
    HAS_HEADER = cfg.get("HAS_HEADER", False)
    if isinstance(HAS_HEADER, str):
        HAS_HEADER = HAS_HEADER.lower() == "true"

    last_folder = os.path.basename(os.path.normpath(input_dir))
    run_id = f"{last_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(name="ppi", log_dir="ppi_logs", run_id=run_id)


    all_gene_list = pd.read_csv(
        GENE_FILE,
        sep="\t",
        header=None,
        usecols=[1],      # Second column
        dtype=str
    )[1].unique().tolist()
    all_genes = set(all_gene_list)

    logger.info(f"Background gene universe count: {len(all_genes)}")
        # Calculate total size N (all possible gene pairs in background gene universe)
    N_genes = len(all_genes)
    N = N_genes * (N_genes - 1) // 2
    logger.info(f"Total gene pair count N: {N}")

    
    # Load PPI data
    logger.info("Starting to load PPI data")
    try:
        if HAS_HEADER:
            ppi_df = pd.read_csv(PPI_FILE)
        else:
            ppi_df = pd.read_csv(PPI_FILE, header=None, names=['protein1', 'protein2'])
        logger.info(f"PPI data loaded successfully, read {len(ppi_df)} edges")
    except Exception as e:
        logger.error(f"PPI data loading failed: {e}")
        raise e

    # 1. Read index-name mapping table
    # Read mapping table
    mapping_df = pd.read_csv(
        PPI_NAME_FILE,
        sep="\t",
        header=0,   # Your table has column names
        dtype=str   # Prevent mixed type warnings
    )
    ppi_df['protein1'] = ppi_df['protein1'].astype(str)
    ppi_df['protein2'] = ppi_df['protein2'].astype(str)

    mapping_df['entrez_id'] = mapping_df['entrez_id'].astype(str)
    # Build Entrez ID → gene symbol mapping
    entrez2name = dict(zip(mapping_df['entrez_id'], mapping_df['symbol']))

    logger.info(f"PPI protein1 examples: \n{ppi_df['protein1'].head().to_list()}")
    logger.info(f"Mapping entrez_id examples: \n{mapping_df['entrez_id'].head().to_list()}")


    # 2. Map two index columns of ppi_df to name columns
    ppi_df['protein1_name'] = ppi_df['protein1'].map(entrez2name)
    ppi_df['protein2_name'] = ppi_df['protein2'].map(entrez2name)

    # Before filtering mapping failures, can print unmapped protein2 IDs for troubleshooting
    missing1 = ppi_df[ppi_df['protein1_name'].isnull()]['protein1'].unique()
    ppi_df_filtered = ppi_df[
        (ppi_df['protein1_name'].isin(all_genes)) &
        (ppi_df['protein2_name'].isin(all_genes))].copy()
    logger.info(f"PPI edge count after filtering background genes: {len(ppi_df_filtered)}")

    logger.info(f"Unmapped protein1 IDs: {missing1.tolist()}")
    missing2 = ppi_df[ppi_df['protein2_name'].isnull()]['protein2'].unique()
    logger.info(f"Unmapped protein2 IDs: {missing2}")

    # 3. Filter out mapping failures (names are NaN)
    ppi_df = ppi_df.dropna(subset=['protein1_name', 'protein2_name'])

    # Save for checking
    ppi_df.to_csv("ppi_mapped_check.csv", index=False)
    logger.info(f"PPI mapped data saved to ppi_mapped_check.csv, remaining {len(ppi_df)} edges")
    # 4. Generate gene name pair set
    ppi_pairs = normalize_pairs(set(zip(ppi_df_filtered['protein1_name'], ppi_df_filtered['protein2_name'])))
    
    # PPI edge count M
    M = len(ppi_pairs)
    all_results = []
    # Traverse all csv files in directory
    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue
        filepath = os.path.join(input_dir, filename)
        logger.info(f"Starting to process file: {filepath}")

        your_df = pd.read_csv(filepath)
        your_pairs_raw = set(zip(your_df.iloc[:,0], your_df.iloc[:,1]))
        your_pairs = normalize_pairs(your_pairs_raw)

        k = len(your_pairs.intersection(ppi_pairs))
        logger.info(f"{filename} overlapping gene pair count: {k}")
        logger.info(f"{filename} overlap ratio: {k / len(your_pairs):.4f}")

        # --- Similarly filter your gene pairs to ensure only background gene pairs ---
        #your_pairs_filtered = set(pair for pair in your_pairs if pair[0] in all_genes and pair[1] in all_genes)
    
        n = len(your_pairs) # Gene pair count in current file

        pval = hypergeom.sf(k-1, N, M, n)
        logger.info(f"{filename} hypergeometric distribution p-value: {pval:.3e}")
        # If you don't have other multiple results, no need to sort and save here, just save one pval
        # Here example saves pval to csv (overwrite or new filename)
        all_results.append({
            "cell_type": filename,
            "overlap": k,
            "pair_count": n,
            "p_value": pval,
            # You can add more fields
        })
    # Save uniformly after traversal ends
    df_results = pd.DataFrame(all_results)
    last_folder = os.path.basename(os.path.normpath(input_dir))
    output_file = os.path.join(output_dir, f"ppi_enrichment_summary_{last_folder}.csv")

    df_results.to_csv(output_file, index=False)
    logger.info(f"All results saved to {output_file}")

if __name__ == "__main__":
    main()
