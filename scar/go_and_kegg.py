import os
import math
import yaml
import logging
import pandas as pd
import gseapy as gp
from scar.config import load_config
from scar.logger import setup_logger
from datetime import datetime

import pandas as pd
import os

def get_genes_from_csv(input_path, logger=None, sep=","):
    """
    Safely extract genes from the first two columns of CSV or TSV files and return a deduplicated list.
    """
    try:
        df = pd.read_csv(input_path, sep=sep)
    except Exception as e:
        if logger:
            logger.error(f"Failed to read file {input_path}: {e}")
        return []

    n_rows, n_cols = df.shape
    if logger:
        logger.info(f"Read file {os.path.basename(input_path)}: {n_rows} rows, {n_cols} columns")

    if n_cols == 0:
        if logger:
            logger.warning(f"{input_path} has no columns, skipping")
        return []

    # Take first two columns (if fewer than 2 columns, take existing columns)
    max_cols = min(2, n_cols)
    genes = pd.concat([df.iloc[:, i] for i in range(max_cols)]).dropna().drop_duplicates().tolist()

    if logger:
        logger.info(f"Extracted gene count: {len(genes)}")
    return genes


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def run_enrichment(genes, gene_sets, species, cutoff=1.0, logger=None):
    try:
        res = gp.enrichr(
            gene_list=genes,
            gene_sets=gene_sets,
            organism=species,
            cutoff=cutoff,
            outdir=None,
            verbose=False,
        )
        if logger:
            logger.debug(f"Enrichment analysis completed, gene count: {len(genes)}, database: {gene_sets}")
        return res.results if res.results is not None else pd.DataFrame()
    except Exception as e:
        if logger:
            logger.error(f"Enrichment analysis failed: {e}")
        return pd.DataFrame()

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(name="go and kegg", log_dir="Enrichment_logs",run_id = run_id)  # Call the previous log configuration function
    # 1. Load configuration parameters
    
    logger.info("Starting to load configuration")
    config = load_config("scar/config.yaml")  

    input_dir = os.path.abspath(config.get("input_dir", "../results"))
    species = config.get("species", "Human")
    batch_size =int(config.get("analysis_batch_size", 100))
    output_dir = os.path.abspath(config.get("output_dir", "enrich_results"))
    go_gene_sets = config.get("go_gene_sets", ["GO_Biological_Process_2025"])
    kegg_gene_sets = config.get("kegg_gene_sets", ["KEGG_2025_Human"])
    top_n = config.get("top_n", 10)

    
    folder_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_dir,folder_name)
    ensure_dir(output_dir)
    logger.error(f"Save directory: {output_dir}")

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        logger.warning(f"No CSV files found in directory: {input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files, starting individual analysis")

    # 3. Analyze files individually
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        logger.info(f"Analyzing file: {csv_file}")

        genes = get_genes_from_csv(input_path, logger=logger, sep=",")

        if len(genes) == 0:
            logger.warning(f"{csv_file} found no genes, skipping")
            continue

        all_go_results, all_kegg_results = [], []
        num_batches = math.ceil(len(genes) / batch_size)

        for i in range(num_batches):
            batch_genes = genes[i * batch_size:(i + 1) * batch_size]
            logger.info(f"[{csv_file}] Analyzing batch {i+1}/{num_batches}, gene count: {len(batch_genes)}")

            go_res = run_enrichment(batch_genes, go_gene_sets, species, logger=logger)
            if not go_res.empty:
                all_go_results.append(go_res)

            kegg_res = run_enrichment(batch_genes, kegg_gene_sets, species, logger=logger)
            if not kegg_res.empty:
                all_kegg_results.append(kegg_res)

        # Save GO results
        # If current file has GO enrichment analysis results
        if all_go_results:
            # Merge all batch GO results into one DataFrame
            go_df = pd.concat(all_go_results, ignore_index=True)
            # Sort by Adjusted P-value ascending and remove duplicate Terms (keep only the most significant one)
            go_df = go_df.sort_values(by="Adjusted P-value", ascending=True).drop_duplicates(subset=["Term"])
            # Take top top_n results after sorting
            # New: Filter out all rows with "Adjusted P-value" > 0.05 and save
            go_p_filtered = go_df[go_df["Adjusted P-value"] > 0.05]
            out_dir = os.path.join(output_dir, os.path.splitext(csv_file)[0])
            ensure_dir(out_dir)
            go_p_filtered.to_csv(os.path.join(out_dir, "GO_top.csv"), index=False)
            #go_top = go_df.head(top_n)
            # Create an independent subfolder for current CSV file in output directory
            # Save top top_n GO enrichment analysis results as GO_top.csv
            #go_top.to_csv(os.path.join(out_dir, "GO_top.csv"), index=False)
            # Log
            logger.info(f"[{csv_file}] GO enrichment results satisfying p>0.05 have been saved")


        # Save KEGG results
        if all_kegg_results:
            kegg_df = pd.concat(all_kegg_results, ignore_index=True)
            kegg_df = kegg_df.sort_values(by="Adjusted P-value", ascending=True).drop_duplicates(subset=["Term"])
            kegg_p_filtered = kegg_df[kegg_df["Adjusted P-value"] > 0.05]
            out_dir = os.path.join(output_dir, os.path.splitext(csv_file)[0])
            ensure_dir(out_dir)
            kegg_p_filtered.to_csv(os.path.join(out_dir, "KEGG_top.csv"), index=False)
            #kegg_top = kegg_df.head(top_n)

            #kegg_top.to_csv(os.path.join(out_dir, "KEGG_top.csv"), index=False)
            logger.info(f"[{csv_file}] KEGG enrichment results satisfying p>0.05 have been saved")

    logger.info("All file analysis completed")

if __name__ == "__main__":
    main()