import os
import pandas as pd
import gseapy as gp
from scar.configuration_manager import load_config
from scar.logging_system import setup_logger
from datetime import datetime


# --- Modified function ---
def get_genes_from_csv(input_path, logger=None, sep=","):
    """
    Safely extract genes from a CSV or TSV file.
    - If the file has two or more columns, merge the first two columns.
    - If the file has only one column, use that column directly.
    - Logs actions declaratively and returns a de-duplicated gene list.
    """
    try:
        df = pd.read_csv(input_path, sep=sep, header=None)  # Use header=None to handle files without headers
    except Exception as e:
        if logger:
            logger.error(f"Failed to read file {input_path}: {e}")
        return []

    n_rows, n_cols = df.shape
    if logger:
        logger.info(f"Read file {os.path.basename(input_path)}: {n_rows} rows, {n_cols} columns")

    if n_cols == 0:
        if logger:
            logger.warning(f"{input_path} is an empty file with no data columns. Skipping.")
        return []

    # --- Declarative processing logic ---
    if n_cols >= 2:
        # Intention: merge the first two columns
        if logger:
            # Use column indices (0 and 1) since headers may be absent
            logger.info(f"File has {n_cols} columns. Merging the first two columns (0 and 1).")

        # Step 1: extract first two columns
        col1 = df.iloc[:, 0]
        col2 = df.iloc[:, 1]

        # Step 2: merge into a unified Series
        combined_series = pd.concat([col1, col2], ignore_index=True)

    else:  # when n_cols == 1
        # Intention: use the single column
        if logger:
            logger.info(f"File has only 1 column; using it directly.")

        # Step 1: use the first column directly
        combined_series = df.iloc[:, 0]

    # Step 3: clean data by removing NA/NaN and duplicates
    unique_genes_series = combined_series.dropna().drop_duplicates()

    # Step 4: convert the cleaned Series to a list
    genes = unique_genes_series.tolist()

    if logger:
        logger.info(f"Extracted and de-duplicated genes from '{os.path.basename(input_path)}': {len(genes)} total")

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
            logger.debug(f"Enrichment finished, gene count: {len(genes)}, databases: {gene_sets}")
        return res.results if res.results is not None else pd.DataFrame()
    except Exception as e:
        if logger:
            logger.error(f"Enrichment failed: {e}")
        return pd.DataFrame()


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(name="go and kegg", log_dir="Enrichment_logs", run_id=run_id)

    logger.info("Starting to load configuration")
    config = load_config("scar/project_settings.yaml")

    input_dir = os.path.abspath(config.get("input_dir", "../results"))
    species = config.get("species", "Human")
    # Removed batch_size; no longer batching
    output_dir = os.path.abspath(config.get("com_output_dir", "enrich_results"))
    go_gene_sets = config.get("go_gene_sets", ["GO_Biological_Process_2025"])
    kegg_gene_sets = config.get("kegg_gene_sets", ["KEGG_2025_Human"])
    top_n = config.get("top_n", 10)  # top_n still used to filter final results; logic adjusted

    folder_name = os.path.basename(input_dir)
    output_dir = os.path.join(output_dir, folder_name)
    ensure_dir(output_dir)
    logger.info(f"Output directory: {output_dir}")

    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        logger.warning(f"No CSV files found in directory: {input_dir}")
        return

    logger.info(f"Found {len(csv_files)} CSV files. Starting analysis.")

    # Analyze files one by one
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        logger.info(f"--- Analyzing file: {csv_file} ---")

        # 1. Get all genes at once
        genes = get_genes_from_csv(input_path, logger=logger, sep=",")

        if not genes:
            logger.warning(f"{csv_file} contains no genes. Skipping.")
            continue

        logger.info(f"[{csv_file}] Extracted {len(genes)} genes. Running overall enrichment.")

        # 2. GO enrichment
        logger.info(f"[{csv_file}] Starting GO enrichment...")
        go_results = run_enrichment(genes, go_gene_sets, species, logger=logger)

        # 3. Save GO results
        if not go_results.empty:
            # Results are complete; no merging needed
            go_df = go_results.sort_values(by="Adjusted P-value", ascending=True)

            # Filter significant results with Adjusted P-value < 0.05
            go_significant = go_df[go_df["Adjusted P-value"] < 0.05]

            out_dir = os.path.join(output_dir, os.path.splitext(csv_file)[0])  # Ensure directory exists
            ensure_dir(out_dir)

            # Save all significant results
            if not go_significant.empty:
                go_significant.to_csv(os.path.join(out_dir, "GO_significant_results.csv"), index=False)
                logger.info(
                    f"[{csv_file}] GO enrichment finished; {len(go_significant)} significant results (Adj P-val < 0.05) saved.")

                # Optionally save top N results
                go_top = go_df.head(top_n)
                go_top.to_csv(os.path.join(out_dir, "GO_top_results.csv"), index=False)
                logger.info(f"[{csv_file}] Also saved top {top_n} GO results.")

            else:
                logger.warning(f"[{csv_file}] No significant GO results found (Adj P-val < 0.05).")
        else:
            logger.warning(f"[{csv_file}] GO enrichment returned no results.")

        # 4. KEGG enrichment
        logger.info(f"[{csv_file}] Starting KEGG enrichment...")
        kegg_results = run_enrichment(genes, kegg_gene_sets, species, logger=logger)

        # 5. Save KEGG results
        if not kegg_results.empty:
            kegg_df = kegg_results.sort_values(by="Adjusted P-value", ascending=True)

            kegg_significant = kegg_df[kegg_df["Adjusted P-value"] < 0.05]

            out_dir = os.path.join(output_dir, os.path.splitext(csv_file)[0])  # Ensure directory exists
            ensure_dir(out_dir)

            if not kegg_significant.empty:
                kegg_significant.to_csv(os.path.join(out_dir, "KEGG_significant_results.csv"), index=False)
                logger.info(
                    f"[{csv_file}] KEGG enrichment finished; {len(kegg_significant)} significant results (Adj P-val < 0.05) saved.")

                kegg_top = kegg_df.head(top_n)
                kegg_top.to_csv(os.path.join(out_dir, "KEGG_top_results.csv"), index=False)
                logger.info(f"[{csv_file}] Also saved top {top_n} KEGG results.")
            else:
                logger.warning(f"[{csv_file}] No significant KEGG results found (Adj P-val < 0.05).")
        else:
            logger.warning(f"[{csv_file}] KEGG enrichment returned no results.")

    logger.info("--- All files analyzed ---")


if __name__ == "__main__":
    main()