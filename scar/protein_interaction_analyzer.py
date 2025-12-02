import os
import pandas as pd
from scipy.stats import hypergeom
from datetime import datetime
from scar.logging_system import setup_logger
from scar.configuration_manager import load_config
from typing import Set, Tuple, Dict, Any, List, Callable


# ----------------------------------------------------------------------------
# 1. Configuration & Environment
# ----------------------------------------------------------------------------

def setup_environment(config_path: str) -> Dict[str, Any]:
    """Load config, create directories, and return paths and logger."""
    cfg = load_config(config_path)
    input_dir = cfg.get("ppi_input_dir", "../results")
    last_folder = os.path.basename(os.path.normpath(input_dir))
    output_dir_base = cfg.get("ppi_output_dir", "../ppi_results")
    output_dir = os.path.join(output_dir_base, last_folder)
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"{last_folder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logger(name="ppi", log_dir="ppi_logs", run_id=run_id)
    return {
        "config": cfg,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "last_folder": last_folder,
        "logger": logger
    }


def normalize_pairs(pairs: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Normalize gene-pair tuples to ensure consistent ordering (pure function)."""
    return {tuple(sorted(pair)) for pair in pairs}


# ----------------------------------------------------------------------------
# 2. Data Loading & Preprocessing (Pure Functions)
# ----------------------------------------------------------------------------

def load_gene_set(filepath: str) -> Set[str]:
    """Load background gene universe from file."""
    genes = pd.read_csv(
        filepath, sep="\t", header=None, usecols=[1], dtype=str
    )[1].unique().tolist()
    return set(genes)


def load_ppi_data(filepath: str, has_header: bool) -> pd.DataFrame:
    """Load raw PPI data."""
    try:
        return pd.read_csv(filepath, header=0 if has_header else None,
                           names=None if has_header else ['protein1', 'protein2'])
    except Exception as e:
        print(f"Failed to load PPI data: {e}")
        raise


def get_entrez_to_name_mapping(filepath: str) -> Dict[str, str]:
    """Create a mapping from Entrez ID to gene symbol."""
    mapping_df = pd.read_csv(filepath, sep="\t", header=0, dtype=str)
    return dict(zip(mapping_df['entrez_id'], mapping_df['symbol']))


def map_and_filter_ppi(
        ppi_df: pd.DataFrame,
        entrez2name: Dict[str, str],
        background_genes: Set[str]
) -> pd.DataFrame:
    """Map PPI IDs to gene names and filter by background gene set (pure)."""
    mapped_df = ppi_df.assign(
        protein1_name=lambda df: df['protein1'].astype(str).map(entrez2name),
        protein2_name=lambda df: df['protein2'].astype(str).map(entrez2name)
    ).dropna(subset=['protein1_name', 'protein2_name'])

    return mapped_df[
        (mapped_df['protein1_name'].isin(background_genes)) &
        (mapped_df['protein2_name'].isin(background_genes))
        ]


# ----------------------------------------------------------------------------
# 3. Core Analysis Logic (Pure Functions)
# ----------------------------------------------------------------------------

def calculate_hypergeometric_pvalue(k: int, M: int, n: int, N: int) -> float:
    """Compute p-value from the hypergeometric distribution (pure)."""
    return hypergeom.sf(k - 1, N, M, n)


def perform_enrichment_analysis(
        cell_type_pairs: Set[Tuple[str, str]],
        background_ppi_pairs: Set[Tuple[str, str]],
        hypergeom_params: Dict[str, int]
) -> Dict[str, Any]:
    """Perform enrichment analysis for a single cell type's gene pairs (pure)."""
    tp_pairs = cell_type_pairs.intersection(background_ppi_pairs)
    fp_pairs = cell_type_pairs - background_ppi_pairs

    TP = len(tp_pairs)
    FP = len(fp_pairs)
    precision = TP / len(cell_type_pairs) if cell_type_pairs else 0.0

    p_value = calculate_hypergeometric_pvalue(
        k=TP,
        M=hypergeom_params["M"],
        n=len(cell_type_pairs),
        N=hypergeom_params["N"]
    )

    return {
        "tp_pairs": tp_pairs,
        "fp_pairs": fp_pairs,
        "overlap": TP,
        "pair_count": len(cell_type_pairs),
        "p_value": p_value,
        "TP": TP,
        "FP": FP,
        "Precision": precision,
        "Total_PPI_in_background": hypergeom_params["M"]
    }


# ----------------------------------------------------------------------------
# 4. Side-Effect Handling
# ----------------------------------------------------------------------------

def save_analysis_details(
        cell_type_name: str,
        analysis_result: Dict[str, Any],
        background_ppi_pairs: Set[Tuple[str, str]],
        output_dir: str
):
    """Save detailed results for a cell type (TP, FP, etc.)."""
    cell_output_dir = os.path.join(output_dir, cell_type_name)
    os.makedirs(cell_output_dir, exist_ok=True)

    pd.DataFrame(list(background_ppi_pairs), columns=["gene1", "gene2"]) \
        .to_csv(os.path.join(cell_output_dir, "background_ppi_network.csv"), index=False, header=False)

    pd.DataFrame(list(analysis_result["tp_pairs"]), columns=['Gene1', 'Gene2']) \
        .to_csv(os.path.join(cell_output_dir, f"TP_{cell_type_name}.csv"), index=False)

    # Note: If the FP set is too large, saving it may cause memory issues. If errors reoccur, consider commenting out the line below.
    pd.DataFrame(list(analysis_result["fp_pairs"]), columns=['Gene1', 'Gene2']) \
        .to_csv(os.path.join(cell_output_dir, f"FP_{cell_type_name}.csv"), index=False)


def save_summary_report(summary_df: pd.DataFrame, output_path: str):
    """Save the final summary report."""
    summary_df.to_csv(output_path, index=False)


# ----------------------------------------------------------------------------
# 5. Main Workflow (Functional Style)
# ----------------------------------------------------------------------------

# >>>>>>>>>>>>>>>>>>>>>>>>> Main changes start here <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def process_cell_type_file(
        filepath: str,
        analysis_function: Callable[[Set[Tuple[str, str]]], Dict[str, Any]],
        logger
) -> Dict[str, Any]:
    """Load a single cell-type file and normalize pairs on-the-fly to save memory."""
    logger.info(f"Processing file: {os.path.basename(filepath)}")
    cell_type_name = os.path.basename(filepath).replace(".csv", "")

    # Create a single set and add normalized pairs while reading
    your_normalized_pairs = set()
    CHUNKSIZE = 1_000_000  # Read 1,000,000 rows per chunk

    try:
        with pd.read_csv(filepath, chunksize=CHUNKSIZE, header=0, low_memory=True) as reader:
            for chunk in reader:
                # Iterate over each row in the chunk, normalize immediately, and add to the set
                for pair in zip(chunk.iloc[:, 0], chunk.iloc[:, 1]):
                    # Ensure gene names are not empty or invalid
                    if pd.notna(pair[0]) and pd.notna(pair[1]):
                        your_normalized_pairs.add(tuple(sorted(pair)))
    except Exception as e:
        logger.error(f"Failed to read or process file {filepath}: {e}")
        return {}  # Return empty dict to avoid downstream errors

    # No need to call normalize_pairs(); pairs are already normalized
    analysis_result = analysis_function(your_normalized_pairs)

    analysis_result["cell_type"] = os.path.basename(filepath)
    analysis_result["cell_type_name"] = cell_type_name
    return analysis_result


# >>>>>>>>>>>>>>>>>>>>>>>>> Changes end <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def main():
    env = setup_environment("scar/project_settings.yaml")
    logger = env["logger"]
    logger.info("--- PPI enrichment analysis started ---")

    cfg = env["config"]
    all_genes = load_gene_set(cfg.get(" .,", "data/liver/genes.tsv"))
    entrez_map = get_entrez_to_name_mapping(
        cfg.get("PPI_NAME_FILE", "interaction_data/ppi/G-SynMiner_miner-geneHUGO.tsv")
    )

    ppi_raw_df = load_ppi_data(
        cfg.get("PPI_FILE", "interaction_data/ppi/PP-Decagon_ppi.csv"),
        cfg.get("HAS_HEADER", False)
    )

    background_ppi_df = map_and_filter_ppi(ppi_raw_df, entrez_map, all_genes)
    background_ppi_pairs = normalize_pairs(
        set(zip(background_ppi_df['protein1_name'], background_ppi_df['protein2_name']))
    )

    logger.info(f"Background gene universe size: {len(all_genes)}")
    logger.info(f"Filtered background PPI edges (all genes): {len(background_ppi_pairs):,}")

    csv_files = [os.path.join(env["input_dir"], f) for f in os.listdir(env["input_dir"]) if f.endswith(".csv")]
    all_predicted_genes = set()

    CHUNKSIZE_GENE_COLLECTION = 1_000_000
    for file in csv_files:
        logger.info(f"Collecting genes from {os.path.basename(file)}...")
        try:
            with pd.read_csv(file, chunksize=CHUNKSIZE_GENE_COLLECTION, header=0, low_memory=True) as reader:
                for chunk in reader:
                    all_predicted_genes.update(set(chunk.iloc[:, 0].dropna()))
                    all_predicted_genes.update(set(chunk.iloc[:, 1].dropna()))
        except Exception as e:
            logger.error(f"Failed reading file {file} during gene collection: {e}")
            continue

    valid_analysis_genes = all_predicted_genes & all_genes
    logger.info(f"Total unique genes across all input files: {len(all_predicted_genes):,}")
    logger.info(f"Valid analysis genes after intersecting with background: {len(valid_analysis_genes):,}")

    N_genes = len(valid_analysis_genes)
    N_total_pairs = N_genes * (N_genes - 1) // 2

    filtered_ppi_pairs = {
        pair for pair in background_ppi_pairs
        if pair[0] in valid_analysis_genes and pair[1] in valid_analysis_genes
    }

    background_ppi_pairs = filtered_ppi_pairs

    hypergeom_params = {
        "N": N_total_pairs,
        "M": len(filtered_ppi_pairs)
    }
    logger.info(f"[Restricted background] Total number of gene pairs N: {hypergeom_params['N']:,}")
    logger.info(f"[Restricted background] Background PPI gene pairs M: {hypergeom_params['M']:,}")
    background_precision = hypergeom_params["M"] / hypergeom_params["N"] if hypergeom_params["N"] > 0 else 0
    logger.info(f"[Restricted background] Inherent Precision (M/N): {background_precision:.6f}")

    enrichment_analyzer = lambda cell_pairs: perform_enrichment_analysis(
        cell_pairs, background_ppi_pairs, hypergeom_params
    )

    all_results = [process_cell_type_file(file, enrichment_analyzer, logger) for file in csv_files]
    all_results = [r for r in all_results if r]

    for result in all_results:
        logger.info(
            f"{result['cell_type_name']:<30} - TP: {result['TP']:,}, Precision: {result['Precision']:.4f}, p-value: {result['p_value']:.3e}"
        )
        # Wrap saving details in try-except to prevent a single large file from breaking the workflow
        try:
            save_analysis_details(result['cell_type_name'], result, background_ppi_pairs, env["output_dir"])
        except MemoryError:
            logger.warning(
                f"Memory error when saving details for {result['cell_type_name']}. FP file may be too large. Skipping details...")
        except Exception as e:
            logger.error(f"Unknown error when saving details for {result['cell_type_name']}: {e}")

    if all_results:
        summary_columns = [
            "cell_type", "overlap", "pair_count", "p_value", "TP", "FP",
            "Precision", "Total_PPI_in_background"
        ]
        summary_df = pd.DataFrame(all_results)[summary_columns]
        output_file = os.path.join(env["output_dir"], f"ppi_enrichment_summary_{env['last_folder']}.csv")
        save_summary_report(summary_df, output_file)
        logger.info(f"All results summarized to: {output_file}")
    else:
        logger.warning("No files were processed successfully; cannot generate summary report.")

    logger.info("--- PPI enrichment analysis completed ---")


if __name__ == "__main__":
    main()