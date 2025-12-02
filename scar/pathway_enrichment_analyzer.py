import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from datetime import datetime
from scar.configuration_manager import load_config
from scar.logging_system import setup_logger

# -------------------------------
# Logging configuration
# -------------------------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(name="reactome_comparison", log_dir="reactome_logs", run_id=run_id)
logger.info("Logger initialized")

# -------------------------------
# Load configuration
# -------------------------------
logger.info("Starting to load configuration")
try:
    config = load_config("scar/project_settings.yaml")
except FileNotFoundError:
    logger.error("Configuration file scar/project_settings.yaml not found. Please check the path.")
    # Provide a fallback config if needed (not implemented here)


base_dir = config["reactome_input_dir"]
pathway_file = config["reactome_pathway_file"]
metrics = config.get("metrics", ["support", "confidence", "leverage", "conviction", "lift"])
logger.info(f"Configuration loaded. Metrics: {metrics}")

# -------------------------------
# Output root directory
# -------------------------------
save_root = config.get("reactome_output_dir", r"D:\software\code\code\scar2\reactome_results")
os.makedirs(save_root, exist_ok=True)
logger.info(f"Results will be saved to: {save_root}")

# -------------------------------
# Function definition
# -------------------------------
def process_celltype(organism_name, celltype_name, ar_file, pathway_df_full):
    logger.info(f"Processing {organism_name}/{celltype_name}")
    save_dir = os.path.join(save_root, organism_name, celltype_name)
    os.makedirs(save_dir, exist_ok=True)

    group_data_save_dir = os.path.join(save_dir, "group_data")
    os.makedirs(group_data_save_dir, exist_ok=True)

    # Read AR data
    try:
        ar_df = pd.read_csv(ar_file)
        if 'Gene_1' in ar_df.columns and 'Gene_2' in ar_df.columns:
            ar_df = ar_df.rename(columns={'Gene_1': 'Gene1', 'Gene_2': 'Gene2'})
        ar_df = ar_df[['Gene1', 'Gene2'] + metrics]
    except (KeyError, FileNotFoundError) as e:
        logger.error(f"Failed to read or process file {ar_file}: {e}. Ensure the file exists and contains 'Gene1' and 'Gene2' columns.")
        return

    # Intersect with pathway table
    merged_df = pd.merge(ar_df, pathway_df_full, on=['Gene1', 'Gene2'], how='inner')
    if merged_df.empty:
        logger.warning(f"No intersection between {ar_file} and pathway table; skipping this cell type.")
        return

    merged_filename = f"{celltype_name}_merged_table.csv"
    merged_df.to_csv(os.path.join(save_dir, merged_filename), index=False)
    logger.info(f"Merged table saved: {merged_filename}")

    # Data cleaning
    for metric in metrics:
        merged_df[metric] = merged_df[metric].replace([np.inf, -np.inf], 100)
        merged_df.loc[merged_df[metric] > 100, metric] = 100

    direction_mapping = {
        '->': 'activation', '<->': 'binding', '-|': 'inhibition',
        '<-': 'upstream_activation', '<-|': 'inhibited_by',
        '|->': 'inhibition_activation', '|-': 'inhibition',
        '-': 'undirected_interaction'
    }

    p_values_all = {}
    # Main dict collects means for group1 and group2
    mean_values_for_celltype = {}

    unique_directions = merged_df['Direction'].unique()

    # Outer loop: iterate over directions
    for dir_type in unique_directions:
        p_values_all[dir_type] = {}
        
        safe_name = direction_mapping.get(str(dir_type), str(dir_type))
        safe_name = safe_name.replace('<', '_').replace('>', '_').replace('|', '_')

        # Create Series for current pathway group1 and group2
        g1_means_series = pd.Series(index=metrics, dtype=float)
        g2_means_series = pd.Series(index=metrics, dtype=float)

        # Inner loop: iterate over metrics
        for metric in metrics:
            # Split data
            group1 = merged_df[merged_df['Direction'] == dir_type][metric].dropna()
            group2 = merged_df[merged_df['Direction'] != dir_type][metric].dropna()
            
            # Compute and store group1 and group2 means
            g1_means_series[metric] = group1.mean()
            g2_means_series[metric] = group2.mean()

            # Save group1 and group2 data
            group1.to_csv(os.path.join(group_data_save_dir, f"{celltype_name}_{safe_name}_{metric}_group1.csv"), index=False, header=[metric])
            group2.to_csv(os.path.join(group_data_save_dir, f"{celltype_name}_{safe_name}_{metric}_group2.csv"), index=False, header=[metric])

            # Differential test
            if len(group1) > 1 and len(group2) > 1:
                stat, p = ttest_ind(group1, group2, equal_var=False)
                p_values_all[dir_type][metric] = p
            else:
                p_values_all[dir_type][metric] = None
        
        # Store g1/g2 mean Series into main dict using composite keys
        mean_values_for_celltype[f"{safe_name}_group1"] = g1_means_series
        mean_values_for_celltype[f"{safe_name}_group2"] = g2_means_series
        logger.info(f"Computed p-values and G1/G2 means for {celltype_name} {safe_name} pathway.")


    # 1. Save p-value summary table (kept)
    p_df_all = pd.DataFrame(p_values_all)
    full_p_file = os.path.join(save_dir, f"{celltype_name}_all_metrics_pathways_P_value.csv")
    p_df_all.to_csv(full_p_file)
    logger.info(f"P-value table for all metrics saved: {full_p_file}")

    # 2. Generate and save group1/group2 mean comparison table
    if not mean_values_for_celltype:
        logger.warning(f"No data available to aggregate means for {celltype_name}.")
    else:
        mean_comparison_df = pd.DataFrame(mean_values_for_celltype)
        # Sort columns to keep group1 and group2 adjacent
        mean_comparison_df = mean_comparison_df.reindex(sorted(mean_comparison_df.columns), axis=1)
        # Update filename to reflect content
        mean_comparison_filename = os.path.join(save_dir, f"{celltype_name}_metrics_mean_comparison.csv")
        mean_comparison_df.to_csv(mean_comparison_filename)
        logger.info(f"G1/G2 metric mean comparison saved: {mean_comparison_filename}")


# -------------------------------
# Main workflow
# -------------------------------
logger.info("Start reading pathway mapping file")
try:
    pathway_df_full = pd.read_csv(pathway_file, sep="\t", usecols=['Gene1', 'Gene2', 'Direction'])
    logger.info(f"Pathway mapping loaded, total {len(pathway_df_full)} records")
except FileNotFoundError:
    logger.error(f"Pathway file {pathway_file} not found. Exiting.")
    exit()

for organism in os.listdir(base_dir):
    organism_dir = os.path.join(base_dir, organism)
    if not os.path.isdir(organism_dir):
        continue

    for file in os.listdir(organism_dir):
        if file.endswith(".csv") and file.startswith("results_"):
            ar_file_path = os.path.join(organism_dir, file)
            celltype_name = file.replace("results_", "").replace(".csv", "")
            process_celltype(organism, celltype_name, ar_file_path, pathway_df_full)

logger.info("All cell types processed.")