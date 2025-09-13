import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from datetime import datetime
from scar.config import load_config
from scar.logger import setup_logger

# -------------------------------
# Log configuration
# -------------------------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(name="reactome_comparison", log_dir="reactome_logs", run_id=run_id)
logger.info("Log initialization completed")

# -------------------------------
# Load configuration
# -------------------------------
logger.info("Starting to load configuration")
try:
    config = load_config("scar/config.yaml")
except FileNotFoundError:
    logger.error("Configuration file scar/config.yaml not found, please check path.")
    # Provide a backup configuration so the script can run


base_dir = config["reactome_input_dir"]
pathway_file = config["reactome_pathway_file"]
metrics = config.get("metrics", ["support", "confidence", "leverage", "conviction", "lift"])
logger.info(f"Configuration loaded, metrics list: {metrics}")

# -------------------------------
# Save directory root path
# -------------------------------
save_root = config.get("reactome_output_dir", r"D:\software\code\code\scar2\reactome_results")
os.makedirs(save_root, exist_ok=True)
logger.info(f"Results will be saved to: {save_root}")

# -------------------------------
# Function definitions
# -------------------------------
def process_celltype(organism_name, celltype_name, ar_file, pathway_df_full):
    logger.info(f"Starting to process {organism_name}/{celltype_name}")
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
        logger.error(f"Error reading or processing file {ar_file}: {e}. Please check if file exists and contains 'Gene1', 'Gene2' columns.")
        return

    # Intersect with pathway table
    merged_df = pd.merge(ar_df, pathway_df_full, on=['Gene1', 'Gene2'], how='inner')
    if merged_df.empty:
        logger.warning(f"File {ar_file} has empty intersection with pathway table, skipping this cell type.")
        return

    merged_df.to_csv(os.path.join(save_dir, f"{celltype_name}_integration_table.csv"), index=False)
    logger.info(f"Integration table saved: {celltype_name}_integration_table.csv")

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
    # Modified: Main dictionary for collecting means of all group1 and group2
    mean_values_for_celltype = {} 

    unique_directions = merged_df['Direction'].unique()

    # Outer loop: traverse each direction type
    for dir_type in unique_directions:
        p_values_all[dir_type] = {}
        
        safe_name = direction_mapping.get(str(dir_type), str(dir_type))
        safe_name = safe_name.replace('<', '_').replace('>', '_').replace('|', '_')

        # Modified: Create Series for current pathway group1 and group2 to store means
        g1_means_series = pd.Series(index=metrics, dtype=float)
        g2_means_series = pd.Series(index=metrics, dtype=float)

        # Inner loop: traverse each metric
        for metric in metrics:
            # Split data
            group1 = merged_df[merged_df['Direction'] == dir_type][metric].dropna()
            group2 = merged_df[merged_df['Direction'] != dir_type][metric].dropna()
            
            # Calculate and store means of group1 and group2
            g1_means_series[metric] = group1.mean()
            g2_means_series[metric] = group2.mean()

            # Save group1 and group2 data (retained)
            group1.to_csv(os.path.join(group_data_save_dir, f"{celltype_name}_{safe_name}_{metric}_group1.csv"), index=False, header=[metric])
            group2.to_csv(os.path.join(group_data_save_dir, f"{celltype_name}_{safe_name}_{metric}_group2.csv"), index=False, header=[metric])

            # Difference test (retained)
            if len(group1) > 1 and len(group2) > 1:
                stat, p = ttest_ind(group1, group2, equal_var=False)
                p_values_all[dir_type][metric] = p
            else:
                p_values_all[dir_type][metric] = None
        
        # Modified: Store both g1 and g2 mean Series in main dictionary using composite keys
        mean_values_for_celltype[f"{safe_name}_group1"] = g1_means_series
        mean_values_for_celltype[f"{safe_name}_group2"] = g2_means_series
        logger.info(f"P-values and G1/G2 means calculated for {celltype_name} pathway {safe_name}.")


    # 1. Save P-value summary table (retained)
    p_df_all = pd.DataFrame(p_values_all)
    full_p_file = os.path.join(save_dir, f"{celltype_name}_all_metrics_pathways_P_value.csv")
    p_df_all.to_csv(full_p_file)
    logger.info(f"All metrics P-value table saved: {full_p_file}")

    # 2. Generate and save group1/group2 mean comparison table
    if not mean_values_for_celltype:
        logger.warning(f"No data available for mean summary in {celltype_name}.")
    else:
        mean_comparison_df = pd.DataFrame(mean_values_for_celltype)
        
        # New: Sort columns to ensure group1 and group2 are adjacent
        mean_comparison_df = mean_comparison_df.reindex(sorted(mean_comparison_df.columns), axis=1)

        # Modified: Update filename to reflect its content
        mean_comparison_filename = os.path.join(save_dir, f"{celltype_name}_metrics_mean_comparison.csv")
        mean_comparison_df.to_csv(mean_comparison_filename)
        logger.info(f"G1/G2 metrics mean comparison table saved: {mean_comparison_filename}")


# -------------------------------
# Main process
# -------------------------------
logger.info("Starting to read pathway mapping table")
try:
    pathway_df_full = pd.read_csv(pathway_file, sep="\t", usecols=['Gene1', 'Gene2', 'Direction'])
    logger.info(f"Pathway mapping table read completed, total {len(pathway_df_full)} records")
except FileNotFoundError:
    logger.error(f"Pathway file {pathway_file} not found, program terminated.")
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