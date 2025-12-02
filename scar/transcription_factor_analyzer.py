import os
import pandas as pd
from scipy.stats import ttest_ind
from datetime import datetime
from scar.configuration_manager import load_config
from scar.logging_system import setup_logger
import numpy as np

# -------------------------------
# Logging configuration
# -------------------------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(name="TF", log_dir="TF_logs", run_id=run_id)
logger.info("Logger initialized")

# -------------------------------
# Load configuration
# -------------------------------
logger.info("Starting to load configuration")
config = load_config("scar/project_settings.yaml")
base_dir = config["TF_input_dir"]
metrics = config.get("metrics", ["support", "confidence", "leverage", "conviction", "lift"])
sampling_interval = config.get("sampling_interval","10")
filter_interval = config.get("filter_interval", "10")

logger.info(f"Configuration loaded. Metrics: {metrics}")

# -------------------------------
# Output root directory
# -------------------------------
save_root = config.get("TF_output_dir", r"D:\software\code\code\scar2\TF_results")
os.makedirs(save_root, exist_ok=True)
logger.info(f"Results will be saved to: {save_root}")
# -------------------------------
# Helper: when sample sizes differ greatly, sample from the larger group
# -------------------------------
def balance_sample_size(group_large, group_small):
    """
    Sample from the larger group using rules:
    1. Ensure group_large is the larger group and group_small is the smaller group (auto-swap if needed).
    2. Sampling size for the large group = small group size + sampling_interval.
    3. If large group size < target sampling size, use all samples from the large group.
    Returns: two groups [sampled_large_group, original_small_group].
    """
    # Step 1: ensure group_large is larger and group_small is smaller (auto-correct)
    if len(group_large) < len(group_small):
        group_large, group_small = group_small, group_large  # swap to keep large group first
        logger.debug("Input order corrected: original small group has more samples; swapped to [large, small]")
    
    # Step 2: compute core parameters
    small_len = len(group_small)
    large_len = len(group_large)
    target_sample_size = small_len + sampling_interval
    
    # Step 3: safety check — if large group has fewer than target size, use all
    if large_len < target_sample_size:
        target_sample_size = large_len
        logger.warning(f"Large group insufficient ({large_len} < target {small_len + sampling_interval}); using all {large_len} samples from large group")
    else:
        logger.info(f"Large group sufficient ({large_len} ≥ target {small_len + sampling_interval})")
    
    # Step 4: random sampling (fixed random_state=42 for reproducibility)
    group_large_sampled = group_large.sample(n=target_sample_size, random_state=42)
    
    # Step 5: log key info
    logger.info(
        f"Sample size balancing complete: "
        f"large {large_len} → sampled {target_sample_size} (small {small_len} + interval), "
        f"balanced sizes: {len(group_large_sampled)} (sampled large) / {small_len} (original small)"
    )
    
    return group_large_sampled, group_small

# -------------------------------
# Function definition
# -------------------------------
def process_celltype(organism_name, celltype_name, ar_file, tf_file):
    """
    Process AR data and transcription factor data for a specific cell type and perform group comparisons.
    """
    logger.info(f"Processing {organism_name}/{celltype_name}")
    save_dir = os.path.join(save_root, organism_name, celltype_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Read AR data
        ar_df = pd.read_csv(ar_file)
        if 'row' in ar_df.columns and 'col' in ar_df.columns:
            ar_df = ar_df.rename(columns={'row': 'Gene_1', 'col': 'Gene_2'})
        ar_df = ar_df[['Gene_1', 'Gene_2'] + metrics]

        # Read transcription factor data
        tf_df = pd.read_csv(tf_file,
                            sep='\s+',
                            header=0,
                            usecols=[0, 2, 4],
                            names=['TF', 'Target', 'Regulation'],
                            comment='#',
                            engine='python')

        # Merge data
        merged_df = pd.merge(ar_df, tf_df,
                             left_on=['Gene_1', 'Gene_2'],
                             right_on=['TF', 'Target'],
                             how='inner')
        merged_filename = f"{celltype_name}_merged_table.csv"
        merged_df.to_csv(os.path.join(save_dir, merged_filename), index=False)
        logger.info(f"Merged table saved: {merged_filename}")

        # Unique values in Regulation
        Uni_Regulation = merged_df['Regulation'].unique()

        p_values_all = {}
        p_values = {}

        if len(Uni_Regulation) > 1:
            # Multiple Regulation types; group by Regulation (ignore in_ar)
            for type in Uni_Regulation:
                p_values_all[type] = {}
                for i in metrics:
                    # Data preprocessing
                    merged_df[i] = merged_df[i].replace([np.inf], 100)
                    merged_df[i] = merged_df[i].replace([-np.inf], -100)
                    merged_df[i] = np.where(merged_df[i] > 100, 100, merged_df[i])
                    merged_df[i] = np.where(merged_df[i] < -100, -100, merged_df[i])
                    # Split data
                    group1 = merged_df[merged_df['Regulation'] == type][i].dropna()
                    group2 = merged_df[merged_df['Regulation'] != type][i].dropna()

                    # Check sample sizes
                    if len(group1) >= 2 and len(group2) >= 2:
                        sample_diff = abs(len(group2) - len(group1))
                        # Case 1: difference within threshold — test directly
                        if sample_diff <= filter_interval:
                            try:
                                stat, p = ttest_ind(group1, group2, equal_var=False)
                                logger.info(f"Metric: {i}, P-value: {p}")
                                p_values_all[type][i] = p
                            except Exception as e:
                                logger.error(f"T-test failed: {e}")
                                p_values_all[type][i] = 0
                        # Case 2: difference too large — sample to balance
                        else:
                            logger.warning(f"Sample size difference too large: {type}, {i} (sizes: {len(group1)}/{len(group2)}, diff: {sample_diff})")
                            group1_balanced, group2_balanced = balance_sample_size(group1, group2)
                            try:
                                stat, p = ttest_ind(group1_balanced, group2_balanced, equal_var=False)
                                logger.info(f"Metric: {i}, P-value after balancing: {p}")
                                p_values_all[type][i] = p
                            except Exception as e:
                                logger.error(f"T-test after balancing failed: {e}")
                                p_values_all[type][i] = None
                    else:
                        # Case 3: at least one group has <2 samples
                        logger.warning(f"Insufficient sample size: {type}, {i} (sizes: {len(group1)}/{len(group2)})")
                        p_values_all[type][i] = 0
        else:
            # Single Regulation type
            logger.info(f"Cell type {celltype_name} has only one Regulation type; grouping on AR data itself")

            the_regulation = Uni_Regulation[0]

            # Create 'is_regulation' flag indicating pairs with the_regulation
            ar_df['is_regulation'] = ar_df.apply(
                lambda row: (row['Gene_1'] in tf_df['TF'].values) and
                            (row['Gene_2'] in tf_df['Target'].values) and
                            (
                                lambda row: tf_df[(tf_df['TF'] == row['Gene_1']) & (tf_df['Target'] == row['Gene_2'])]['Regulation'].iloc[0] == the_regulation
                                 if not tf_df[(tf_df['TF'] == row['Gene_1']) & (tf_df['Target'] == row['Gene_2'])].empty
                                 else False
                            )(row),
                axis=1
            )

            # Iterate each metric
            for i in metrics:
                # Data preprocessing
                ar_df[i] = ar_df[i].replace([np.inf], 100)
                ar_df[i] = ar_df[i].replace([-np.inf], -100)
                ar_df[i] = np.where(ar_df[i] > 100, 100, ar_df[i])
                ar_df[i] = np.where(ar_df[i] < -100, -100, ar_df[i])
                # Split data
                group1 = ar_df[ar_df['is_regulation'] == True][i].dropna()
                group2 = ar_df[ar_df['is_regulation'] == False][i].dropna()

                # Basic sample size check
                if len(group1) >= 2 and len(group2) >= 2:
                    sample_diff = abs(len(group2) - len(group1))
                    
                    if sample_diff <= filter_interval:
                        try:
                            stat, p = ttest_ind(group1, group2, equal_var=False)
                            logger.info(f"Metric: {i}, P-value: {p}")
                            p_values[i] = p
                        except Exception as e:
                            logger.error(f"T-test failed: {e}")
                            p_values[i] = None
                    
                    else:
                        logger.warning(f"Sample size difference too large: {i} (sizes: {len(group1)}/{len(group2)}, diff: {sample_diff})")
                        large_group_balanced, small_group_balanced = balance_sample_size(group1, group2)
                        if len(group1) > len(group2):
                            group1_balanced = large_group_balanced
                            group2_balanced = small_group_balanced
                        else:
                            group1_balanced = small_group_balanced
                            group2_balanced = large_group_balanced
                        
                        logger.info(f"Balanced sample sizes: {len(group1_balanced)}/{len(group2_balanced)}")
                        try:
                            stat, p = ttest_ind(group1_balanced, group2_balanced, equal_var=False)
                            logger.info(f"Metric: {i}, P-value after balancing: {p}")
                            p_values[i] = p
                        except Exception as e:
                            logger.error(f"T-test after balancing failed: {e}")
                            p_values[i] = None
                else:
                    logger.warning(f"Insufficient sample size, skip T-test: {i}")
                    p_values[i] = 0

            p_values_all[the_regulation] = p_values

        # Convert results to DataFrame and save
        p_df_all = pd.DataFrame(p_values_all)
        full_p_file = os.path.join(save_dir, f"{celltype_name}_P_values.csv")
        p_df_all.to_csv(full_p_file)
        logger.info(f"P-values saved to {full_p_file}")

    except Exception as e:
        logger.error(f"Failed to process cell type {celltype_name}: {e}")


# -------------------------------
# Main workflow
# -------------------------------
logger.info("Start reading transcription factor mapping file")
try:
    tf_file = config["TF_pathway_file"]  # Load tf_file path
except KeyError:
    logger.error("Missing TF_pathway_file in configuration")
    exit()

for organism in os.listdir(base_dir):
    organism_dir = os.path.join(base_dir, organism)
    if not os.path.isdir(organism_dir):
        continue

    # Iterate AR CSV files under organism; each file is treated as a cell type
    for file in os.listdir(organism_dir):
        if file.endswith(".csv") and file.startswith("results_"):
            ar_file = os.path.join(organism_dir, file)
            # Auto-generate cell type name: remove 'results_' prefix and '.csv' suffix
            celltype_name = file.replace("results_", "").replace(".csv", "")
            process_celltype(organism, celltype_name, ar_file, tf_file)





















            
