import os
import pandas as pd
from scipy.stats import ttest_ind
from datetime import datetime
from scar.config import load_config
from scar.logger import setup_logger
import numpy as np

# -------------------------------
# Log configuration
# -------------------------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger(name="TF", log_dir="TF_logs", run_id=run_id)
logger.info("Log initialization completed")

# -------------------------------
# Load configuration
# -------------------------------
logger.info("Starting to load configuration")
config = load_config("scar/config.yaml")
base_dir = config["TF_input_dir"]
metrics = config.get("metrics", ["support", "confidence", "leverage", "conviction", "lift"])
sampling_interval = config.get("sampling_interval","10")
filter_interval = config.get("filter_interval", "10")

logger.info(f"Configuration loaded, metrics list: {metrics}")

# -------------------------------
# Save directory root path
# -------------------------------
save_root = config.get("TF_output_dir", r"D:\software\code\code\scar2\TF_results")
os.makedirs(save_root, exist_ok=True)
logger.info(f"Results will be saved to: {save_root}")
# -------------------------------
# Helper function: when sample size difference is too large, randomly sample from larger group to match smaller group
# -------------------------------
def balance_sample_size(group_large, group_small):
    """
    Sample from larger group, rules:
    1. First ensure group_large is large group, group_small is small group (auto swap order)
    2. Large group sample size = small group sample size + 10 (core requirement: add 10 in all cases)
    3. If large group total sample size < sample size, take all large group samples (avoid error)
    Returns: sampled two groups (order: [sampled large group, original small group])
    """
    # Step 1: Ensure group_large is large group, group_small is small group (auto correct input order)
    if len(group_large) < len(group_small):
        group_large, group_small = group_small, group_large  # Swap to ensure large group comes first
        logger.debug(f"Input order corrected: original small group has larger sample size, swapped to [large group, small group]")
    
    # Step 2: Calculate core parameters
    small_len = len(group_small)    # Small group sample size
    large_len = len(group_large)    # Large group original sample size
    target_sample_size = small_len + sampling_interval  # Target sample size: small group + 10 (applies to all cases)
    
    # Step 3: Safety check: if large group insufficient "small group + 10", take all large group samples
    if large_len < target_sample_size:
        target_sample_size = large_len  # Take large group maximum sample size
        logger.warning(f"Large group sample size insufficient ({large_len} < target {small_len+10}), will use all {large_len} large group samples")
    else:
        logger.info(f"Large group sample size sufficient ({large_len} ≥ target {small_len+10})")
    
    # Step 4: Random sampling from large group (fixed random_state=42 for reproducibility)
    group_large_sampled = group_large.sample(n=target_sample_size, random_state=42)
    
    # Step 5: Log key information (for traceability)
    logger.info(
        f"Sample size balancing completed:"
        f"Original large group {large_len} → sampled {target_sample_size} (small group {small_len} + 10), "
        f"Balanced two groups sample size: {len(group_large_sampled)} (sampled large group)/{small_len} (original small group)"
    )
    
    # Return format: [sampled large group, original small group] (consistent with original function)
    return group_large_sampled, group_small

# -------------------------------
# Function definitions
# -------------------------------
def process_celltype(organism_name, celltype_name, ar_file, tf_file):
    """
    Process AR data and transcription factor data for specific cell type, and perform group comparison.
    """
    logger.info(f"Starting to process {organism_name}/{celltype_name}")
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
        merged_df.to_csv(os.path.join(save_dir, f"{celltype_name}_integration_table.csv"), index=False)
        logger.info(f"Integration table saved: {celltype_name}_integration_table.csv")

        # Extract unique values from Regulation column
        Uni_Regulation = merged_df['Regulation'].unique()

        p_values_all = {}  # Store P-values for all metrics
        p_values = {}  # Store P-values for all metrics

        if len(Uni_Regulation) > 1:
            # Multiple Regulation types, group by Regulation type (not considering in_ar)
            for type in Uni_Regulation:
                p_values_all[type] = {}
                for i in metrics:
                    # Data preprocessing
                    merged_df[i] = merged_df[i].replace([np.inf], 100)
                    merged_df[i] = merged_df[i].replace([-np.inf], -100)
                    merged_df[i] = np.where(merged_df[i] > 100, 100, merged_df[i])
                    merged_df[i] = np.where(merged_df[i] < -100, -100, merged_df[i])
                    # Split data (not considering in_ar)
                    group1 = merged_df[merged_df['Regulation'] == type][i].dropna()
                    group2 = merged_df[merged_df['Regulation'] != type][i].dropna()

                    # Check sample size
                    if len(group1) >= 2 and len(group2) >= 2:
                        sample_diff = abs(len(group2) - len(group1))
                        # Case 1: difference ≤ 10, test directly
                        if sample_diff <= filter_interval:
                            try:
                                stat, p = ttest_ind(group1, group2, equal_var=False)
                                logger.info(f"Metric: {i}, P-value: {p}")  # Add log
                                p_values_all[type][i] = p
                            except Exception as e:
                                logger.error(f"T-test failed: {e}")
                                p_values_all[type][i] = 0  # Test failure also assigns 0, consistent with your logic
                        # Case 2: difference > 10, sample from large group to match small group
                        else:
                            logger.warning(f"Sample size difference too large: {type}, {i} (original sample size: {len(group1)}/{len(group2)}, difference: {sample_diff})")
                            # Call helper function to sample and balance sample size
                            group1_balanced, group2_balanced = balance_sample_size(group1, group2)
                            try:
                                stat, p = ttest_ind(group1_balanced, group2_balanced, equal_var=False)
                                logger.info(f"Metric: {i}, P-value after balanced sampling: {p}")
                                p_values_all[type][i] = p
                            except Exception as e:
                                logger.error(f"T-test failed after balanced sampling: {e}")
                                p_values_all[type][i] = None
                    # Case 3: at least one group sample size < 1 (completely insufficient)
                    else:
                        logger.warning(f"Insufficient sample size: {type}, {i} (sample size: {len(group1)}/{len(group2)})")
                        p_values_all[type][i] = 0  # Assign 0 as per your requirement
        else:
            # Only one Regulation type
            logger.info(f"Cell type {celltype_name} has only one Regulation type, grouping AR data itself")

            the_regulation = Uni_Regulation[0]  # Get unique Regulation type

            # Create 'is_regulation' column, mark which gene pairs have the_regulation
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


            # Traverse each metric
            for i in metrics:
                # Data preprocessing
                ar_df[i] = ar_df[i].replace([np.inf], 100)
                ar_df[i] = ar_df[i].replace([-np.inf], -100)
                ar_df[i] = np.where(ar_df[i] > 100, 100, ar_df[i])
                ar_df[i] = np.where(ar_df[i] < -100, -100, ar_df[i])
                # Split data
                group1 = ar_df[ar_df['is_regulation'] == True][i].dropna()
                group2 = ar_df[ar_df['is_regulation'] == False][i].dropna()

                # Check basic sample size
                if len(group1) >= 2 and len(group2) >= 2:
                    sample_diff = abs(len(group2) - len(group1))
                    
                    if sample_diff <= filter_interval:
                        # Difference within allowed range, test directly
                        try:
                            stat, p = ttest_ind(group1, group2, equal_var=False)
                            logger.info(f"Metric: {i}, P-value: {p}")
                            p_values[i] = p
                        except Exception as e:
                            logger.error(f"T-test failed: {e}")
                            p_values[i] = None
                    
                    else:
                        # Difference too large, call existing balance_sample_size function to handle
                        logger.warning(f"Sample size difference too large: {i} (original sample size: {len(group1)}/{len(group2)}, difference: {sample_diff})")
                        
                        # Directly call existing balancing function (auto identify large/small group, sample by rules)
                        large_group_balanced, small_group_balanced = balance_sample_size(group1, group2)
                        
                        # Restore original grouping order
                        if len(group1) > len(group2):
                            # group1 is large group, use sampled large group and original small group
                            group1_balanced = large_group_balanced
                            group2_balanced = small_group_balanced
                        else:
                            # group2 is large group, use original small group and sampled large group
                            group1_balanced = small_group_balanced
                            group2_balanced = large_group_balanced
                        
                        logger.info(f"Sample size after balancing: {len(group1_balanced)}/{len(group2_balanced)}")
                        
                        # Execute test
                        try:
                            stat, p = ttest_ind(group1_balanced, group2_balanced, equal_var=False)
                            logger.info(f"Metric: {i}, P-value after balanced sampling: {p}")
                            p_values[i] = p
                        except Exception as e:
                            logger.error(f"T-test failed after balanced sampling: {e}")
                            p_values[i] = None
                else:
                    logger.warning(f"Insufficient sample size, skipping T-test: {i}")
                    p_values[i] = 0

            p_values_all[the_regulation] = p_values

        # Convert results to DataFrame and save
        p_df_all = pd.DataFrame(p_values_all)
        full_p_file = os.path.join(save_dir, f"{celltype_name}_P_values.csv")  # Simplified filename
        p_df_all.to_csv(full_p_file)
        logger.info(f"P-values saved to {full_p_file}")

    except Exception as e:
        logger.error(f"Failed to process cell type {celltype_name}: {e}")


# -------------------------------
# Main process
# -------------------------------
logger.info("Starting to read transcription factor mapping table")
try:
    tf_file = config["TF_pathway_file"]  # Load tf_file path
except KeyError:
    logger.error("Configuration file missing TF_pathway_file")
    exit()

for organism in os.listdir(base_dir):
    organism_dir = os.path.join(base_dir, organism)
    if not os.path.isdir(organism_dir):
        continue

    # Traverse all AR CSV files under organism, each file treated as a cell type
    for file in os.listdir(organism_dir):
        if file.endswith(".csv") and file.startswith("results_"):
            ar_file = os.path.join(organism_dir, file)
            # Auto generate cell type name: remove "results_" prefix and ".csv" suffix
            celltype_name = file.replace("results_", "").replace(".csv", "")
            process_celltype(organism, celltype_name, ar_file, tf_file)





















            
