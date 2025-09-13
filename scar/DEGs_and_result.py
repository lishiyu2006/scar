import os
import pandas as pd
import scanpy as sc
from datetime import datetime
from scar.config import load_config
from scar.logger import setup_logger

# ---------------------------
# Configuration
# ---------------------------

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(name="DEGs_and_results", log_dir="DEGs_logs",run_id = run_id)  # Call the previous log configuration function
    # Read configuration

    try:
        config = load_config("scar/config.yaml")
    except Exception as e:
        logger.error(f"Configuration file loading failed: {e}")
        raise
    logger.info("Starting to load configuration")
    # 1. Load configuration parameters
    as_rule_file = config["assoc_rules_file"]
    deg_dir = config["degs_file"]
    output_base_dir = config.get("output_base_dir", "./DEG_and_results")
    last_dir = os.path.basename(os.path.normpath(as_rule_file))
    out_dir = os.path.join(output_base_dir, last_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    os.makedirs(out_dir, exist_ok=True) 
    # ---------------------------
    # Loop through each cell type folder
    # ---------------------------
    deg_files = [f for f in os.listdir(deg_dir) if f.endswith(".csv")]

    for deg_file in deg_files:
        deg_path = os.path.join(deg_dir, deg_file)
        deg_df = pd.read_csv(deg_path)
        deg_genes_set = set(deg_df['names'].dropna())

        assoc_file_path = os.path.join(as_rule_file, deg_file)
        assoc_df = pd.read_csv(assoc_file_path)
        assoc_genes_set = set(pd.concat([assoc_df.iloc[:,0], assoc_df.iloc[:,1]]).dropna())

        # Take intersection
        common_genes = list(deg_genes_set & assoc_genes_set)

        # Save
        common_genes_file = os.path.join(out_dir, f"intersection_{deg_file}")
        pd.DataFrame({'gene': common_genes}).to_csv(common_genes_file, index=False)
        logger.info(f"Intersection genes saved to: {common_genes_file}")


if __name__ == "__main__":
    main()