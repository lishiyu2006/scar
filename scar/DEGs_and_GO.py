import os
import yaml
import scanpy as sc
import pandas as pd
from datetime import datetime
from scar.config import load_config
from scar.logger import setup_logger


# ---------- Small tools ----------
def _find_first_exists(dirpath, names):
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.exists(p):
            return p
    return None

def _load_genes(genes_path):
    # genes.tsv / features.tsv, may have 1 or 2 columns: gene_id, gene_symbol
    df = pd.read_csv(genes_path, sep="\t", header=None)
    col = df.iloc[:,1] if df.shape[1] >= 2 else df.iloc[:,0]
    return col.astype(str).values

def _load_barcodes(barcodes_path):
    bc = pd.read_csv(barcodes_path, sep="\t", header=None).iloc[:,0].astype(str).values
    return bc

def _load_cell_labels(cell_type_path, obs_names):
    """
    Support two formats:
    1) Only one column: one label per row, aligned with barcodes in order
    2) Two or more columns: contains barcode column and label column (column names can be barcode / cell_type / condition / group / label)
    """
    # First assume there is a header
    df = pd.read_csv(cell_type_path, sep="\t")

    # Determine if first row was mistakenly treated as header (e.g. only read one column)
    if df.shape[1] == 1:
        # Actually no header, read again
        df = pd.read_csv(cell_type_path, sep="\t", header=None)
        use_header = False
        print(f"[cell_type] File {cell_type_path} identified as no header")
    else:
        use_header = True
        print(f"[cell_type] File {cell_type_path} identified as having header")

    if df.shape[1] == 1:
        labels = df.iloc[:,0].astype(str).values
        if len(labels) != len(obs_names):
            diff = len(obs_names) - len(labels)
            raise ValueError(
                f"cell_type file is single column, but row count({len(labels)}) does not match cell count({len(obs_names)}), "
                f"missing {diff} rows.\n"
                f"obs_names example: {list(obs_names[:3])} ... {list(obs_names[-3:])}\n"
                f"cell_type example: {labels[:3]} ... {labels[-3:]}"
                f"cell_type file is single column, but row count({len(labels)}) does not match cell count({len(obs_names)})."
            )
        return pd.Series(labels, index=obs_names)

    # Two or more columns: find barcode column & label column
    if use_header:
        cols_lower = [c.lower() for c in df.columns]
        # barcode column
        if "barcode" in cols_lower:
            bc_col = df.columns[cols_lower.index("barcode")]
        else:
            bc_col = df.columns[0]
        # label column: prioritize these keywords
        label_col = None
        for key in ["cell_type", "celltype", "condition", "group", "label"]:
            if key in cols_lower:
                label_col = df.columns[cols_lower.index(key)]
                break
        if label_col is None:
            label_col = df.columns[1]
        mapping = pd.Series(df[label_col].astype(str).values,
                            index=df[bc_col].astype(str).values)
        labels = [mapping.get(bc, "NA") for bc in obs_names]
        return pd.Series(labels, index=obs_names)
    else:
        # No header: default column 0 is barcode, column 1 is label
        mapping = pd.Series(df.iloc[:,1].astype(str).values,
                            index=df.iloc[:,0].astype(str).values)
        labels = [mapping.get(bc, "NA") for bc in obs_names]
        return pd.Series(labels, index=obs_names)



def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(name="DEGs", log_dir="DGEs_logs",run_id = run_id)  # Call the previous log configuration function
    # Read configuration
    logger.info("Starting to load configuration")
    try:
        config = load_config("scar/config.yaml")
    except Exception as e:
        logger.error(f"Configuration file loading failed: {e}")
        raise

        # 1. Load configuration parameters
    data_dir = config["degs_data_dir"]
    base_output_dir = config["degs_output_dir"]

    mtx_path = os.path.join(data_dir, "matrix.mtx")
    genes_path = os.path.join(data_dir, "genes.tsv")      # or features.tsv, depending on your data
    barcodes_path = os.path.join(data_dir, "barcodes.tsv")
    cell_type_path = os.path.join(data_dir, "cell_type.tsv")

    for p, name in [(mtx_path,"matrix.mtx"), (genes_path,"genes.tsv/features.tsv"),
                    (barcodes_path,"barcodes.tsv"), (cell_type_path,"cell_type.tsv")]:
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"Not found {name}, please check {data_dir} or explicitly specify path in config.")

    # === Get last folder name from input path ===
    last_dir = os.path.basename(os.path.normpath(data_dir))
    out_dir = os.path.join(base_output_dir, last_dir)
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    p_threshold = float(config.get("p_threshold", 0.05))
    p_filter_dir = config.get("p_filter_direction", "lt").lower()  # "lt" or "gt"
    min_cells_per_group = int(config.get("min_cells_per_group", 10))
    de_method = config.get("de_method", "wilcoxon") # "wilcoxon"/"t-test"/"logreg"

    # 1) Read sparse matrix as cells x genes
    logger.info("Reading matrix.mtx ...")
    ad = sc.read_mtx(mtx_path).T

    # 2) Set gene names & cell names
    logger.info("Reading genes / barcodes ...")
    gene_symbols = _load_genes(genes_path)
    barcodes = _load_barcodes(barcodes_path)

    if ad.n_vars != len(gene_symbols):
        raise ValueError(f"Gene count mismatch: mtx has {ad.n_vars} columns, but genes has {len(gene_symbols)} rows.")
    if ad.n_obs != len(barcodes):
        raise ValueError(f"Cell count mismatch: mtx has {ad.n_obs} rows, but barcodes has {len(barcodes)} rows.")


    ad.var_names = gene_symbols
    ad.var_names_make_unique()
    ad.obs_names = barcodes

    # 3) Load grouping (cell_type/condition)
    logger.info("Reading cell_type / condition ...")
    labels = _load_cell_labels(cell_type_path, ad.obs_names)
    ad.obs["group"] = labels.astype("category")


    # Clean NA, too small groups
    valid_mask = ad.obs["group"] != "NA"
    if valid_mask.sum() < ad.n_obs:
        logger.warning(f"{(~valid_mask).sum()} cells not found in cell_type file (marked as NA), will be removed.")
        ad = ad[valid_mask].copy()


    vc = ad.obs["group"].value_counts()
    small_groups = vc[vc < min_cells_per_group].index.tolist()
    if len(small_groups) > 0:
        logger.warning(f"Following groups have cell count < {min_cells_per_group}, will be removed: {small_groups}")
        ad = ad[~ad.obs["group"].isin(small_groups)].copy()

    if ad.obs["group"].nunique() < 2:
        raise ValueError("Insufficient valid groups (less than 2), cannot perform differential analysis. Please check cell_type file content.")

    # 4) Preprocessing + differential analysis
    logger.info("Normalize + log1p ...")
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    logger.info(f"Differential analysis (method={de_method})...")
    sc.tl.rank_genes_groups(ad, groupby="group", method=de_method)

    # 5) Export results (total table + p filtering + grouped sub-tables)
    logger.info("Exporting total table and filtered table ...")
    other_path = os.path.join(out_dir, f"other")
    os.makedirs(other_path, exist_ok=True)
    all_df = sc.get.rank_genes_groups_df(ad, group=None)
    all_csv = os.path.join(other_path, "DEGs_all_groups.csv")
    all_df.to_csv(all_csv, index=False)
    # Select p-values less than 0.05
    if p_filter_dir == "lt":
        flt_df = all_df[all_df["pvals_adj"] < p_threshold]
        tag = f"padj_lt_{p_threshold}"
    else:
        flt_df = all_df[all_df["pvals_adj"] > p_threshold]
        tag = f"padj_gt_{p_threshold}"



    flt_csv = os.path.join(other_path, f"DEGs_filtered_{tag}.csv")
    flt_df.to_csv(flt_csv, index=False)

    # Save each group separately (easier to view)
    logger.info("Exporting individual results for each group ...")
    for grp in ad.obs["group"].cat.categories:
        # Get DEGs for each group
        sub_df = sc.get.rank_genes_groups_df(ad, group=grp)
        
        # Filter significantly upregulated genes
        up_sig_df = sub_df[(sub_df["pvals_adj"] < 0.05) & (sub_df["logfoldchanges"] > 0)]
        
        # Save CSV
        sub_path = os.path.join(out_dir, f"results_{grp}.csv")
        up_sig_df.to_csv(sub_path, index=False)
        
        # Output log
        logger.info(f"Saving group {grp} -> {sub_path} (significantly upregulated genes rows={len(up_sig_df)}, columns={len(up_sig_df.columns)})")

    logger.info(f"Completed! Total table -> {all_csv} | Filtered table -> {flt_csv}")

    logger.info(f"Differential analysis completed! Results saved to {out_dir}")



if __name__ == "__main__":
    main()