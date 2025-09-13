import os
import pandas as pd
import anndata as ad
import scipy.io
import h5py
import scipy.sparse
import loompy
import numpy as np
import logging
#import pyreadr
from scar.pre_input import prepare_columns_for_loading

logger = logging.getLogger("scAR") 

import threading

def input_with_timeout(prompt, timeout, default):
    user_input = [default]  # Use list for easy modification

    def ask():
        user_input[0] = input(prompt)

    thread = threading.Thread(target=ask)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(f"\nTimeout, using default path: {default}")
        return default
    else:
        return user_input[0]


def split_df_by_columns(df: pd.DataFrame, batch_size: int):
    """
    Split DataFrame by columns in batches, ensuring genes as rows and cells as columns.
    """
    n_cols = df.shape[1]
    for start in range(0, n_cols, batch_size):
        end = min(start + batch_size, n_cols)
        yield df.iloc[:, start:end]


def split_mtx_by_celltype_batches(mtx_path, celltype_path, batch_size=2000):
    base_dir = os.path.dirname(mtx_path)
    genes_path = os.path.join(base_dir, "genes.tsv")
    barcodes_path = os.path.join(base_dir, "barcodes.tsv")

    matrix = scipy.io.mmread(mtx_path).tocsc()
    genes = pd.read_csv(genes_path, header=None, sep='\t')[1].tolist()
    barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')[0].tolist()
    cell_types = pd.read_csv(celltype_path, header=None, sep='\t')[0].tolist()

    if len(barcodes) != len(cell_types):
        raise ValueError("Number of barcodes does not match number of cell types")

    barcode2celltype = pd.DataFrame({
        'barcode': barcodes,
        'cell_type': cell_types
    })

    # Build barcode to column index mapping
    barcode_to_idx = {bc: i for i, bc in enumerate(barcodes)}
    # Get column index based on barcode
    barcode2celltype['col_idx'] = barcode2celltype['barcode'].map(barcode_to_idx)
    # Group by cell type
    grouped = barcode2celltype.groupby('cell_type')

    for cell_type, group in grouped:
        cols_idx = group['col_idx'].dropna().astype(int).tolist()
        n_cells = len(cols_idx)
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            batch_cols = cols_idx[start:end]
            sub_matrix = matrix[:, batch_cols]
            yield {
                'cell_type': cell_type,
                'expr': sub_matrix,
                'genes': genes,
                'barcodes': [barcodes[i] for i in batch_cols]
            }

def load_expression_matrix(path: str, batch_size: int = 2000):
    """
    Automatically identify file format and read expression matrix in batches by cells (columns), returning DataFrame iterator.
    All returned DataFrames ensure "genes as rows, cells as columns".
    """
    path = os.path.abspath(path)

    if path.endswith(".csv") or path.endswith(".tsv"):
        return csv_read_by_rows(path, batch_size)

    elif path.endswith(".mtx") or path.endswith(".mtx.gz"):
        return mtx_read_by_rows(path, batch_size)

    elif path.endswith(".h5ad"):
        return read_h5ad_by_cols(path, batch_size)

    elif path.endswith(".loom"):
        return read_loom_by_cols(path, batch_size)

#    elif path.endswith(".rds") or path.endswith(".RData"):
#        return read_rds_by_cols(path, batch_size)

    elif path.endswith(".npy") or path.endswith(".npz"):
        return read_npy_by_cols(path, batch_size)

    elif path.endswith(".h5"):
        return read_h5_by_cols(path, batch_size)
    else:
        raise ValueError(f"Unsupported input format: {path}")

def read_h5_by_cols(h5_path: str, batch_size: int):
    store = pd.HDFStore(h5_path, mode='r')
    n_cells = store.get_storer('expr').ncols
    gene_names = store.select('expr', start=0, stop=1).index  # Only read gene names

    for start in range(0, n_cells, batch_size):
        stop = min(start + batch_size, n_cells)
        batch = store.select('expr', start=None, stop=None, columns=list(range(start, stop)))
        batch.index = gene_names
        yield batch

    store.close()


def csv_read_by_rows(csv_path, batch_size=2000, start=1):
    index_name, data_cols = prepare_columns_for_loading(csv_path, sep=",")
    if data_cols is None:
        raise ValueError("Column names are empty")
    else:
        logger.info(f"Column extraction completed")    
    
    if data_cols is None:
        raise ValueError("Must provide data_cols parameter to avoid repeated reading of column names")

    total = len(data_cols)

    for i in range(start - 1, total, batch_size):
        end = min(i + batch_size, total)
        selected_cols = [index_name] + data_cols[i:end]
        batch_df = pd.read_csv(csv_path, sep=",", usecols=selected_cols, index_col=0)
        yield batch_df


def mtx_read_by_rows(mtx_path: str, batch_size: int):
    # Automatically load genes and barcodes
    base_dir = os.path.dirname(mtx_path)
    genes_path = os.path.join(base_dir, "features.tsv")
    barcodes_path = os.path.join(base_dir, "barcodes.tsv")

    matrix = scipy.io.mmread(mtx_path).tocsc()
    genes = pd.read_csv(genes_path, header=None, sep='\t')[0].tolist()
    barcodes = pd.read_csv(barcodes_path, header=None, sep='\t')[0].tolist()

    df = pd.DataFrame.sparse.from_spmatrix(matrix, index=genes, columns=barcodes)
    h5_path = mtx_path.replace(".mtx", ".h5").replace(".gz", "")
    df.to_hdf(h5_path, key="expr", mode="w")

    return read_h5_by_cols(h5_path, batch_size)

def read_h5ad_by_cols(h5ad_path: str, batch_size: int):
    adata = ad.read_h5ad(h5ad_path)
    n_cells = adata.n_obs
    gene_names = adata.var_names

    for start in range(0, n_cells, batch_size):
        stop = min(start + batch_size, n_cells)
        submatrix = adata.X[:, start:stop]
        if hasattr(submatrix, "toarray"):
            submatrix = submatrix.toarray()
        df = pd.DataFrame(submatrix.T, columns=gene_names).T
        yield df

def read_loom_by_cols(loom_path: str, batch_size: int):
    ds = loompy.connect(loom_path, mode="r")
    gene_names = ds.ra["Gene"] if "Gene" in ds.ra else [f"gene_{i}" for i in range(ds.shape[0])]

    for start in range(0, ds.shape[1], batch_size):
        stop = min(start + batch_size, ds.shape[1])
        submatrix = ds[:, start:stop]
        df = pd.DataFrame(submatrix, index=gene_names)
        yield df

    ds.close()



#def read_rds_by_cols(path: str, batch_size: int = 2000):
    # Read .rds/.RData files (assuming Seurat saved expression matrix with genes as rows, cells as columns)
#    result = pyreadr.read_r(path)
#    if len(result) == 0:
#        raise ValueError(f"No objects found in {path}")
    
    # Take the first object
#    df = result[None] if None in result else list(result.values())[0]

#    if not isinstance(df, pd.DataFrame):
#        raise TypeError(f"Expected DataFrame in RData, got {type(df)}")

#    n_cols = df.shape[1]
#    for start in range(0, n_cols, batch_size):
#        end = min(start + batch_size, n_cols)
#        yield df.iloc[:, start:end]


def read_npy_by_cols(path: str, batch_size: int = 2000):
    if path.endswith(".npz"):
        loaded = np.load(path, allow_pickle=True)
        if "matrix" in loaded:
            matrix = loaded["matrix"]
        else:
            raise KeyError(f"No 'matrix' key found in {path}")
    else:
        matrix = np.load(path, allow_pickle=True)

    if scipy.sparse.issparse(matrix):
        matrix = matrix.toarray()

    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Expected ndarray, got {type(matrix)}")

    # Default behavior: genes as rows, cells as columns
    df = pd.DataFrame(matrix)

    # Assign index names
    df.index = [f"gene_{i}" for i in range(df.shape[0])]
    df.columns = [f"cell_{i}" for i in range(df.shape[1])]

    n_cols = df.shape[1]
    for start in range(0, n_cols, batch_size):
        end = min(start + batch_size, n_cols)
        yield df.iloc[:, start:end]