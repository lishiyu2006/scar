
import numpy as np
import pandas as pd
import os
import time

def generate_expression_matrix(
    n_genes: int = 100,
    n_cells: int = 150,
    binary: bool = True,
    sparsity: float = 0.033,
    save_path: str = None,
    add_timestamp: bool = True
) -> pd.DataFrame:
    """
    Generate random expression matrix with genes as rows and cells as columns.

    Parameters:
        n_genes (int): Number of genes (rows)
        n_cells (int): Number of cells (columns)
        binary (bool): Whether to generate 0/1 binary matrix
        sparsity (float): Sparsity rate (higher means more sparse)
        save_path (str): Optional, save path (.csv)

    Returns:
        pd.DataFrame: Expression matrix (gene × cell)
    """
    shape = (n_genes, n_cells)
    if binary:
        #matrix = (np.random.rand(*shape) > sparsity).astype(int)
        # Use float32 (reduces memory by half)
        # Generate float32 random numbers directly to avoid intermediate float64 arrays
        matrix = (np.random.default_rng().random(shape, dtype=np.float32) > sparsity).astype(np.int8)
    else:
        matrix = np.random.poisson(lam=1.0, size=shape)

    gene_names = [f"Gene{i}" for i in range(n_genes)]
    cell_names = [f"Cell{j}" for j in range(n_cells)]
    df = pd.DataFrame(matrix, index=gene_names, columns=cell_names)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if add_timestamp:
            # Add timestamp in format: YYYYMMDD_HHMMSS
            base, ext = os.path.splitext(save_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"{base}_{timestamp}{ext}"
        
        print(f"Save path: {save_path}")
        df.to_csv(save_path)

    return df, save_path
