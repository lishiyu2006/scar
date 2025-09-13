import numpy as np
import pandas as pd
import torch
import logging
import math
import config
import os
from datetime import datetime
from scar.logger import setup_logger
from scar.config import load_config,get
from scar.output import compute_metrics_with_any_after_full_calc

logger = logging.getLogger("scAR")

def other_function(config):
    # Example: access parameters
    thresholds = config.get("thresholds", {})
    support_th = thresholds.get("support", 0.0)
    confidence_th = thresholds.get("confidence", 0.0)
    lift_th = thresholds.get("lift", 0.0)
    leverage_th = thresholds.get("leverage", 0.0)
    conviction_th = thresholds.get("conviction", 0.0)


def binarize_and_to_cuda(batch_df: pd.DataFrame, device: str = 'cuda:0') -> torch.Tensor:
    # bool tensor saves more GPU memory
    tensor = torch.tensor(batch_df.values, dtype=torch.bool, device=device)
    return tensor



def calculate_support(matrix: torch.Tensor) -> torch.Tensor:
    # matrix: bool tensor [N_genes, N_cells]
    # Support = number of cells expressing this gene / total number of cells
    support = matrix.float().mean(dim=1)
    return support  # [N_genes], float tensor


def calculate_support_ac_chunked(matrix: torch.Tensor, chunk_size=1000, device="cuda"):
    """
    Calculate support(A∩C) = (X @ X.T) / num_cells, process in row chunks to avoid OOM.
    
    Parameters:
        matrix: Tensor, shape [N_genes, N_cells]
        chunk_size: Number of rows per chunk
        device: GPU or CPU

    Returns:
        support_ac: Tensor, shape [N_genes, N_genes], on CPU
    """
    N, M = matrix.shape
    support_ac_parts = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = matrix[start:end].float()  # Convert to float for matrix multiplication
        # Calculate product of this chunk with all subsequent rows, only upper triangle and diagonal
        # Calculate all first, then filter upper triangle later
        partial = torch.matmul(chunk, matrix.T.float()) / M  # [chunk_size, N]
        support_ac_parts.append(partial.cpu())  # Convert to CPU immediately to release GPU memory

    return torch.cat(support_ac_parts, dim=0)  # Concatenate all chunks into a complete matrix


def compute_confidence_chunked(support_ac, support_a, chunk_size=1000, device="cuda"):
    N = support_ac.size(0)
    result_chunks = []
    support_ac = support_ac.to(device)
    support_a = support_a.to(device)

    for start in range(0, N, chunk_size):
        logger.info(f"confidence_Batch {start/chunk_size}")
        end = min(start + chunk_size, N)
        # Extract support_ac submatrix (rows) for current chunk
        ac_chunk = support_ac[start:end, :]  # [chunk_size, N]
        # Corresponding support_a subvector (broadcast division)
        a_chunk = support_a[start:end].view(-1, 1)  # [chunk_size, 1]
        # Calculate confidence and handle NaN
        confidence_chunk = torch.nan_to_num(ac_chunk / a_chunk, nan=0.0)
        # Can save immediately or collect results first
        logger.info(f"Calculate confidence_chunk device: {confidence_chunk.device}")
        
        result_chunks.append(confidence_chunk.cpu())  # Convert back to CPU first to save GPU memory
        del confidence_chunk, ac_chunk, a_chunk
        torch.cuda.empty_cache()
    # Merge all chunks
    confidence = torch.cat(result_chunks, dim=0)  # [N, N]
    return confidence


def calculate_lift_chunked(confidence: torch.Tensor, support_c: torch.Tensor, chunk_size: int = 1000, device='cuda'):
    n = confidence.size(0)
    results = []
    confidence = confidence.to(device)
    support_c = support_c.to(device).view(1, -1)
    for start in range(0, n, chunk_size):
        logger.info(f"lift_Batch {start/chunk_size}")
        end = min(start + chunk_size, n)

        conf_chunk = confidence[start:end, :]
        lift_chunk = torch.nan_to_num(conf_chunk / support_c.view(1, -1), nan=0.0)
        logger.info(f"Calculate lift_chunk device: {lift_chunk.device}")
        results.append(lift_chunk.cpu())
        del lift_chunk, conf_chunk
        torch.cuda.empty_cache()
    return torch.cat(results, dim=0)


def calculate_leverage_chunked(support_ac: torch.Tensor,support_a: torch.Tensor,support_c: torch.Tensor,chunk_size: int = 1000,device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    """
    Calculate leverage in chunks to avoid creating expected_support all at once and using too much GPU memory.
    """
    support_a = support_a.to(device)
    support_c = support_c.to(device)
    support_ac = support_ac.to(device)

    n_genes = support_a.shape[0]
    result_chunks = []

    for start in range(0, n_genes, chunk_size):
        logger.info(f"leverage_Batch {start/chunk_size}")
        end = min(start + chunk_size, n_genes)

        # support_a for current chunk ([chunk, 1])
        chunk_a = support_a[start:end].view(-1, 1)  # shape: [chunk_size, 1]
        # All support_c ([1, N])
        expected_chunk = chunk_a * support_c.view(1, -1)  # shape: [chunk_size, N]
        # support_ac submatrix for current chunk
        ac_chunk = support_ac[start:end, :]  # shape: [chunk_size, N]

        # leverage = support_ac - expected
        leverage_chunk = ac_chunk - expected_chunk
        logger.info(f"Calculate leverage_chunk device: {leverage_chunk.device}")
        result_chunks.append(leverage_chunk.cpu())  # Save to CPU to release GPU memory
        del leverage_chunk, expected_chunk, chunk_a, ac_chunk
        torch.cuda.empty_cache()
    # Concatenate all result chunks to get complete leverage matrix
    return torch.cat(result_chunks, dim=0)


def calculate_conviction_chunked(confidence: torch.Tensor, support_c: torch.Tensor, chunk_size: int = 1000,device='cuda') -> torch.Tensor:
    """
    Calculate conviction in chunks to avoid OOM.
    confidence: [N, N]
    support_c: [N]
    """
    n_genes = confidence.size(0)
    final_result = torch.empty((n_genes, n_genes), dtype=torch.float32)  # Pre-allocate directly on CPU

    confidence = confidence.to(device)
    support_c = support_c.view(1, -1)  # shape: [1, N]

    for start in range(0, n_genes, chunk_size):
        logger.info(f"Batch {start/chunk_size}")

        end = min(start + chunk_size, n_genes)
        conf_chunk = confidence[start:end, :]   # [chunk_size, N]

        # Calculate conviction for corresponding chunk
        denominator = torch.clamp(1 - conf_chunk, min=1e-6)
        numerator = 1 - support_c  # [1, N], broadcasted
        result = numerator / denominator
        
        # Replace conf=1 positions with inf
        result[conf_chunk == 1] = float("inf")
        logger.info(f"Calculate conviction_chunk device: {result.device}")
        # Write directly to pre-allocated matrix
        final_result[start:end, :] = result.cpu()
        del result, denominator, numerator, conf_chunk
        torch.cuda.empty_cache()
    return final_result  # shape: [N, N]


def compute_metrics(gene_names, support_a: torch.Tensor, support_ac: torch.Tensor, device: str = "cuda",
                    input_thresholds: dict = None,  # New parameter for passing threshold dictionary
                    chunk_size=1000, result_dir="results/results.csv"):

    n_genes = len(gene_names)
    support_ac = support_ac.to(device)
    support_a = support_a.to(device)

    # ----------------------
    # 1. Calculate various metrics (chunked)
    # ----------------------
    logger.info(f"Starting confidence metric calculation")
    confidence = compute_confidence_chunked(support_ac, support_a, chunk_size=chunk_size, device=device)
    logger.info(f"Save confidence device: {confidence.device}")

    logger.info(f"Starting lift metric calculation")
    lift = calculate_lift_chunked(confidence, support_a, chunk_size=chunk_size, device=device)
    logger.info(f"Save lift device: {lift.device}")

    logger.info(f"Starting leverage metric calculation")
    leverage = calculate_leverage_chunked(support_ac, support_a, support_a, chunk_size=chunk_size, device=device)
    logger.info(f"Save leverage device: {leverage.device}")

    logger.info(f"Starting conviction metric calculation")
    conviction = calculate_conviction_chunked(confidence, support_a, chunk_size=chunk_size, device=device)
    logger.info(f"Save conviction device: {conviction.device}")
    

    #save_matrix_as_csv(support_ac, gene_names, "other_data/support.csv")
    #save_matrix_as_csv(confidence, gene_names, "other_data/confidence.csv")
    #save_matrix_as_csv(lift, gene_names, "other_data/lift.csv")
    #save_matrix_as_csv(leverage, gene_names, "other_data/leverage.csv")
    #save_matrix_as_csv(conviction, gene_names, "other_data/conviction.csv")


    logger.info("Starting data saving")
    # ----------------------
    # 2. Unified mask filtering
    # ----------------------
    df = compute_metrics_with_any_after_full_calc(gene_names,
                                                  support_ac,confidence,lift,leverage,conviction,
                                                  support_th=input_thresholds['support'],
                                                  confidence_th=input_thresholds['confidence'],
                                                  lift_th=input_thresholds['lift'],
                                                  leverage_th=input_thresholds['leverage'],
                                                  conviction_th=input_thresholds['conviction'],
                                                  result_dir=result_dir)

    return df

def get_num_batches(num_cells: int, batch_size: int) -> int:
    return math.ceil(num_cells / batch_size)
def save_matrix_as_csv(matrix, gene_names, filename):
    # matrix: torch.Tensor，shape (n_genes, n_genes)
    df = pd.DataFrame(matrix.cpu().numpy(), index=gene_names, columns=gene_names)
    df.to_csv(filename)



def compute_in_chunks(
    gene_names, 
    support_a: torch.Tensor,
    support_ac_sum: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    input_thresholds: dict = None,
    chunk_size=1000,
    result_dir="output_dir",
) -> None:
    logger = logging.getLogger("scAR")
    logger.info(f"Starting calculation of other metrics besides support")

    # Directly call compute_metrics to calculate metrics
    result_df = compute_metrics(gene_names, support_a, support_ac_sum, device = device,
                                input_thresholds=input_thresholds,
                                chunk_size=chunk_size, result_dir = result_dir)
    return result_df

