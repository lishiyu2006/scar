import os
import torch
import csv
import logging
logger = logging.getLogger("scAR")

def compute_metrics_with_any_after_full_calc(
    gene_names,
    support_ac,
    confidence,
    lift,
    leverage,
    conviction,
    device="cuda",
    support_th=0.0,
    confidence_th=0.0,
    lift_th=0.0,
    leverage_th=0.0,
    conviction_th=0.0,
    result_dir="results",
    chunk_size=1000
):

    logger.info(f"Filter values support_th={support_th} confidence_th={confidence_th} lift_th={lift_th} leverage_th={leverage_th} conviction_th={conviction_th}")
    n_genes = len(gene_names)
    thresholds = torch.tensor(
        [support_th, confidence_th, lift_th, leverage_th, conviction_th],
        device=device
    )

    os.makedirs(os.path.dirname(result_dir), exist_ok=True)

    with open(result_dir, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Gene1", "Gene2",
            "support", "confidence", "lift", "leverage", "conviction",
            "Gene1_index", "Gene2_index"
        ])

        logger.info("Starting chunked filtering and saving results")
        for start in range(0, n_genes, chunk_size):
            end = min(start + chunk_size, n_genes)
            logger.info(f"Processing gene row chunk {start}:{end}")

            # Only take current chunk rows to save memory
            blocks = [
                support_ac[start:end, :].to(device),
                confidence[start:end, :].to(device),
                lift[start:end, :].to(device),
                leverage[start:end, :].to(device),
                conviction[start:end, :].to(device)
            ]

            # Print range
            for name, mat in zip(["support", "confidence", "lift", "leverage", "conviction"], blocks):
                logger.info(f"{name} min={mat.min().item():.5f}, max={mat.max().item():.5f}")

            # Mask
            mask_block = torch.ones_like(blocks[0], dtype=torch.bool, device=device)
            for mat, th in zip(blocks, thresholds):
                mask_block &= (mat >= th)

            # Global row & column numbers for current chunk
            global_rows = torch.arange(start, end, device=device).unsqueeze(1).expand(-1, mask_block.size(1))
            global_cols = torch.arange(0, n_genes, device=device).unsqueeze(0).expand(mask_block.size(0), -1)

            # Remove diagonal
            mask_block &= (global_rows != global_cols)

            # Find coordinates that meet conditions
            coords = mask_block.nonzero(as_tuple=False).cpu()
            if coords.numel() == 0:
                del blocks, mask_block
                torch.cuda.empty_cache()
                continue

            rows_idx = coords[:, 0] + start  # Global row number
            cols_idx = coords[:, 1]

            # Extract metric values from GPU (chunked)
            values = torch.stack([m.cpu()[coords[:, 0], coords[:, 1]] for m in blocks], dim=1)

            # Write to CSV
            for i in range(len(rows_idx)):
                if rows_idx[i].item() == cols_idx[i].item():
                    continue
                writer.writerow([
                    gene_names[rows_idx[i].item()],
                    gene_names[cols_idx[i].item()],
                    *values[i].tolist(),
                    rows_idx[i].item(),
                    cols_idx[i].item()
                ])

            logger.info(f"Number of elements in current chunk that meet conditions: {len(rows_idx)}")

            # Release GPU memory
            del blocks, mask_block, values, coords
            torch.cuda.empty_cache()

    logger.info(f"Results saved to {result_dir}")
