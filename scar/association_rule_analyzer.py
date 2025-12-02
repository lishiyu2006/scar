from ntpath import supports_unicode_filenames
import os
import ctypes
import pandas as pd
from datetime import datetime
import torch
import logging
from scar.data_loader import load_expression_matrix,split_mtx_by_celltype_batches,split_df_by_columns,input_with_timeout
from scar.computation_engine import compute_in_chunks,calculate_support,calculate_support_ac_chunked,binarize_and_to_cuda,    save_matrix_as_csv
from scar.logging_system import setup_logger
from scar.configuration_manager import load_config



# get_gpu_status function (can be imported from an external module)
def get_gpu_status(nvml_path=r"C:\Windows\System32\nvml.dll"):
    nvml = ctypes.CDLL(nvml_path)

    ret = nvml.nvmlInit_v2()
    if ret != 0:
        raise RuntimeError(f"nvmlInit_v2 initialization failed, error code: {ret}")

    device_count = ctypes.c_uint()
    ret = nvml.nvmlDeviceGetCount_v2(ctypes.byref(device_count))
    if ret != 0:
        raise RuntimeError(f"nvmlDeviceGetCount_v2 failed, error code: {ret}")

    class NVMLMemory(ctypes.Structure):
        _fields_ = [
            ('total', ctypes.c_ulonglong),
            ('free', ctypes.c_ulonglong),
            ('used', ctypes.c_ulonglong),
        ]

    gpu_list = []
    for i in range(device_count.value):
        handle = ctypes.c_void_p()
        ret = nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(handle))
        if ret != 0:
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex_v2 failed, GPU index {i}, error code: {ret}")

        name = ctypes.create_string_buffer(100)
        ret = nvml.nvmlDeviceGetName(handle, name, ctypes.c_uint(100))
        if ret != 0:
            raise RuntimeError(f"nvmlDeviceGetName failed, GPU index {i}, error code: {ret}")

        mem_info = NVMLMemory()
        ret = nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem_info))
        if ret != 0:
            raise RuntimeError(f"nvmlDeviceGetMemoryInfo failed, GPU index {i}, error code: {ret}")

        util = ctypes.c_uint()
        ret = nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
        if ret != 0:
            raise RuntimeError(f"nvmlDeviceGetUtilizationRates failed, GPU index {i}, error code: {ret}")

        gpu_list.append({
            "index": i,
            "name": name.value.decode(),
            "memory_total_MB": mem_info.total // (1024 ** 2),
            "memory_used_MB": mem_info.used // (1024 ** 2),
            "memory_free_MB": mem_info.free // (1024 ** 2),
            "gpu_utilization_percent": util.value,
        })

    nvml.nvmlShutdown()
    return gpu_list

def log_gpu_status():
    logger = logging.getLogger("scAR")  # Reuse the same logger instance
    try:
        gpus = get_gpu_status()
        for gpu in gpus:
            logger.info(
                f"GPU {gpu['index']} {gpu['name']}: "
                f"Memory {gpu['memory_used_MB']}MB / {gpu['memory_total_MB']}MB, "
                f"Utilization {gpu['gpu_utilization_percent']}%"
            )
    except Exception as e:
        logger.warning(f"Failed to get GPU status: {e}")


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Output filename
    logger = setup_logger(name="scAR", log_dir="logs",run_id = run_id)  # Use the existing logging setup function


    config = load_config("scar/project_settings.yaml")


    # Load configuration

    batch_size = config.get("batch_size", 2000)
    input_path = config.get("main_input_path", "data/pbmc/matrix.mtx")
    use_gpu = config.get("use_gpu", True)
    device = config.get("device", "cuda:0" if use_gpu else "cpu")
    chunk_size = config.get("chunk_size", 1000)
    n_genes = config.get("n_gene",)
    clear_time = config.get("clear",20)

    output_dir = config.get("main_output_dir", "results")


    # Normalize path separators (optional but recommended)
    input_dir = input_path.replace("/", os.sep).replace("\\", os.sep)

    # Extract the first folder name after 'data/'
    if input_dir.startswith("data" + os.sep):
        relative_path = input_dir[len("data" + os.sep):] # Get "data/" after the part
        parts = relative_path.split(os.sep)
        if len(parts) > 1:
            intermediate_folder = parts[0] # Get first folder name
            output_dir = os.path.join(output_dir, intermediate_folder)
        # If only a file name remains, keep output_dir unchanged
        

    logger.info(f"{output_dir}") 

    thresholds = config.get("thresholds", {})
    logger.info("===== Starting single-cell analysis =====")  # General progress info
    
    logger.info("Loading expression matrix...")  # Inform the user that data is loading now
    data_dir = input_with_timeout("Enter directory or file path for expression matrix (e.g., data or data/...csv): ", timeout=20, default=input_path)
    
    # If MTX format and a cell type file exists
    if data_dir.endswith(".mtx"):
        base_dir = os.path.dirname(data_dir)
        celltype_path = os.path.join(base_dir, "cell_type.tsv")
        if os.path.exists(celltype_path):
            logger.info(f"Detected MTX format with cell type file; splitting analysis by cell type")
            batch_size = 2000
            batches = list(split_mtx_by_celltype_batches(data_dir, celltype_path, batch_size))
            logger.info(f"Detected {len(batches)} cell type batches")
            for batch_data in batches:
                cell_type = batch_data['cell_type']
                expr = batch_data['expr']
                genes = batch_data['genes']
                barcodes = batch_data['barcodes']

                logger.info(f"Processing cell type {cell_type}, cells: {len(barcodes)}, matrix shape: {expr.shape}")

                # Convert to a format compatible with analysis functions (e.g., DataFrame)
                df = pd.DataFrame.sparse.from_spmatrix(expr, index=genes, columns=barcodes)

                # Revised batch loop analysis logic
                n_cells = len(barcodes)
                support_a_sum = None
                support_ac_sum = None
                clear_time = 10  # Clear GPU memory every 10 batches

                for idx, batch_df in enumerate(split_df_by_columns(df, batch_size=batch_size)):
                    logger.info(f"[{cell_type} Batch {idx}] Input matrix shape: {batch_df.shape}")
                    gene_names = batch_df.index.tolist()

                    try:
                        batch_tensor = binarize_and_to_cuda(batch_df)

                        support_a = calculate_support(batch_tensor).cpu()
                        if support_a_sum is None:
                            support_a_sum = support_a.clone()
                        else:
                            support_a_sum += support_a

                        support_ac = calculate_support_ac_chunked(batch_tensor, chunk_size=chunk_size, device=device).cpu()
                        if support_ac_sum is None:
                            support_ac_sum = support_ac.clone()
                        else:
                            support_ac_sum += support_ac

                        logger.info(f"[{cell_type} Batch {idx}] Metrics computed")

                        del batch_tensor, support_ac, support_a
                        if idx % clear_time == 0:
                            torch.cuda.empty_cache()
                            logger.info(f"[{cell_type} Batch {idx}] Cleared GPU memory")

                    except Exception as e:
                        logger.error(f"[{cell_type} Batch {idx}] Analysis failed: {type(e).__name__}: {e}")
                        continue

                #support_a_sum.cpu()
                #support_ac_sum.cpu()

                try:
                    results = compute_in_chunks(gene_names, support_a_sum, support_ac_sum, device,
                                            thresholds, chunk_size, result_dir=f"{output_dir}/results_{cell_type}.csv")
                except Exception as e:
                    logger.error(f"{cell_type} computation failed: {str(e)}")
                    continue
                logger.info(f"{cell_type} metrics computed")
                log_gpu_status()
                logger.info(f"Results saved to results_{cell_type}.csv")

            logger.info("===== All cell types analysis completed =====")  # End message
            return

    
    
    logger.info(f"Detected existing expression matrix file, loading {input_path}")
    output_file = f"{output_dir}/results_{run_id}.csv"
    logger.info(f"{output_file}") 

    gene_names = []
    n_cells = 0
    support_a_sum = None 
    support_ac_sum = None

    for idx, batch_df in enumerate(load_expression_matrix(input_path, batch_size=batch_size)):
        logger.info(f"[Batch {idx}] Input matrix shape: {batch_df.shape}")
        #df.index retrieves the DataFrame index
        #.tolist() converts the index object to a Python list
        gene_names = batch_df.index.tolist()
        n_cells += batch_df.shape[1]
        try:
            # ➤ Step 1: Binarize and move to GPU
            batch_tensor = binarize_and_to_cuda(batch_df)
            
            # ➤ Step 2: Compute support_a (gene expression proportion) and accumulate on CPU
            # For current batch:
            support_a = calculate_support(batch_tensor)  # support_a is a 1D Tensor of length n_genes

            if support_a_sum is None:
                support_a_sum = torch.zeros_like(support_a)
                support_a_sum = support_a.clone()  # Initialize accumulator
            else:
                support_a_sum += support_a  # Elementwise accumulation

            # ➤ Step 3: Compute support_ac in chunks (avoid OOM) and accumulate on CPU
            support_ac = calculate_support_ac_chunked(batch_tensor, chunk_size=chunk_size, device=device)

            if support_ac_sum is None:
                support_ac_sum = torch.zeros_like(support_ac)
                support_ac_sum = support_ac
            else:
                support_ac_sum += support_ac

            logger.info(f"[Batch {idx}] Metrics computed")
            
            # ➤ Step 4: Actively free GPU memory
            del batch_tensor, support_ac
            if idx %  clear_time == 0:
                torch.cuda.empty_cache()
                logger.info("Cleared GPU memory")

        except Exception as e:
            logger.error(f"[Batch {idx}] Analysis failed: {e}")
            continue  # On error, continue to next batch
    #support_ac_sum .cpu()
    #support_a_sum .cpu()

    try:
        compute_in_chunks(
            gene_names,
            support_a_sum,
            support_ac_sum,
            device,
            thresholds,
            chunk_size,
            output_file
        )  # Analysis function
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return  # Exit on error
    logger.info("All metrics computed")
    # Print GPU status after computation
    log_gpu_status()
    
    logger.info("===== Analysis completed, results saved =====")



if __name__ == "__main__":
    main()