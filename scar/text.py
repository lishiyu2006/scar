# text.py

from ntpath import supports_unicode_filenames
import os
import ctypes
import threading
import time
from datetime import datetime
from turtle import clear
import torch
import yaml
import logging
import config
from scar.synthetic import generate_expression_matrix
from scar.input import load_expression_matrix
from scar.engine import compute_in_chunks,calculate_support,calculate_support_ac_chunked,binarize_and_to_cuda,    save_matrix_as_csv
from scar.logger import setup_logger
from scar.config import load_config,get
from scar.pre_input import prepare_columns_for_loading



# This is the get_gpu_status function (can be placed in external module for import)
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
    logger = logging.getLogger("scAR")  # Re-acquire the same logger instance here
    try:
        gpus = get_gpu_status()
        for gpu in gpus:
            logger.info(
                f"GPU {gpu['index']} {gpu['name']}: "
                f"GPU Memory {gpu['memory_used_MB']}MB / {gpu['memory_total_MB']}MB, "
                f"Utilization {gpu['gpu_utilization_percent']}%"
            )
    except Exception as e:
        logger.warning(f"Failed to get GPU status: {e}")

class GPUMonitor:
    def __init__(self, gpu_index=0, interval=0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self.max_util = 0
        self.max_mem = 0
        self._running = False
        self._thread = None

    def _monitor(self):
        while self._running:
            try:
                # Call nvidia-smi to get utilization and memory
                cmd = [
                    "nvidia-smi",
                    f"--query-gpu=utilization.gpu,memory.used",
                    f"--format=csv,nounits,noheader",
                    f"-i {self.gpu_index}"
                ]
                result = subprocess.check_output(" ".join(cmd), shell=True).decode().strip()
                util_str, mem_str = result.split(", ")
                util = int(util_str)
                mem = int(mem_str)

                if util > self.max_util:
                    self.max_util = util
                if mem > self.max_mem:
                    self.max_mem = mem

            except Exception as e:
                pass
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def get_max(self):
        return self.max_util, self.max_mem


def main():
    #gpu_monitor = GPUMonitor(gpu_index=0, interval=0.05)
    #gpu_monitor.start()
    # Generate run ID at the beginning of main function
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Output file name
    logger = setup_logger(name="scAR", log_dir="logs",run_id = run_id)  # Call the previous log configuration function
    output_file = f"results/results_{run_id}.csv"
    logger.info(f"{output_file}") 


    config = load_config("scar/config.yaml") 
    # Load configuration

    batch_size = config.get("batch_size", 2000)
    input_path = config.get("input_path", "data/synthetic_expression.csv")
    use_gpu = config.get("use_gpu", True)
    device = config.get("device", "cuda:0" if use_gpu else "cpu")
    chunk_size = config.get("chunk_size", 1000)
    n_genes = config.get("n_gene",)
    clear_time = config.get("clear",20)

    thresholds = config.get("thresholds", {})
    support_th = thresholds.get("support", 0.0)
    confidence_th = thresholds.get("confidence", 0.0)
    lift_th = thresholds.get("lift", 0.0)
    leverage_th = thresholds.get("leverage", 0.0)
    conviction_th = thresholds.get("conviction", 0.0)

    print(f"input_path: {input_path}")
    print(f"batch_size: {batch_size}, chunk_size: {chunk_size}")
    print(f"device: {device}, n_genes: {n_genes}, clear_time: {clear_time}")
    print(f"Filter values: support_th: {support_th}, confidence_th: {confidence_th}, lift_th: {lift_th} leverage_th: {leverage_th}, conviction_th: {conviction_th}")

    logger.info("===== Starting single-cell data analysis =====")  # General progress information
    # 1. Generate simulated data
    gene_names = []
    n_cells = 0
    support_a_sum = None 
    support_ac_sum = None
    if not os.path.exists(input_path):
        
        df, csv_path = generate_expression_matrix(n_genes=3, n_cells=5, save_path=input_path)
        logger.info(f"No expression matrix file detected, starting to generate simulated data and save to {csv_path}")

        for idx, batch_df in enumerate(load_expression_matrix(csv_path, batch_size=batch_size)):
            logger.info(f"[Batch {idx}] Input matrix shape: {batch_df.shape}")
            #df.index gets the DataFrame index
            #.tolist() method converts index object to Python list
            gene_names = batch_df.index.tolist()
            n_cells += batch_df.shape[1]
            try:
                # ➤ Step 1: Binarize and transfer to GPU
                batch_tensor = binarize_and_to_cuda(batch_df, device=device)
                logger.info(f"Binarization completed")
                # ➤ Step 2: Calculate support_a (expression ratio for each gene), transfer back to CPU for accumulation
                # For current batch:
                
                support_a = calculate_support(batch_tensor)  # support_a is 1D Tensor with length n_genes

                if support_a_sum is None:
                    support_a_sum = support_a.clone()  # Initialize accumulator
                else:
                    support_a_sum += support_a  # Element-wise accumulation

                n_cells += batch_df.shape[1]  # Accumulate cell count

                logger.info(f"[support_a_Batch {idx}] Metric calculation completed")

                # ➤ Step 3: Calculate support_ac in chunks (avoid OOM), transfer back to CPU for accumulation
                support_ac = calculate_support_ac_chunked(batch_tensor, chunk_size=chunk_size, device=device).cpu()

                if support_ac_sum is None:
                    support_ac_sum = support_ac
                else:
                    support_ac_sum += support_ac

                logger.info(f"[support_ac_Batch {idx}] Metric calculation completed")
                
                # ➤ Step 4: Actively release GPU memory
                del batch_tensor, support_ac
                if idx %  clear_time == 0:
                    torch.cuda.empty_cache()
                    logger.info("Clearing GPU memory")
                    

            except Exception as e:
                logger.error(f"[support_Batch {idx}] Analysis failed: {type(e).__name__}: {e}")
                continue  # Continue to next batch on error
        support_ac_sum .cpu()
        support_a_sum .cpu()
        support_ac_sum /= n_cells
        support_a_sum /= n_cells
        try:
            compute_in_chunks(gene_names, support_a_sum,support_ac_sum,device,
            thresholds,
            chunk_size,output_file)   # Your analysis function
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")  # Record error information
            return  # Exit on error
        logger.info("All metric calculations completed")
    else:
        logger.info(f"Detected existing expression matrix file, loading {input_path}")

        for idx, batch_df in enumerate(load_expression_matrix(input_path, batch_size=batch_size)):
            logger.info(f"[Batch {idx}] Input matrix shape: {batch_df.shape}")
            #df.index gets the DataFrame index
            #.tolist() method converts index object to Python list
            gene_names = batch_df.index.tolist()
            n_cells += batch_df.shape[1]
            try:
                # ➤ Step 1: Binarize and transfer to GPU
                batch_tensor = binarize_and_to_cuda(batch_df)
                
                # ➤ Step 2: Calculate support_a (expression ratio for each gene), transfer back to CPU for accumulation
                # For current batch:
                support_a = calculate_support(batch_tensor)  # support_a is 1D Tensor with length n_genes

                if support_a_sum is None:
                    support_a_sum = torch.zeros_like(support_a)
                    support_a_sum = support_a.clone()  # Initialize accumulator
                else:
                    support_a_sum += support_a  # Element-wise accumulation

                # ➤ Step 3: Calculate support_ac in chunks (avoid OOM), transfer back to CPU for accumulation
                support_ac = calculate_support_ac_chunked(batch_tensor, chunk_size=chunk_size, device=device)

                if support_ac_sum is None:
                    support_ac_sum = torch.zeros_like(support_ac)
                    support_ac_sum = support_ac
                else:
                    support_ac_sum += support_ac

                logger.info(f"[Batch {idx}] Metric calculation completed")
                
                # ➤ Step 4: Actively release GPU memory
                del batch_tensor, support_ac
                if idx %  clear_time == 0:
                    torch.cuda.empty_cache()
                    logger.info("Clearing GPU memory")

            except Exception as e:
                logger.error(f"[Batch {idx}] Analysis failed: {e}")
                continue  # Continue to next batch on error
        support_ac_sum .cpu()
        support_a_sum .cpu()

        try:
            compute_in_chunks(gene_names, support_a_sum,support_ac_sum,device,
            thresholds ,
            chunk_size,output_file)   # Your analysis function
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")  # Record error information
            return  # Exit on error
        logger.info("All metric calculations completed")

    #gpu_monitor.stop()
    #max_util, max_mem = gpu_monitor.get_max()

    #print(f"Maximum GPU utilization: {max_util}%")
    #print(f"Maximum GPU memory usage: {max_mem} MB")
    
    # Print GPU status after calculation
    log_gpu_status()



if __name__ == "__main__":
    main()