import logging
import time
import os
from datetime import datetime
import psutil  # For monitoring memory and CPU
import torch   # Check GPU status (if using GPU)

def setup_logger(name: str = "scAR", log_dir: str = "logs", run_id = "Nan",level=logging.INFO) -> logging.Logger:
    """
    Set up logger with support for both console and file output.
    
    Parameters:
        name (str): Logger name.
        log_dir (str): Log file save directory.
        level (int): Log level, e.g. logging.INFO or logging.DEBUG.
    
    Returns:
        logging.Logger: Configured logger.
    """
    os.makedirs(log_dir, exist_ok=True)
    # Create log save directory
    log_file = os.path.join(log_dir, f"{name}_{run_id}.log")
    # Create logger
    logger = logging.getLogger(name)
    # Set log level
    logger.setLevel(level)
    # Clear old handlers to avoid duplicate output
    logger.handlers = []
    # Create console output
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    # Create file output
    fh = logging.FileHandler(log_file)
    # Set log level
    fh.setLevel(level)
    # Set log format
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    # Add handler
    logger.addHandler(ch)
    logger.addHandler(fh)
    # Return logger
    return logger


class ProgressMonitor:
    """
    Class for monitoring progress, logging, and calculating time consumption, supports with usage.
    """

    def __init__(self, total: int, task_name: str = "Task", logger: logging.Logger = None,
                 show_resource: bool = True):
        """
        Initialize monitor.

        Parameters:
            total (int): Total number of tasks.
            task_name (str): Current task name.
            logger (Logger): Logger object.
            show_resource (bool): Whether to show resource usage (CPU/GPU/memory).
        """
        self.total = total
        self.task_name = task_name
        self.logger = logger or logging.getLogger("scAR")
        self.start_time = None
        self.show_resource = show_resource

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"🔄 Starting task: {self.task_name} | Total {self.total} steps")
        return self

    def update(self, current: int):
        """
        Update progress.

        Parameters:
            current (int): Current number of completed steps.
        """
        elapsed = time.time() - self.start_time
        percent = current / self.total * 100
        self.logger.info(f"➡️ [{current}/{self.total}] ({percent:.1f}%) completed, time used {elapsed:.1f} seconds")

        if self.show_resource:
            self._log_resources()

    def _log_resources(self):
        """
        Print resource usage information: CPU, memory, GPU (if available).
        """
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        msg = f"Resource usage: CPU: {cpu:.1f}%, Memory: {mem:.1f}%"

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
            msg += f", GPU: {gpu_mem:.1f}MB / {total_mem:.1f}MB"

        self.logger.info(msg)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Record total time consumption when exiting, also record errors if any.
        """
        elapsed = time.time() - self.start_time
        if exc_type is not None:
            self.logger.error(f"❌ Error: {exc_val}")
        self.logger.info(f"✅ Task completed: {self.task_name} | Total time: {elapsed:.1f} seconds")

'''
from scar.longger import ProgressMonitor

with ProgressMonitor(total=n_steps, task_name="Rule Mining") as longger:
    for i in range(n_steps):
        # Call your calculation function
        longger.update(i + 1)

logger = setup_logger("scAR-test")
'''