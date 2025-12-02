import logging
import time
import os
from datetime import datetime
import psutil  # Used to monitor memory and CPU
import torch   # Check GPU status (if using GPU)

def setup_logger(name: str = "scAR", log_dir: str = "logs", run_id = "Nan",level=logging.INFO) -> logging.Logger:
    """
    Configure a logger that outputs to both console and file.
    
    Args:
        name (str): Logger name.
        log_dir (str): Directory to save log files.
        level (int): Logging level, e.g., logging.INFO or logging.DEBUG.
    
    Returns:
        logging.Logger: A configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    # Create log directory
    log_file = os.path.join(log_dir, f"{name}_{run_id}.log")
    # Create logger
    logger = logging.getLogger(name)
    # Set level
    logger.setLevel(level)
    # Clear existing handlers to avoid duplicate outputs
    logger.handlers = []
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    # File handler
    fh = logging.FileHandler(log_file)
    # Set level
    fh.setLevel(level)
    # Set format
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s'))
    # Add handlers
    logger.addHandler(ch)
    logger.addHandler(fh)
    # Return
    return logger


class ProgressMonitor:
    """
    Monitor progress, record logs, and measure elapsed time. Supports `with` usage.
    """

    def __init__(self, total: int, task_name: str = "Task", logger: logging.Logger = None,
                 show_resource: bool = True):
        """
        Initialize the monitor.

        Args:
            total (int): Total number of steps.
            task_name (str): Current task name.
            logger (Logger): Logger instance.
            show_resource (bool): Whether to show resource usage (CPU/GPU/Memory).
        """
        self.total = total
        self.task_name = task_name
        self.logger = logger or logging.getLogger("scAR")
        self.start_time = None
        self.show_resource = show_resource

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"🔄 Start task: {self.task_name} | {self.total} steps total")
        return self

    def update(self, current: int):
        """
        Update progress.

        Args:
            current (int): Number of steps completed.
        """
        elapsed = time.time() - self.start_time
        percent = current / self.total * 100
        self.logger.info(f"➡️ [{current}/{self.total}] ({percent:.1f}%) completed, elapsed {elapsed:.1f} s")

        if self.show_resource:
            self._log_resources()

    def _log_resources(self):
        """
        Log resource usage: CPU, memory, and GPU (if available).
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
        On exit, record total elapsed time; log errors if any.
        """
        elapsed = time.time() - self.start_time
        if exc_type is not None:
            self.logger.error(f"❌ Error: {exc_val}")
        self.logger.info(f"✅ Task finished: {self.task_name} | Total time: {elapsed:.1f} s")

