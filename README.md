# scAR — Single-cell Association Rules Toolkit

## Overview

scAR (Single-cell Association Rules) is a toolkit for analyzing single-cell RNA-seq data to compute association metrics between genes. It efficiently handles large datasets, supports 10x Genomics input formats, and offers optional GPU acceleration.

## Features

- GPU acceleration using CUDA to speed up computation
- Association rules: Support, Confidence, Lift, Leverage, Conviction
- Memory optimization via chunked processing to avoid OOM
- Flexible configuration via a YAML settings file

## Project Structure

- `association_rule_analyzer.py`: Main entry for association rule analysis
- `scar/computation_engine.py`: Core computation engine for association metrics
- `scar/data_loader.py`: Data input handling for supported formats
- `scar/configuration_manager.py`: Configuration management utilities
- `scar/logging_system.py`: Logging utilities
- `scar/result_writer.py`: Result output utilities
- Additional analyzers:
  - `differential_expression_analyzer.py`
  - `functional_enrichment_analyzer.py`
  - `pathway_enrichment_analyzer.py`
  - `protein_interaction_analyzer.py`
  - `Community_Detection.py`
  - `transcription_factor_analyzer.py`

## Metrics

1. Support — co-expression frequency
2. Confidence — rule reliability
3. Lift — improvement over independence
4. Leverage — difference from expected frequency
5. Conviction — strength of implication

## Installation

### Requirements

- Python 3.9+
- PyTorch (CUDA optional)
- `numpy`, `pandas`, `scipy`
- `scanpy` (for some analyses)
- `gseapy` (for enrichment analyses)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd scar

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Settings are defined in `scar/project_settings.yaml`. Key options include:

### I/O

```yaml
main_input_path: "data/pbmc/matrix.mtx"  # input expression matrix path
main_output_dir: "results"               # output directory
save_intermediate: false                  # whether to persist intermediate files
```

### Compute Parameters

```yaml
batch_size: 2000   # number of cells per input batch
use_gpu: true      # enable GPU if available
chunk_size: 1000   # computation chunk size
```

### Thresholds (example)

```yaml
thresholds:
  support: 0.01
  confidence: 0.6
  lift: 1.1
  leverage: 0.001
  conviction: 1.4
```

## Usage

### Association Rules

```bash
python association_rule_analyzer.py
```

- Input: 10x MTX with `genes.tsv`, `barcodes.tsv`, and optional `cell_type.tsv`
- Output: results in `results/` with timestamped CSVs and logs

### Other Modules

- Differential expression: `python differential_expression_analyzer.py`
- Functional enrichment: `python functional_enrichment_analyzer.py`
  - Uses `top_n` to filter and save top-ranked significant terms
- Pathway enrichment: `python pathway_enrichment_analyzer.py`
- Protein interaction analysis: `python protein_interaction_analyzer.py`
- Community detection: `python Community_Detection.py`
- Transcription factor analysis: `python transcription_factor_analyzer.py`

## Data Format

- Expression matrix: genes as rows, cells as columns
- Supported: MTX (+ `genes.tsv`, `barcodes.tsv`), H5AD, CSV/TSV

## Core Algorithm

### Chunked Strategy

- Split input by cell batches to control memory
- Accumulate metric results across chunks
- Periodically clear GPU caches to avoid OOM

### GPU Optimization

- PyTorch CUDA tensors for parallelism
- Boolean tensors to reduce memory footprint
- Chunked matrix ops to fit device memory

## Outputs

- `results_{timestamp}.csv`: primary association rule results
- Logs for analysis process and performance
- GPU status reports (if GPU is used)

## Troubleshooting

- Out of memory: lower `batch_size` and `chunk_size`
- GPU unavailable: check CUDA and PyTorch installation
- Data format errors: verify input files and paths

View logs:

```bash
tail -f logs/scAR.log
```

## Contributing

Issues and pull requests are welcome.

## License

To be added.

## Contact

For questions, please contact the maintainers.