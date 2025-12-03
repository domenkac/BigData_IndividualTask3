
# Parallel and Vectorized Matrix Multiplication (Assignment Project)

This project implements and benchmarks several matrix multiplication algorithms:

- **NAIVE** – baseline triple-loop algorithm (no optimizations)
- **BLOCKED** – cache-friendly blocked/tiled multiplication
- **UNROLLED** – loop-unrolled version (micro-optimization, more compiler-friendly)
- **NUMPY** – vectorized multiplication using NumPy / BLAS (uses SIMD internally)
- **PARALLEL** – parallel matrix multiplication using Python's `multiprocessing`

It is designed to satisfy the assignment:

> Parallel (and Vectorized) Matrix Multiplication  
> - Implement a parallel version of matrix multiplication.  
> - Test with large matrices and analyze the performance gain from vectorization and parallelization.  
> - OPTIONAL: Implement a vectorized version of matrix multiplication.  
> - OPTIONAL: Compare both approaches with the basic matrix multiplication algorithm.  
> - Metrics: Speedup, efficiency, resource usage.

## 1. Installation

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Running the Benchmark

Run the benchmark script, for example:

```bash
python benchmark_dense.py --sizes 300,600,900 --repeats 3 --parallel-workers 4 --out data/results_dense.csv
```

Arguments:

- `--sizes` – comma-separated list of `n` for `n x n` matrices
- `--repeats` – how many times to run each algorithm per size
- `--parallel-workers` – number of worker processes used by the PARALLEL algorithm
- `--out` – CSV file where results are stored

The script prints timing and memory information to the console and writes detailed results to the CSV.

## 3. Analyzing Speedup and Efficiency

After running the benchmark, analyze results with:

```bash
python analyze_results.py
```

This script reads `data/results_dense.csv` and prints, for each matrix size:

- **Speedup** of each algorithm relative to `NAIVE`
- **Efficiency** of the `PARALLEL` algorithm (speedup / workers)
- **Memory usage** (average start and end in MB)

You can copy these numbers into your report or plot them in Excel / another tool.

## 4. Files Overview

- `matrix/naive.py` – basic matrix multiplication
- `matrix/blocked.py` – blocked multiplication
- `matrix/unrolled.py` – unrolled multiplication
- `matrix/numpy_impl.py` – NumPy (vectorized) implementation
- `matrix/parallel.py` – parallel implementation using `multiprocessing`
- `benchmark_dense.py` – runs benchmarks and records metrics
- `analyze_results.py` – computes speedup and efficiency from CSV
- `data/` – output directory for result CSV files

