
import argparse
import csv
import math
import os
import random
import time
import psutil
import multiprocessing

from matrix.naive import multiply_naive
from matrix.blocked import multiply_blocked
from matrix.unrolled import multiply_unrolled
from matrix.numpy_impl import multiply_numpy
from matrix.parallel import multiply_parallel

ALGOS = ["NAIVE", "BLOCKED", "UNROLLED", "NUMPY", "PARALLEL"]


def gen_random_matrix(n, m):
    """Generate n x m matrix of random floats in [0,1)."""
    return [[random.random() for _ in range(m)] for _ in range(n)]


def get_memory_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def measure_once(algo, A, B, block_size, parallel_workers):
    """Run one timing for a given algorithm, returning (ms, mem_start, mem_end)."""
    mem_start = get_memory_mb()
    t0 = time.perf_counter()

    if algo == "NAIVE":
        multiply_naive(A, B)
    elif algo == "BLOCKED":
        multiply_blocked(A, B, block_size)
    elif algo == "UNROLLED":
        multiply_unrolled(A, B)
    elif algo == "NUMPY":
        multiply_numpy(A, B)
    elif algo == "PARALLEL":
        multiply_parallel(A, B, num_workers=parallel_workers)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    t1 = time.perf_counter()
    mem_end = get_memory_mb()
    elapsed_ms = (t1 - t0) * 1000.0
    return elapsed_ms, mem_start, mem_end


def run_benchmarks(sizes, out_path, repeats, block_size, parallel_workers):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = [
        "algo", "size", "repeats",
        "avg_ms", "std_ms",
        "mem_start_mb", "mem_end_mb",
        "logical_cores", "parallel_workers",
    ]

    logical_cores = multiprocessing.cpu_count()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n in sizes:
            print(f"\n=== Size: {n} x {n} ===")
            A = gen_random_matrix(n, n)
            B = gen_random_matrix(n, n)

            for algo in ALGOS:
                print(f"Running {algo} ...", end="", flush=True)
                times = []
                mem_starts = []
                mem_ends = []

                for _ in range(repeats):
                    t_ms, mem_s, mem_e = measure_once(algo, A, B, block_size, parallel_workers)
                    times.append(t_ms)
                    mem_starts.append(mem_s)
                    mem_ends.append(mem_e)

                avg_ms = sum(times) / len(times)
                if len(times) > 1:
                    var = sum((t - avg_ms) ** 2 for t in times) / (len(times) - 1)
                    std_ms = math.sqrt(var)
                else:
                    std_ms = 0.0

                mem_start_avg = sum(mem_starts) / len(mem_starts)
                mem_end_avg = sum(mem_ends) / len(mem_ends)

                print(f"  avg = {avg_ms:.2f} ms, std = {std_ms:.2f} ms, mem ~ {mem_start_avg:.1f}->{mem_end_avg:.1f} MB")  # noqa: E501

                writer.writerow({
                    "algo": algo,
                    "size": n,
                    "repeats": repeats,
                    "avg_ms": f"{avg_ms:.6f}",
                    "std_ms": f"{std_ms:.6f}",
                    "mem_start_mb": f"{mem_start_avg:.3f}",
                    "mem_end_mb": f"{mem_end_avg:.3f}",
                    "logical_cores": logical_cores,
                    "parallel_workers": parallel_workers if algo == "PARALLEL" else 1,
                })


def parse_sizes(s):
    return [int(x) for x in s.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense matrix multiplication benchmark (parallel + vectorized)")  # noqa: E501
    parser.add_argument("--sizes", default="200,400,600", help="Comma-separated matrix sizes (n for n x n)")  # noqa: E501
    parser.add_argument("--out", default="data/results_dense.csv", help="Output CSV path")  # noqa: E501
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per algorithm")  # noqa: E501
    parser.add_argument("--block-size", type=int, default=64, help="Block size for BLOCKED algorithm")  # noqa: E501
    parser.add_argument("--parallel-workers", type=int, default=4, help="Number of workers for PARALLEL algorithm")  # noqa: E501

    args = parser.parse_args()
    sizes = parse_sizes(args.sizes)

    print("Benchmarking algorithms:", ", ".join(ALGOS))
    run_benchmarks(sizes, args.out, args.repeats, args.block_size, args.parallel_workers)
