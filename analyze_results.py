
import csv
from collections import defaultdict

def load_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["size"] = int(r["size"])
            r["repeats"] = int(r["repeats"])
            r["avg_ms"] = float(r["avg_ms"])
            r["std_ms"] = float(r["std_ms"])
            r["mem_start_mb"] = float(r["mem_start_mb"])
            r["mem_end_mb"] = float(r["mem_end_mb"])
            r["logical_cores"] = int(r["logical_cores"])
            r["parallel_workers"] = int(r["parallel_workers"])
            rows.append(r)
    return rows


def main(csv_path="data/results_dense.csv"):
    rows = load_results(csv_path)

    by_size = defaultdict(list)
    for r in rows:
        by_size[r["size"]].append(r)

    for size in sorted(by_size.keys()):
        group = by_size[size]
        print(f"\n===== Size {size} x {size} =====")
        naive_time = None
        for r in group:
            if r["algo"] == "NAIVE":
                naive_time = r["avg_ms"]
                break
        if naive_time is None:
            print("No NAIVE baseline found, skipping.")
            continue

        print(f"NAIVE: {naive_time:.2f} ms (baseline)")

        print("algo        avg_ms   speedup   efficiency   mem_start  mem_end")  # noqa: E501
        for r in group:
            algo = r["algo"]
            t = r["avg_ms"]
            speedup = naive_time / t if t > 0 else 0.0
            eff_str = "-"
            if algo == "PARALLEL" and r["parallel_workers"] > 0:
                efficiency = speedup / r["parallel_workers"]
                eff_str = f"{efficiency:9.3f}"
            print(f"{algo:10s} {t:7.2f} {speedup:9.3f} {eff_str:>12s}   {r['mem_start_mb']:9.1f} {r['mem_end_mb']:9.1f}")  # noqa: E501


if __name__ == "__main__":
    main()
