
from multiprocessing import Pool, cpu_count
from .utils import zeros

def _multiply_chunk(args):
    """Worker: compute rows [row_start, row_end) of C = A * B."""
    A, B, row_start, row_end = args
    m = len(A[0])
    p = len(B[0])
    rows = row_end - row_start

    C_chunk = zeros(rows, p)

    for local_i, i in enumerate(range(row_start, row_end)):
        ci = C_chunk[local_i]
        Ai = A[i]
        for k in range(m):
            aik = Ai[k]
            bk = B[k]
            for j in range(p):
                ci[j] += aik * bk[j]
    return row_start, C_chunk


def multiply_parallel(A, B, num_workers=None, chunk_size=32):
    """Parallel matrix multiplication using process pool.

    Splits rows of A into chunks and distributes them across worker processes.
    """
    n = len(A)
    if n == 0:
        return []
    p = len(B[0])

    if num_workers is None:
        num_workers = cpu_count()

    C = zeros(n, p)

    tasks = []
    for row_start in range(0, n, chunk_size):
        row_end = min(row_start + chunk_size, n)
        tasks.append((A, B, row_start, row_end))

    with Pool(processes=num_workers) as pool:
        results = pool.map(_multiply_chunk, tasks)

    # Assemble result matrix in correct order
    for row_start, C_chunk in results:
        for offset, row in enumerate(C_chunk):
            C[row_start + offset] = row

    return C
