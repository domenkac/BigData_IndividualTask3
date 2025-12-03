
from .utils import zeros

def multiply_naive(A, B):
    """Baseline O(n^3) triple-loop matrix multiplication.

    A: n x m
    B: m x p
    Returns C = A * B (n x p)
    """
    n = len(A)
    if n == 0:
        return []
    m = len(A[0])
    p = len(B[0])

    C = zeros(n, p)
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            for j in range(p):
                C[i][j] += aik * B[k][j]
    return C
