
from .utils import zeros

def multiply_unrolled(A, B):
    """Loop-unrolled version of matrix multiplication (unroll by factor 4 on inner loop)."""
    n = len(A)
    if n == 0:
        return []
    m = len(A[0])
    p = len(B[0])

    C = zeros(n, p)

    for i in range(n):
        ci = C[i]
        Ai = A[i]
        for k in range(m):
            aik = Ai[k]
            bk = B[k]
            j = 0
            # Unroll inner loop by 4
            while j + 3 < p:
                ci[j]     += aik * bk[j]
                ci[j + 1] += aik * bk[j + 1]
                ci[j + 2] += aik * bk[j + 2]
                ci[j + 3] += aik * bk[j + 3]
                j += 4
            # Handle remainder
            while j < p:
                ci[j] += aik * bk[j]
                j += 1
    return C
