
from .utils import zeros

def multiply_blocked(A, B, block_size=64):
    """Cache-friendly blocked (tiled) matrix multiplication."""
    n = len(A)
    if n == 0:
        return []
    m = len(A[0])
    p = len(B[0])

    C = zeros(n, p)

    for ii in range(0, n, block_size):
        for kk in range(0, m, block_size):
            for jj in range(0, p, block_size):
                i_max = min(ii + block_size, n)
                k_max = min(kk + block_size, m)
                j_max = min(jj + block_size, p)
                for i in range(ii, i_max):
                    ci = C[i]
                    Ai = A[i]
                    for k in range(kk, k_max):
                        aik = Ai[k]
                        bk = B[k]
                        for j in range(jj, j_max):
                            ci[j] += aik * bk[j]
    return C
