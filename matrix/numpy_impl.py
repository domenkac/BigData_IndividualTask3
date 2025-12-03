
import numpy as np

def multiply_numpy(A, B):
    """Vectorized matrix multiplication using NumPy / BLAS (SIMD under the hood)."""
    a = np.array(A, dtype=float)
    b = np.array(B, dtype=float)
    c = a.dot(b)
    return c.tolist()
