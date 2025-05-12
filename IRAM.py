import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

def iram(matvec, n, k, which='LM', tol=1e-8, maxiter=None):
    """
    Compute k eigenpairs of an n×n operator using the Implicitly
    Restarted Arnoldi Method (via ARPACK).
    
    Parameters:
    - matvec(v): function that returns A @ v without requiring A explicitly
    - n: dimension of the operator
    - k: number of eigenvalues/vectors to compute
    - which: string specifying which eigenvalues ('LM', 'SM', etc.)
    - tol: solver tolerance
    - maxiter: maximum number of iterations
    
    Returns:
    - vals: computed eigenvalues
    - vecs: corresponding eigenvectors (as columns)
    """
    A_linop = LinearOperator((n, n), matvec=matvec)
    vals, vecs = eigs(A_linop, k=k, which=which, tol=tol, maxiter=maxiter)
    return vals, vecs

# Example usage
def example():
    n = 100
    # Define A implicitly: A is diagonal with entries 1..n
    def matvec(v):
        return np.arange(1, n+1) * v
    
    k = 5
    vals, vecs = iram(matvec, n, k)
    print("Computed eigenvalues:", np.real(vals))

if __name__ == '__main__':
    example()
