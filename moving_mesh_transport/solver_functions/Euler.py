import numpy as np
from numba import njit
# @njit
def backward_euler(f, ts, y0, jac=None, tol=1e-8, maxiter=50):
    """
    Solve y' = f(t,y) with the backward‐Euler method on time‐grid ts.

    Parameters
    ----------
    f    : callable
           f(t, y) -> dy/dt   (returns a length-m array)
    ts   : array_like, shape (N,)
           Strictly increasing sequence of time‐points t0 < t1 < ... < t_{N-1}
    y0   : array_like, shape (m,)
           Initial condition at t0
    jac  : callable, optional
           jac(t,y) -> df/dy  (returns m×m Jacobian).  If None, 
           a finite‐difference jacobian is used.
    tol      : float, optional
               Convergence tolerance for Newton’s method (on the residual)
    maxiter  : int,   optional
               Maximum Newton iterations per step

    Returns
    -------
    Y : ndarray, shape (N, m)
        Solution values at each ts[i]
    """
    ts = np.asarray(ts)
    m  = y0.size
    Y  = np.zeros((m, len(ts)))
    Y[:,0] = y0

    for i in range(len(ts)-1):

        t_prev, t_next = ts[i], ts[i+1]
        dt = t_next - t_prev
        y_prev = Y[:,i]

        # initial guess: explicit Euler
        y_new = y_prev + dt * f(t_prev, y_prev)

        # Newton solve:  G(y) = y - y_prev - dt * f(t_next, y) = 0
        for _ in range(maxiter):

            F = y_new - y_prev - dt * f(t_next, y_new)
            if np.linalg.norm(F) < tol:
                break

            # assemble Jacobian J = dG/dy = I - dt * df/dy
            if jac is not None:
                Jf = jac(t_next, y_new)
            else:
                # finite-difference approx
                eps = np.sqrt(np.finfo(float).eps)
                f0  = f(t_next, y_new)
                Jf  = np.zeros((m,m))
                for j in range(m):
                    y_eps     = y_new.copy()
                    y_eps[j] += eps
                    Jf[:,j] = (f(t_next, y_eps) - f0) / eps

            J = np.eye(m) - dt * Jf
            delta = np.linalg.solve(J, -F)
            y_new += delta

        Y[:,i+1] = y_new

    return Y



# Example: dy/dt = -y  ⇒  y(t)=y0*exp(-t)
if __name__ == '__main__':
    def f(t, y):
        return -y

    # optional Jacobian for faster Newton:
    def jac(t, y):
        return np.array([[-1.0]])

    ts = np.linspace(0, 5, 5001)
    y0 = np.array([1.0])

    Y = backward_euler(f, ts, y0, jac=jac)

    # Compare to exact:
    import matplotlib.pyplot as plt
    plt.plot(ts, Y[:,0], 'o', label='BEuler')
    plt.plot(ts, np.exp(-ts), '-', label='exact')
    plt.legend()
    plt.show()
