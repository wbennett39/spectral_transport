import numpy as np
from numba import njit
@njit
def backward_euler(f, ts, y0,  mesh, matrices, num_flux, source, uncollided_sol, flux, transfer, sigma_class, thermal_couple, N_ang, N_space, N_groups, M, rhs,jac=None, tol=1e-7, maxiter=10):
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
    Y =  np.ascontiguousarray(Y)
    Y[:,0] = y0

    for i in range(len(ts)-1):


        t_prev = ts[i]
        t_next =  ts[i+1]
        dt = t_next - t_prev
        y_prev = Y[:,i]
        y_prev = np.ascontiguousarray(y_prev)
        # initial guess: explicit Euler
        y_new = y_prev + dt * f(t_prev, y_prev,  mesh, matrices, num_flux, source, uncollided_sol, flux, transfer, sigma_class, thermal_couple, N_ang, N_space, N_groups, M, rhs)
        y_new = np.ascontiguousarray(y_new)

        # Newton solve:  G(y) = y - y_prev - dt * f(t_next, y) = 0
        for _ in range(maxiter):

            F = y_new - y_prev - dt * f(t_next, y_new,  mesh, matrices, num_flux, source, uncollided_sol, flux, transfer, sigma_class, thermal_couple, N_ang, N_space, N_groups, M, rhs)
            if np.linalg.norm(F) < tol:
                break

            # assemble Jacobian J = dG/dy = I - dt * df/dy
            if jac is not None:
                Jf = jac(t_next, y_new)
            else:
                # finite-difference approx
                # eps = np.sqrt(np.finfo(float).eps)
                eps = 1.5e-8
                # eps = np.sqrt(eps)
                f0  = f(t_next, y_new,  mesh, matrices, num_flux, source, uncollided_sol, flux, transfer, sigma_class, thermal_couple, N_ang, N_space, N_groups, M, rhs)
                Jf  = np.zeros((m,m))
                for j in range(m):
                    y_eps     = y_new.copy()
                    y_eps[j] += eps
                    Jf[:,j] = (f(t_next, y_eps,  mesh, matrices, num_flux, source, uncollided_sol, flux, transfer, sigma_class, thermal_couple, N_ang, N_space, N_groups, M, rhs) - f0) / eps

            J = np.eye(m) - dt * Jf
            delta = np.linalg.solve(J, -F)
            y_new += delta

        Y[:,i+1] = y_new

    return Y






import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator


def backward_euler_krylov(f, ts, y0, jac=None, tol=1e-8, newton_maxiter=50,
                          krylov_tol=1e-4, krylov_maxiter=None):
    """
    Backward Euler with Newton–Krylov linear solver (GMRES) and optional analytic Jacobian preconditioning.

    Parameters
    ----------
    f : callable
        f(t, y) -> dy/dt (returns array of shape (m,))
    ts : array_like, shape (N,)
        Time grid
    y0 : array_like, shape (m,)
        Initial state
    jac : callable, optional
        jac(t, y) -> df/dy (m×m), analytic Jacobian for preconditioning
    tol : float
        Newton convergence tolerance
    newton_maxiter : int
        Maximum Newton iterations per step
    krylov_tol : float
        GMRES relative residual tolerance
    krylov_maxiter : int or None
        Max GMRES iterations (defaults to m)

    Returns
    -------
    Y : ndarray, shape (N, m)
        Solution at each ts
    """
    ts = np.asarray(ts)
    m = y0.size
    Y = np.zeros((len(ts), m))
    Y[0] = y0
    eps_jv = np.sqrt(np.finfo(float).eps)

    for i in range(len(ts) - 1):
        t_prev, t_next = ts[i], ts[i + 1]
        dt = t_next - t_prev
        y_prev = Y[i].copy()
        y_new = y_prev + dt * f(t_prev, y_prev)

        for nit in range(newton_maxiter):
            F = y_new - y_prev - dt * f(t_next, y_new)
            if np.linalg.norm(F) < tol:
                break

            f0 = f(t_next, y_new)

            def matvec(v):
                v = v.reshape(m)
                jv = (f(t_next, y_new + eps_jv * v) - f0) / eps_jv
                return (v - dt * jv).ravel()

            A = LinearOperator((m, m), matvec=matvec)

            if jac is not None:
                Jf = jac(t_next, y_new)
                def psolve(v):
                    return np.linalg.solve(np.eye(m) - dt * Jf, v)
                M = LinearOperator((m, m), matvec=psolve)
            else:
                M = None

            b = -F.ravel()
            restart = 1 if m == 1 else min(m, 20)
            maxiter = krylov_maxiter if krylov_maxiter is not None else m

            delta, info = gmres(A, b, M=M, tol=krylov_tol, atol=0,
                                restart=restart, maxiter=maxiter)
            if info > 0:
                raise RuntimeError(f"GMRES failed to converge in {info} iterations")
            elif info < 0:
                raise RuntimeError("GMRES encountered an illegal input or breakdown")

            y_new = (y_new + delta).reshape(m)
        else:
            raise RuntimeError("Newton iteration failed to converge")

        Y[i + 1] = y_new

    return Y

# Example usage & test:
if __name__ == "__main__":
    def f_test(t, y): return -y
    def jac_test(t, y): return np.array([[-1.0]])
    ts = np.linspace(0, 5, 51)
    y0 = np.array([1.0])
    Y = backward_euler_krylov(f_test, ts, y0, jac=jac_test)
    print("Solution at t=5:", Y[-1, 0])


# Example usage:
if __name__ == "__main__":
    # Simple test: dy/dt = -y, y(0)=1
    def f_test(t, y):
        return -y

    ts = np.linspace(0, 5, 51)
    y0 = np.array([1.0])
    Y = backward_euler_krylov(f_test, ts, y0)
    print("Solution at t=5:", Y[-1, 0])

# Display the function definition to the user
# print(backward_euler_krylov.__doc__)
# Example: dy/dt = -y  ⇒  y(t)=y0*exp(-t)
if __name__ == '__main__':
    def f(t, y):
        return -y

    # optional Jacobian for faster Newton:
    def jac(t, y):
        return np.array([[-1.0]])

    ts = np.linspace(0, 5, 5001)
    y0 = np.array([1.0])

    Y = backward_euler_krylov(f, ts, y0)

    # Compare to exact:
    import matplotlib.pyplot as plt
    plt.plot(ts, Y[:,0], 'o', label='BEuler')
    plt.plot(ts, np.exp(-ts), '-', label='exact')
    plt.legend()
    plt.show()