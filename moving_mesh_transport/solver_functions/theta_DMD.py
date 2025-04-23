import numpy as np
# This was written by ChatGPT
def theta_DMD(
    data, 
    time_array, 
    theta=0.5, 
    rank_spatial=None, 
    rank_system=None
):
    """
    Implements the theta-DMD method for potentially non-uniformly sampled snapshots.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_features, n_snapshots)
        The high-dimensional data matrix whose columns are snapshots in time.
    time_array : np.ndarray, shape (n_snapshots,)
        The array of time instants t_0, t_1, ..., t_{ns-1}.
        These need not be equally spaced.
    theta : float, optional
        Theta parameter in [0, 1]. The method reduces to Crank-Nicolson at theta = 0.5,
        explicit-like at theta=0, and implicit-like at theta=1.
    rank_spatial : int or None, optional
        Rank for the first SVD-based dimensional reduction on the snapshot matrix.
        If None, no truncation is performed (full rank).
    rank_system : int or None, optional
        Rank for the second SVD-based reduction (the “system” projection).
        If None, no truncation is performed (full rank).
    
    Returns
    -------
    results : dict
        A dictionary containing fields:
        - 'mu' : 1D array of length rtilde, the eigenvalues of the reduced system.
        - 'Phi' : np.ndarray, shape (n_features, rtilde), the DMD modes in high-dimensional space.
        - 'frequencies' : 1D array of modal frequencies (if you convert from discrete mu).
        - 'growth_rates' : 1D array of modal growth/decay rates.
        - 'amplitudes' : 1D array of mode amplitudes for best-fit reconstruction (optional).
        - 'L0' : the left singular vectors from the first SVD (if you need them).
        - 'S0' : the singular values from the first SVD.
        - 'R0' : the right singular vectors from the first SVD.
    
    Notes
    -----
    This code follows the approach in:
      B. Li, J. Garicano-Mena, and E. Valero,
      "A dynamic mode decomposition technique for the analysis of non-uniformly sampled flow data,"
      Journal of Computational Physics 468 (2022) 111495.
    See especially Eq. (20) and surrounding discussion for M_delta and M_theta.
    """

    # ----------------------------------------------------------------
    # 1) DIMENSIONAL REDUCTION (SVD) OF THE ORIGINAL SNAPSHOT MATRIX
    #    data.shape = (n_features, n_snapshots)
    #    We factor data = L0 * S0 * R0^T,  then define chronos C = S0 * R0^T
    #    For large data, consider a truncated SVD or randomized SVD if needed.
    # ----------------------------------------------------------------
    U0, s0, VT0 = np.linalg.svd(data, full_matrices=False)  # shape: U0=(n_feat,ns), s0=(ns,), VT0=(ns,ns)

    
    if rank_spatial is not None:
        # Truncate
        U0 = U0[:, :rank_spatial]    # shape (n_features, r0)
        s0 = s0[:rank_spatial]       # shape (r0,)
        VT0 = VT0[:rank_spatial, :]  # shape (r0, n_snapshots)
        
    # Build the "chronos" matrix:  C = S0 * R0^T  (but each row is the unit basis in reduced space)
    # We'll store them as well:
    L0 = U0                      # shape (n_features, r0)
    S0 = np.diag(s0)            # shape (r0, r0)
    R0 = VT0.T                  # shape (n_snapshots, r0)
    
    # chronos matrix: (r0, n_snapshots)
    C = S0 @ R0.T               # shape (r0, n_snapshots)

    # ----------------------------------------------------------------
    # 2) BUILD M_delta and M_theta  (Eq. (20) in the paper)
    #    The time steps are dt_j = time_array[j+1] - time_array[j]
    #    Each is used to fill the diagonal entries of M_delta, M_theta.
    # ----------------------------------------------------------------
    t = time_array
    ns = t.size
    if ns < 2:
        raise ValueError("Need at least two snapshots in time_array.")
    
    # We'll build M_delta and M_theta as shape (ns, ns-1).
    # The paper's formula for M_delta is basically difference-based; M_theta is a linear combination
    # that uses theta.  For j=1..ns-1, dt_j = t[j] - t[j-1].
    
    M_delta = np.zeros((ns, ns-1), dtype=float)
    M_theta = np.zeros((ns, ns-1), dtype=float)
    
    for j in range(ns-1):
        dt_j = t[j+1] - t[j]
        # M_delta has   1/dt_j  and -1/dt_j in consecutive rows,
        # but for the simplest consistent approach we do something akin to:
        #   ( c_{j+1} - c_j ) / dt_j
        # one way is to place -1/dt_j in row j, +1/dt_j in row j+1
        M_delta[j,   j] = -1.0 / dt_j
        M_delta[j+1, j] =  1.0 / dt_j
        
        # M_theta has to incorporate the factor ( (1-theta)* c_j + theta*c_{j+1} )
        # We'll place   theta   in row j   (for c_{j+1}),
        # and    (1-theta) in row j+1 (for c_j).
        # But be careful about sign conventions.  The paper's eq. (20) uses the form:
        #    c * M_delta = Ac ( c * M_theta ).
        # You can cross-check eq. (19)/(20) if needed, but the simplest approach is:
        
        M_theta[j,   j] = theta
        M_theta[j+1, j] = (1.0 - theta)
    
    # ----------------------------------------------------------------
    # 3) FORM X and Y from C and the above M_delta, M_theta.
    #    X = C * M_theta, Y = C * M_delta  (both shape (r0, ns-1))
    # ----------------------------------------------------------------
    X = C @ M_theta   # shape (r0, ns-1)
    Y = C @ M_delta   # shape (r0, ns-1)

    # ----------------------------------------------------------------
    # 4) SVD OF X  =>  X = L1 * S1 * R1^T
    #    Then form A_tilde = L1^T * Y * R1 * inv(S1).
    #    This is the “reduced” operator that approximates A_c from the paper.
    # ----------------------------------------------------------------
    U1, s1, VT1 = np.linalg.svd(X, full_matrices=False)  # shapes: U1=(r0,r0), s1=(r0,), VT1=(r0,ns-1)
    Sinv = np.zeros(s1.size)
    Spos = s1[s1/np.cumsum(s1)>1e-18]
    Sinv[0:Spos.size] = 1.0/Spos.copy()
    
    if rank_system is not None:
        U1 = U1[:, :rank_system]
        s1 = s1[:rank_system]
        VT1 = VT1[:rank_system, :]
    
    # S1_inv = np.diag(1.0/s1)                # shape (rtilde, rtilde)
    S1_inv = np.diag(Sinv)
    L1 = U1                                 # shape (r0,   rtilde)
    R1 = VT1.T                              # shape (ns-1, rtilde)
    
    A_tilde = L1.T @ Y @ R1 @ S1_inv        # shape (rtilde, rtilde)

    # ----------------------------------------------------------------
    # 5) EIGEN-DECOMPOSITION of A_tilde => W, Lambda.
    #    Here, mu[i] are the discrete eigenvalues (the "lambda" in the paper).
    # ----------------------------------------------------------------
    mu, W = np.linalg.eig(A_tilde)          # mu.shape=(rtilde,), W.shape=(rtilde, rtilde)
    
    # ----------------------------------------------------------------
    # 6) RECOVER HIGH-DIMENSIONAL MODES:  Phi = L0 * L1 * W
    #    shape: (n_features, rtilde).
    # ----------------------------------------------------------------
    Phi = L0 @ (L1 @ W)
    
    # ----------------------------------------------------------------
    # Optional: compute frequencies, growth rates, etc.
    #   In the discrete-time sense, mu[i] is the eigenvalue for one "step" in the sense of eq. (21).
    #   If you want a continuous-time interpretation, you'd interpret dt as local or something else.
    #   A rough approach: pick an average dt_avg, then
    #       lambda_cont = log(mu) / dt_avg
    #       freq = imag(lambda_cont), growth = real(lambda_cont).
    #   For usage, you typically interpret each mu in terms of magnitude and argument:
    #       |mu|  => growth/decay per step
    #       arg(mu) => "radians per step" => frequency
    # ----------------------------------------------------------------
    # For a single characteristic time scale, use an average or median dt:
    dt_avg = np.mean(np.diff(time_array))
    # growth rates (approx)
    growth_rates = np.log(np.abs(mu)) / dt_avg
    # frequencies
    frequencies  = np.angle(mu) / (2.0 * np.pi * dt_avg)
    
    # ----------------------------------------------------------------
    # 7) OPTIONAL: Solve for mode amplitudes alpha (least-squares fit).
    #    One standard approach is from Tu et al. (J. Comput. Dyn. 2014):
    #       alpha = (Phi^dagger) * first_snapshot.
    #    But we can also do a least-squares fit against the entire data (Eq. (12) in the paper).
    # ----------------------------------------------------------------
    # We'll do the simpler approach: alpha that matches the first snapshot.
    snapshot0 = data[:, 0]  # shape (n_features,)
    
    # shape( n_features, rtilde ) => pseudo-inverse shape (rtilde, n_features )
    Phi_pinv = np.linalg.pinv(Phi) 
    alpha = Phi_pinv @ snapshot0  # shape (rtilde,)
    
    # Package into a dict and return
    results = {
        'mu'          : mu,
        'Phi'         : Phi,
        'growth_rates': growth_rates,
        'frequencies' : frequencies,
        'amplitudes'  : alpha,
        'L0'          : L0,
        'S0'          : s0,
        'R0'          : R0
    }
    # return results
    return mu

#
#  Example usage:
#
if __name__ == "__main__":
    # Suppose we have data in shape (n_features, n_snapshots), plus times:
    # For a quick test, let's build synthetic data with a single frequency,
    # but sample times in a non-uniform manner:
    
    import numpy as np
    
    n_features  = 50
    n_snapshots = 80
    t_uniform   = np.linspace(0, 4*np.pi, n_snapshots)
    # Make it non-uniform, e.g. random perturbation:
    rng         = np.random.default_rng(42)
    t_random    = t_uniform + 0.3*rng.random(n_snapshots)
    t_random    = np.sort(t_random)
    
    # Build a simple sinusoidal "flow" in each snapshot
    X = np.linspace(0, 2*np.pi, n_features)
    data_mat = np.zeros((n_features, n_snapshots))
    freq_true = 2.0  # rad/time
    for j in range(n_snapshots):
        data_mat[:, j] = np.sin(freq_true* t_random[j] + X)
    
    # Now do theta-DMD on that non-uniform data
    results = theta_DMD(data_mat, t_random, theta=0.5, rank_spatial=20, rank_system=10)
    
    print("Discrete eigenvalues mu:")
    print(results['mu'])
    print("Approx continuous growth rates:", results['growth_rates'])
    print("Approx continuous frequencies:  ", results['frequencies'])
