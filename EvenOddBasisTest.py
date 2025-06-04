import numpy as np
from scipy.special import j0
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1) A small helper to build Gauss–Radau–Legendre nodes on [-1,1], with xi0=-1
#    (so that we can enforce u'(0)=0 in the first cell).  We take N total interior
#    points so that we end up with N+1 nodes: xi_0 = -1, xi_1..xi_{N-1} = roots of P_N', xi_N= +1.
# ─────────────────────────────────────────────────────────────────────────────
def gauss_radau_legendre(N):
    """
    Return N+1 nodes on [-1,1] consisting of:
      - xi[0] = -1
      - xi[1..N-1] = the N-1 interior roots of (d/dx)P_N(x)
      - xi[N] = +1
    These are the Gauss–Radau–Legendre points with a fixed endpoint at -1.
    """
    from numpy.polynomial.legendre import Legendre

    # Build the Legendre polynomial P_N(x):
    Pn = Legendre.basis(N)
    # Its derivative Pn'(x) is a degree-(N-1) polynomial:
    dPn = Pn.deriv()
    # Find the roots of Pn'(x) in (-1,1):
    interior = np.sort(dPn.roots())
    # Now concatenate with the endpoints -1 and +1:
    xi = np.empty(N+1)
    xi[0]  = -1.0
    xi[1:-1] = interior
    xi[-1] = +1.0
    return xi

# ─────────────────────────────────────────────────────────────────────────────
# 2) SETUP
#    - Partition [0,1] into four equal cells
#    - On the first cell we will use Gauss–Radau–Legendre nodes (so that r=0 is a node
#      with enforced u'(0)=0).  On each other cell we clamp both endpoints (Gauss–Lobatto).
#    - In all cases we pick "N+1" nodes per cell, where N=4 → 5 points per cell.
# ─────────────────────────────────────────────────────────────────────────────
cells = [(0.0, 0.25),
         (0.25, 0.50),
         (0.50, 0.75),
         (0.75, 1.00)]

# polynomial order per cell = 4  (→ 5 nodes per cell)
N = 4

# We will do a fine plot of x in [0,1] to compare true J0 vs approximation
x_plot = np.linspace(0, 1, 500)
u_true = j0(x_plot)

# Storage for the piecewise‐interpolant:
u_interp = np.zeros_like(x_plot)

# ─────────────────────────────────────────────────────────────────────────────
# 3) BUILD PIECEWISE SPECTRAL‐ELEMENT INTERPOLATION
#
# For each cell [a,b]:
#   3a) Build N+1 reference nodes xi_j in [-1,1].
#        - if a==0.0: use Gauss–Radau–Legendre
#        - else:       use Gauss–Lobatto–Legendre (same code, since we fix both ends)
#   3b) Map xi_j → s_j ∈ [0,1] by s_j = (xi_j+1)/2   → r_j = a + L*s_j.
#   3c) Evaluate u_j = J0(r_j).
#   3d) Build the barycentric weights w_j for interpolation in xi‐space.
#        The standard formula is:
#           w_j = 1 / ∏_{m≠j} (xi_j - xi_m).
#   3e) For each x ∈ [a,b], we compute s = (x−a)/L, xi_val = 2s−1, then do:
#           if xi_val == xi_j for some j:   p(x) = u_j
#           else:
#             p(x) = [∑_{j=0}^N ( w_j * u_j  / (xi_val - xi_j) )]
#                     / [∑_{j=0}^N   w_j      / (xi_val - xi_j) ].
#   3f) Store p(x) into u_interp for those x‐values.
# ─────────────────────────────────────────────────────────────────────────────
for (a, b) in cells:
    L = b - a

    # 3a) choose nodes:
    if a == 0.0:
        # First cell: fix xi0 = -1 → Gauss–Radau
        xi = gauss_radau_legendre(N)
    else:
        # Other cells: want Gauss–Lobatto (xi0 = -1, xiN = +1, interior = roots of P_N')
        xi = gauss_radau_legendre(N)

    # 3b) map xi_j ∈ [-1,1] to r_j ∈ [a,b]:
    s_nodes = 0.5 * (xi + 1.0)        # s ∈ [0,1]
    r_nodes = a + L * s_nodes         # physical node positions
    u_nodes = j0(r_nodes)             # sample the test function

    # 3d) build barycentric weights for {xi_j}
    w = np.ones(N+1)
    for j in range(N+1):
        # w_j = 1 / ∏_{m≠j} (xi_j - xi_m)
        diffs = xi[j] - np.delete(xi, j)
        w[j] = 1.0 / np.prod(diffs)

    # 3e) interpolate at each x in [a,b]
    mask = (x_plot >= a) & (x_plot <= b)
    x_cell = x_plot[mask]
    s_cell = (x_cell - a) / L   # ∈ [0,1]
    xi_val = 2.0 * s_cell - 1.0 # map to reference [-1,1]

    p_vals = np.zeros_like(x_cell)
    for k, xv in enumerate(xi_val):
        # if xv exactly equals one of the nodes, pick u_nodes[j]
        diffs = xv - xi
        if np.any(np.abs(diffs) < 1e-14):
            p_vals[k] = u_nodes[np.argmin(np.abs(diffs))]
        else:
            # standard barycentric formula
            numer   = np.sum(w * u_nodes / diffs)
            denom   = np.sum(w        / diffs)
            p_vals[k] = numer / denom

    # store into the global array
    u_interp[mask] = p_vals

# ─────────────────────────────────────────────────────────────────────────────
# 4) PLOT
plt.figure(figsize=(7,4))
plt.plot(x_plot, u_true,   'k-',  lw=2, label='$J_0(r)$')
plt.plot(x_plot, u_interp, 'b--', lw=2, label='Nodal Spectral–Element Interp.')
plt.xlabel('$r$')
plt.ylabel('Function value')
plt.title('Piecewise Spectral‐Element Interpolation of $J_0(r)$ on [0,1]')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()
