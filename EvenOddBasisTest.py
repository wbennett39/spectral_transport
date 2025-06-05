import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

# 1. Define parameters
R = 1.0             # Domain upper limit
Ne = 5              # Number of even basis functions
No = 5              # Number of odd basis functions
num_points = 1000   # Grid resolution

# Define the domain grid in [0, R]
r = np.linspace(0, R, num_points)
dr = r[1] - r[0]

# 2. Compute local coordinate zeta in [-1, 1] for basis definitions
zeta = 2 * r / R - 1  # Maps r in [0,R] to zeta in [-1,1]

# 3. Define even and odd basis functions using Legendre polynomials
#    Even basis: P_{2j}(zeta), j = 0, 1, ..., Ne-1
#    Odd  basis: P_{2j+1}(zeta), j = 0, 1, ..., No-1
even_basis = [sp.legendre(2 * j)(zeta) for j in range(Ne)]
odd_basis  = [sp.legendre(2 * j + 1)(zeta) for j in range(No)]

even_basis = np.array(even_basis)  # Shape: (Ne, num_points)
odd_basis  = np.array(odd_basis)   # Shape: (No, num_points)

# 4. Define a test function f(r). Here we use the Bessel function J0
#    for demonstration: f(r) = J0(k0 * r) with k0 chosen so that f varies.
k0 = 5.0
f = sp.jv(0, k0 * r)

# 5. Compute expansion coefficients for even and odd bases via least-squares (orthonormal projection):
#    coefficient c_j = ( ∫ phi_j(r) f(r) dr ) / ( ∫ phi_j(r)^2 dr )
coeff_even = np.zeros(Ne)
coeff_odd  = np.zeros(No)

for j in range(Ne):
    phi_j = even_basis[j, :]
    # Numerator: ∫ phi_j(r) * f(r) dr
    numerator = np.trapz(phi_j * f, r)
    # Denominator: ∫ phi_j(r)^2 dr
    denominator = np.trapz(phi_j * phi_j, r)
    coeff_even[j] = numerator / denominator

for j in range(No):
    phi_j = odd_basis[j, :]
    numerator = np.trapz(phi_j * f, r)
    denominator = np.trapz(phi_j * phi_j, r)
    coeff_odd[j] = numerator / denominator

# 6. Construct truncated expansions
f_even_approx = np.zeros_like(r)
f_odd_approx  = np.zeros_like(r)

for j in range(Ne):
    f_even_approx += coeff_even[j] * even_basis[j, :]

for j in range(No):
    f_odd_approx += coeff_odd[j] * odd_basis[j, :]

# Combined approximation (even + odd)
f_total_approx = f_even_approx + f_odd_approx

# 7. Plotting
plt.figure(figsize=(10, 8))

# Original test function
plt.plot(r, f, 'k-', label='Original f(r) = J0(5r)', linewidth=2)

# Even expansion
plt.plot(r, f_even_approx, 'b--', label=f'Even expansion (Ne={Ne})')

# Odd expansion
plt.plot(r, f_odd_approx, 'r-.', label=f'Odd expansion (No={No})')

# Combined expansion
plt.plot(r, f_total_approx, 'g-', label=f'Combined expansion (Ne+No={Ne+No})', linewidth=1.5)

plt.xlabel('r')
plt.ylabel('Function value')
plt.title('Expansion of f(r) = J0(5r) in Even and Odd Legendre-Based Basis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
