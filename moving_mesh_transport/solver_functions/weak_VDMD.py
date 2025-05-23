import numpy as np
from scipy.interpolate import interp1d as interp1d

def basis_hat(t, a, b):
    """
    Compactly supported hat function. Returns basis value and derivative
    """
    c = (b + a)/2
    if a <=t <= c:
        return (t-a)/ (c-a), 1/(c-a)
    elif c <t <b:
        return (b-t)/ (b-c), -1/(b-c)
    else:
        return 0, 0
    
def midpoint_integration(f, a, b, n):
    """
    Approximate the definite integral of function f over [a, b]
    using the midpoint rule with n subintervals.
    
    Parameters:
    - f: callable, the integrand function f(x)
    - a: float, the lower limit of integration
    - b: float, the upper limit of integration
    - n: int, the number of subintervals
    
    Returns:
    - float, the approximate value of the integral
    """
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        total += f(x_mid)
    return total * h

# Example usage:
# import math

# Approximate ∫₀¹ x² dx = 1/3
# result = midpoint_integration(lambda x: x**2, 0, 1, 1000)
# print("Approximate integral of x^2 from 0 to 1:", result)


def weak_VDMD(Y_minus, ts):
    # Form V-, Vplus
    n = 8
    Y_minus_interp = interp1d(ts, Y_minus)
    Vminus = np.zeros((ts.size, Y_minus.size))
    Vplus = np.zeros((ts.size, Y_minus.size))
    for it, tt in enumerate(ts):
        a = ts[it]
        b = ts[it+2]
        for jt in range(Y_minus.size):
            f = lambda t: Y_minus_interp(t) * basis_hat(t, a, b)[1]
            f2 = lambda t: Y_minus_interp(t) * basis_hat(t, a, b)[0]
            Vminus[it, jt] = -midpoint_integration(f, a, b, n)
            Vplus[it, jt] = midpoint_integration(f2, a, b, n)
    return Vminus, Vplus



    