import numpy as np
from scipy.interpolate import interp1d as interp1d
import matplotlib.pyplot as plt
import math
from numba import njit
def basis_hat(t, a, b, c):
    """
    Compactly supported hat function. Returns basis value and derivative
    """
    # c = (b + a)/2
    if a <=t <= c:
        return (t-a)/ (c-a), 1/(c-a)
    elif c <t <b:
        return (b-t)/ (b-c), -1/(b-c)
    else:
        return 0, 0

def basis_square(t, a, b, c):
    if a<= t <= b:
        return 1, 0
    else:
        return 0, 0 
@njit
def basis_quadratic(t, a, b):
    phi = 0.0
    dphi = 0.0
    if a <=t<=b:
        phi = -(t-a) * (t-b) /(.25 * (a-b)**2)
        dphi = (-2*t + a+ b)/(.25 * (a-b)**2)
    if t == a:
        dphi += -(t-a) * (t-b) /(.25 * (a-b)**2)
    elif t == b:
        dphi += -(t-a) * (t-b) /(.25 * (a-b)**2)
    return phi, dphi


def basis_gauss(t, a, sigma):
    return np.exp(-(t-a)**2 /sigma), -2/ sigma * np.exp(-(t-a)**2 /sigma) * (t-a)

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
@njit
def trapz_nonuniform(x, y):
    total =np.zeros(len(y[0,:]))
    for i in range(len(x)-1):
        h = x[i+1] - x[i]
        total += 0.5 * h * (y[i, :] + y[i+1, :])
    return total

#test trap integrator
# f = lambda x: np.sin(x) + x**2
# xs = np.linspace(0,2)
# print(trapz_nonuniform(xs, f(xs)), 'should be 4.08281')
# assert 0


# Approximate ∫₀¹ x² dx = 1/3
# result = midpoint_integration(lambda x: x**2, 0, 1, 1000)
# print("Approximate integral of x^2 from 0 to 1:", result)
def my_midpoint(t1, t2, t3, Ys):
    res = Ys * 0
    m1 = (t1 + t2) / 2
    m2 = (t2 + t3) / 2
    delta1 = t2- t1
    delta2 = t3 - t1

@njit
def weak_VDMD(Y_minus, Y_plus, ts, basis = 'quadratic'):
    # Form V-, Vplus
    I = len(Y_minus[:, 0])
    J0 = max(ts.size -2, 0) # basis functions covering 3 time points 
    J1 = max(ts.size -4, 0) # basis functions covering 5 time points
    J2 = max(ts.size -6, 0)  # basis functions covering 7 time points
    J3 = max(ts.size -8, 0)# basis functions covering 9 time points
    J4 = max(ts.size - 10, 0) # etc


    Vminus = np.zeros((J0 + J1  + J2  + J3 + J4, I))
    Vplus = np.zeros((J0 + J1 + J2 + J3 + J4, I))
    # form J0 part of matrix
    for j0 in range(J0 + J1 + J2 + J3 + J4):
        # form integrand vector
        basis_vec1 = ts.copy() # derivative of basis
        basis_vec2 = ts.copy() # basis
   
        if j0 < J0:
            jp = j0
            if j0 < ts.size -2:
                a = ts[j0]
                b = ts[j0+2]
                c = ts[j0+1]
            else:
                a = ts[-3]
                b = ts[-1]
                c = ts[-2]
        elif j0 >= J0 and j0 < J1 + J0:
            jp = j0 - J0
            if jp < ts.size -4:
                a = ts[jp]
                b = ts[jp+4]
                c = ts[jp+2]
            else:
                a = ts[-5]
                b = ts[-1]
                c = ts[-3]
        elif j0 >= J1 +J0 and j0 < J2 + J1 + J0:
            jp = j0 - J1 - J0
            if jp < ts.size -6:
                a = ts[jp]
                b = ts[jp+6]
                c = ts[jp+3]
            else:
                a = ts[-7]
                b = ts[-1]
                c = ts[-4]
        elif j0 >= J2 + J1 + J0  and j0 < J0 + J1 + J2 + J3:
            jp = j0 - J2 -J1 - J0
            if jp < ts.size -8:
                a = ts[jp]
                b = ts[jp+8]
                c = ts[jp+4]
            else:
                a = ts[-9]
                b = ts[-1]
                c = ts[-5]
        elif j0 >= J3 + J2 + J1 + J0:
            
            jp = j0 - J2 -J1 - J0
            if jp < ts.size -10:
                a = ts[jp]
                b = ts[jp+10]
                c = ts[jp+6]
            else:
                a = ts[-11]
                b = ts[-1]
                c = ts[-5]

        for it in range(ts.size):
            if basis == 'quadratic':
                basis_vec1[it] = basis_quadratic(ts[it], a, b)[1]
                basis_vec2[it] = basis_quadratic(ts[it], a, b)[0]
        # form integrands
        # plt.ioff()
        # plt.plot(ts, basis_vec2)
        print(basis_vec2)
        integrand_vec = np.zeros((ts.size, I))
        integrand_vec2 = np.zeros((ts.size, I))
        for it3 in range(ts.size):
            for ii in range(I):
                integrand_vec[it3, ii] = Y_minus[ii, it3] * basis_vec1[it3]
                integrand_vec2[it3, ii] = Y_minus[ii, it3] * basis_vec2[it3]
        Vplus[j0, :]= -trapz_nonuniform(ts, integrand_vec )
        Vminus[j0, :] = trapz_nonuniform(ts, integrand_vec2 )
        # print(Vplus[j0, :], 'V+')
        # print(Vminus[j0, :], 'V-')

            


    # for it in range(ts.size-1):
    #     if it < ts.size-4:
    #         a = ts[it]
    #         b = ts[it+4]
    #         c = ts[it+2]
    #     else:
    #         a = ts[ts.size-4]
    #         b = ts[-1]
    #         c = ts[ts.size-2]
            # if it < ts.size -2:
            #     a = ts[it]
            #     b = ts[it + 2] 
            #     c = ts[it+1]
            # else:
            #     a = ts[it]
            #     b = ts[it+1]
            #     c = (a + b)/2
        # print(a, b, c)
        # delta_t = ts[it+1] - ts[it]
        # basis_vec1 = ts.copy()
        # basis_vec2 = ts.copy()
        # for t in range(ts.size):
        #     if basis == 'hat':
        #         basis_vec1[t] = basis_hat(ts[t], a, b, c)[1]
        #         basis_vec2[t] = basis_hat(ts[t], a, b, c)[0]
        #     elif basis == 'Gauss':
        #         # dist_from_zero = np.min(ts[t], abs(ts[t] - 100.0))
        #         dist_from_zero = c
        #         threesigma = dist_from_zero
        #         sigma = threesigma /3
        #         print(sigma)
        #         res = min(math.sqrt(sigma),10)
        #         basis_vec1[t] = basis_gauss(ts[t], c, res)[1]
        #         basis_vec2[t] = basis_gauss(ts[t], c, res)[0]
        #     elif basis== 'square':
        #         basis_vec1[t] = basis_square(ts[t], a, b, c)[1]
        #         basis_vec2[t] = basis_square(ts[t], a, b, c)[0]
        #     elif basis == 'quadratic':
        #         basis_vec1[t] = basis_quadratic(ts[t], a, b)[1]
        #         basis_vec2[t] = basis_quadratic(ts[t], a, b)[0]

        # plt.ion()
        # plt.plot(ts, basis_vec2)
        # plt.plot(ts, ts*0, 'k|')
        # plt.show()
        # assert(abs(basis_vec2[0])) < 1e-6
        # assert(abs(basis_vec2[-1])) < 1e-6
    # print(Vminus, Vplus)
    return Vminus[J0:,:], Vplus[J0:,:]



    