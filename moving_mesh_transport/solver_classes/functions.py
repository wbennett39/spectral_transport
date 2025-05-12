from numba import njit, types, prange
# import quadpy
import ctypes
from numba.extending import get_cython_function_address
import numpy as np
import math
from scipy.special import expi
import matplotlib.pyplot as plt
from ..plots.plot_functions.show import show
import numpy.polynomial as poly
from functools import partial
from scipy.special import roots_legendre
import numpy.polynomial as poly
import scipy.special as sps



@njit 
def integrate_quad(a, b, xs, ws, func1, func2):
    return (b-a)/2 * np.sum(ws * func1((b-a)/2*xs + (a+b)/2) * func2((b-a)/2*xs + (a+b)/2))

_dble = ctypes.c_double
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_0_1eval_legendre")
functype = ctypes.CFUNCTYPE(_dble, _dble, _dble)
eval_legendre_float64_fn = functype(addr)

# @njit("float64[:](float64,float64[:])")  
@njit
def numba_eval_legendre_float64(n, x):
      return eval_legendre_float64_fn(n, x)
  
addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1expi")
functype = ctypes.CFUNCTYPE(_dble, _dble)
expn_fn = functype(addr)

@njit("float64(float64)")
def numba_expi(x):
    return expn_fn(x)

# @njit("float64[:](float64,float64[:],float64,float64)", looplift=False, parallel=False)
@njit('float64[:](int64, float64[:], float64, float64)')
def normPn(n,x,a=-1.0,b=1.0):
    tmp = 0*x
    for count in prange(x.size):
        z = (b+a-2*x[count])/(a-b)
        fact = np.sqrt((2*n+1)/(b-a)) #*(x>=a)*(x<=b)
        # tmp[count] = sc.eval_legendre(n,z)*fact
        tmp[count] = numba_eval_legendre_float64(n, z)
    return tmp * fact



@njit('float64[:](int64, float64[:], float64, float64)')
def Pn(n,x,a=-1.0,b=1.0):
    tmp = 0*x
    for count in prange(x.size):
        z = x[count]
        # tmp[count] = sc.eval_legendre(n,z)*fact
        tmp[count] = numba_eval_legendre_float64(n, z)
    return tmp
@njit('float64(int64, float64, float64, float64)')
def normPn_scalar(n,x,a=-1.0,b=1.0):
    tmp = 0.0
    z = (b+a-2*x)/(a-b)
    fact = np.sqrt((2*n+1)/(b-a)) #*(x>=a)*(x<=b)
    # tmp[count] = sc.eval_legendre(n,z)*fact
    tmp = numba_eval_legendre_float64(n, z)
    return tmp * fact
@njit('float64(int64, float64, float64, float64)')
def Pn_scalar(n,x,a=-1.0,b=1.0):
    tmp = 0.0
    z = x

    # tmp[count] = sc.eval_legendre(n,z)*fact
    tmp = numba_eval_legendre_float64(n, z)
    return tmp 

@njit
def dx_normPn(n, x, a = -1.0, b = 1.0):
    tmp = 0*x
    fact = np.sqrt((2*n+1)/(b-a))
    for count in prange(x.size):
        z = (b+a-2*x[count])/(a-b)
        if n == 0:
            tmp[count] = 0.0
        elif n == 1:
            tmp[count] = 1.0
        elif n == 2:
            tmp[count] = 3*z
        elif n == 3:
            tmp[count] = (-3 + 15*z**2)/2.
        elif n == 4:
            tmp[count] = (-60*z + 140*z**3)/8.
        elif n == 5:
            tmp[count] = (15 - 210*z**2 + 315*z**4)/8.
        elif n == 6:
            tmp[count] = (210*z - 1260*z**3 + 1386*z**5)/16.
        elif n == 7:
            tmp[count] = (-35 + 945*z**2 - 3465*z**4 + 3003*z**6)/16.
        elif n == 8:
            tmp[count] = (-2520*z + 27720*z**3 - 72072*z**5 + 51480*z**7)/128.
        elif n == 9:
            tmp[count] = (315 - 13860*z**2 + 90090*z**4 - 180180*z**6 + 109395*z**8)/128.
        elif n == 10:
            tmp[count] = (6930*z - 120120*z**3 + 540540*z**5 - 875160*z**7 + 461890*z**9)/256.
        elif n == 11:
            tmp[count] = (-693 + 45045*z**2 - 450450*z**4 + 1531530*z**6 - 2078505*z**8 + 969969*z**10)/256.


    return tmp * fact




# @njit("float64[:](int64, float64[:], float64[:], float64[:,:,:], int64, float64[:,:])", parallel = True, looplift = True, fastmath = True)
def make_phi(N_ang, ws, xs, u, M, edges):
    output = xs*0
    psi = np.zeros((N_ang, xs.size))
    for ang in range(N_ang):
        for count in range(xs.size):
            idx = np.searchsorted(edges[:], xs[count])
            if (idx == 0):
                idx = 1
            if (idx >= edges.size):
                idx = edges.size - 1
            for i in range(M+1):
                psi[ang, count] += u[ang,idx-1,i] * normPn(i,xs[count:count+1],float(edges[idx-1]),float(edges[idx]))[0]
    output = np.sum(np.multiply(psi.transpose(), ws), axis = 1)
    return output
@njit 
def surf_func(speed,u,space,j,side,xL,xR,N_space):
#    print(side)
    if j ==0:
        B_right = 1/math.sqrt(xR-xL)
        B_left = 1/math.sqrt(xR-xL)
    else:
         B_right = math.sqrt(2*j+1)/math.sqrt(xR-xL)
         if j%2 ==0:
                B_left = math.sqrt(2*j+1)/math.sqrt(xR-xL)
         else:
                B_left = -math.sqrt(2*j+1)/math.sqrt(xR-xL)
    if speed == 0:
        return 0
    elif speed > 0 and side == "R":
        return u[space,j]*B_right
    elif speed > 0 and side =="L":
        if space !=0:
#            print(u[k-1,j])
            return u[space-1,j]*B_right 
        else:
            return 0
    elif speed < 0 and side =="R":
        if space != N_space-1:
            return u[space+1,j]*B_left
        else:
            return 0
    elif speed < 0 and side =="L":
        return u[space,j]*B_left

@njit
def LU_surf_func(u,space,N_space,mul,M,xL,xR,dxL,dxR):
    sumright = 0
    sumleft = 0
    rightspeed = mul - dxR
    leftspeed = mul-dxL
    for j in range(0,M+1):
        sumright += surf_func(rightspeed,u,space,j,"R",xL,xR,N_space)
        sumleft += surf_func(leftspeed,u,space,j,"L",xL,xR,N_space)
    LU = np.zeros(M+1).transpose()
    for i in range(0,M+1):
        if i == 0:
            B_right = 1/math.sqrt(xR-xL)
            B_left = 1/math.sqrt(xR-xL)
        elif j>0:
            B_right = math.sqrt(2*i+1)/math.sqrt(xR-xL)
            if i%2 ==0:
                B_left = math.sqrt(2*i+1)/math.sqrt(xR-xL)
            else: 
                B_left = -math.sqrt(2*i+1)/math.sqrt(xR-xL)
        LU[i] = rightspeed*B_right*(sumright) - leftspeed*B_left*(sumleft)
    return LU 

def find_nodes(edges, M, geometry):
    # scheme = quadpy.c1.gauss_legendre(M+1)
    if M == 0:
        M = 1
    if geometry['slab'] == True:
        xs_quad, ws_quad = poly.legendre.leggauss(M)
        # xs_quad = scheme.points
        ixx = xs_quad.size
        xs_list = np.zeros(ixx*(edges.size-1))
        for k in range(edges.size-1):
            xL = edges[k]
            xR = edges[k+1]
            xs_list[k*ixx:(k+1)*ixx] = xL + (xs_quad + 1)*(xR - xL)/2
    
    elif geometry['sphere'] == True:
        xs_quad, ws_quad = quadrature(M, 'chebyshev')
        ixx = xs_quad.size
        xs_list = np.zeros(ixx*(edges.size-1))
        for k in range(edges.size-1):
            xL = edges[k]
            xR = edges[k+1]
            xs_list[k*ixx:(k+1)*ixx] = np.sort(xL + (xs_quad + 1)*(xR - xL)/2)

    return xs_list

def convergence(err1, x1, err2, x2):
    return -math.log(err2/err1)/math.log(x2/x1)
@njit
def f1(t, tau, x0):
    return -x0 * numba_expi(tau-t)
@njit    
def f2(t, tau, x0, x):
    if tau != t:
        return 0.5*((-x0 + abs(x)) * numba_expi(tau-t) + math.exp(tau - t))
    else:
        return 0.5 
@njit
def f3(t, tau, x0, x):
    return math.exp(tau-t)
@njit
def uncollided_square_s2(x, t, x0, t0):
    t_ceiling = min(t,t0)
    if t > 0:
        tau_1 = 0.0
        end = min(t_ceiling, t - abs(x) + x0)
        if end <= 0.0:
            return 0.0
        tau_2 = min(end, t - x0 - abs(x))
        if tau_2 < 0.0:
            tau_2 = 0.0
        tau_3 = min(end, t - x0 + abs(x))
        if tau_3 < 0.0:
            tau_3 = 0.0
        tau_4 = end
        if tau_4 < 0.0:
            tau_4 = 0.0
        t1 = f1(t, tau_2, x0) - f1(t, tau_1, x0)
        t2 = f2(t, tau_3, x0, x) - f2(t, tau_2, x0, x)
        t3 = f3(t, tau_4, x0, x) - f3(t, tau_3, x0, x)
        
        return t1 + t2 + t3
    else:
        return 0.0

@njit
def s2_F(t,tau):
    """ integrand for uncollided square s2
    """
    return 0.5 * math.exp(-t + tau)

@njit 
def uncollided_su_olson_s2(x,t,x0,t0):
    sqrt3 = math.sqrt(3)
    abx = abs(x)
    edge = min(t,t0)
    if t <= t0:
        if (abx > x0):
            arg1 = max(0,edge - sqrt3 * (abx - x0))
            arg2 = max(0,edge - sqrt3 * (abx + x0))
            # arg2 = max(arg2, edge)
            # arg2 = min(arg2,t0)
            # arg1 = min(arg2,t0)
            return s2_F(t, arg1) - s2_F(t, arg2)
        
        elif (abx <= x0):
            if (edge + sqrt3 * abx <= sqrt3 * x0):
                return  2 * (s2_F(t, edge) - s2_F(t, 0))
            elif (edge + sqrt3 * abx > sqrt3 * x0) and (edge - sqrt3 * (abx + x0) > 0):
                arg2 = edge - sqrt3 * (x0 - abx)
                arg1 = edge - sqrt3 * (abx + x0)
                if arg1 <0 or arg2 <0:
                    print("error negative bounds")
                return s2_F(t, arg2) - s2_F(t, arg1) + 2 * (s2_F(t, edge) - s2_F(t, arg2))
            elif (edge + sqrt3 * abx > sqrt3 * x0) and (edge - sqrt3 * (abx + x0) <= 0): 
                arg1 = max(0,edge - sqrt3 * (x0 - abx))
                if arg1 <0:
                    print("error negative bounds")
                return 2 * (s2_F(t, edge) - s2_F(t, arg1)) + s2_F(t, arg1) - s2_F(t, 0)
            else:
                print("missed case")
    elif t > t0:
        T0 = edge
        x = abs(x)
        if x0 - math.sqrt(3)*(t-T0)/3.0 <= x <= x0 + math.sqrt(3)*(t-T0)/3.0:
            arg = max(t - (x-x0)*3/math.sqrt(3),0)
            argp = min(arg, T0)
            arg2 = max(t - (x+x0)*3/math.sqrt(3),0)
            arg2p = min(arg2, T0)
            return s2_F(t,  argp)   - s2_F(t, arg2p) 
        # elif x <= x0 + math.sqrt(3)*(t-T0)/3.0 and x >= x0 - math.sqrt(3)*(t-T0)/3.0:
        #     arg = min(t - (x-x0)*3/math.sqrt(3),T0)
        #     arg2 = max(t - (x+x0)*3/math.sqrt(3),0)
        #     return s2_F(t,  arg)   - s2_F(t, arg2) 

            
        elif x > x0 + math.sqrt(3)*(t-T0)/3.0:
            arg = max(t - (x-x0)*3/math.sqrt(3),0)
            arg2 = max(t - (x+x0)*3/math.sqrt(3),0)
            arg = min(arg, edge)
            arg2 = min(arg2, edge)
            print(arg, arg2)
            print('here')
            return s2_F(t,  arg) - s2_F(t, arg2)

        # elif x < x0 + math.sqrt(3)*(t-T0)/3.0:
            
        elif x < x0 - math.sqrt(3)*(t-T0)/3.0:
            arg = t - (x0-x)*3/math.sqrt(3)
            if t - (x0-x)*3/math.sqrt(3) <= 0:
                return  2*(s2_F(t,  T0) - s2_F(t, 0))
            elif t - (x0-x)*3/math.sqrt(3) > 0 and t - (x0-x)*3/math.sqrt(3) < T0:
                return 2*(s2_F(t,  T0) - s2_F(t, arg)) + s2_F(t, arg) - s2_F(t, 0)
            else:
                return 0
         
def su_olson_s2_integrand(tau,x,t,x0,t0):
    return  (np.exp(-t + tau)*(-np.heaviside((-3*x - 3*x0 + math.sqrt(3)*(t - tau))/(t - tau),1) - np.heaviside((3*x - 3*x0 + math.sqrt(3)*(t - tau))/(t - tau),1) + np.heaviside((-3*x + 3*x0 + math.sqrt(3)*(t - tau))/(t - tau),1) + np.heaviside((3*x + 3*x0 + math.sqrt(3)*(t - tau))/(t - tau),1)))/2.
    
            
@njit
def uncollided_s2_gaussian(x,t,sigma,t0):
    tf = min(t,t0)

    return (math.exp(-(math.sqrt(3)*x) + (3*sigma**2)/4.)*math.sqrt(3*math.pi)*sigma*(math.erf((-2*t + 2*tf + 2*math.sqrt(3)*x - 3*sigma**2)/(2.*math.sqrt(3)*sigma)) + math.exp(2*math.sqrt(3)*x)*(math.erf(t/(math.sqrt(3)*sigma) + x/sigma + (math.sqrt(3)*sigma)/2.) - math.erf((2*t - 2*tf + 2*math.sqrt(3)*x + 3*sigma**2)/(2.*math.sqrt(3)*sigma))) + math.erf((2*math.sqrt(3)*t - 6*x + 3*math.sqrt(3)*sigma**2)/(6.*sigma))))/4.

    
@njit
def uncollided_s2_gaussian_thick(x,t,sigma,t0):
    return (6*t**5 + t**6 + 12*t**3*(10 + 15*x**2 - 3*sigma**2) + 3*t**4*(10 + 15*x**2 - 3*sigma**2) +  18*t*(40 + 15*x**2*(4 + x**2) - 6*(2 + 3*x**2)*sigma**2 + 6*sigma**4) +  9*t**2*(40 + 15*x**2*(4 + x**2) - 6*(2 + 3*x**2)*sigma**2 + 6*sigma**4) - 9*(-80 - 3*x**2*(40 + 10*x**2 + x**4) + 24*sigma**2 + 9*x**2*(4 + x**2)*sigma**2 - 6*(2 + 3*x**2)*sigma**4 + 18*sigma**6) + math.exp(t0)*(-t**6 + 6*t**5*(-1 + t0) + 6*t0**5 - t0**6 + 12*t0**3*(10 + 15*x**2 - 3*sigma**2) - 3*t0**4*(10 + 15*x**2 - 3*sigma**2) - 3*t**4*(10 + 5*(-2 + t0)*t0 + 15*x**2 - 3*sigma**2) + 4*t**3*(-30 + 5*t0*(6 + (-3 + t0)*t0) + 45*(-1 + t0)*x**2 - 9*(-1 + t0)*sigma**2) + 18*t0*(40 + 15*x**2*(4 + x**2) - 6*(2 + 3*x**2)*sigma**2 + 6*sigma**4) - 9*t0**2*(40 + 15*x**2*(4 + x**2) - 6*(2 + 3*x**2)*sigma**2 + 6*sigma**4) + 9*(-80 - 3*x**2*(40 + 10*x**2 + x**4) + 24*sigma**2 + 9*x**2*(4 + x**2)*sigma**2 - 6*(2 + 3*x**2)*sigma**4 + 18*sigma**6) - 3*t**2*(5*(24 + t0*(-24 + t0*(12 + (-4 + t0)*t0))) + 45*x**4 - 18*(2 + (-2 + t0)*t0)*sigma**2 + 18*sigma**4 + 18*x**2*(10 + 5*(-2 + t0)*t0 - 3*sigma**2)) + 6*t*(-120 + t0*(120 + t0*(-60 + t0*(20 + (-5 + t0)*t0))) + 45*(-1 + t0)*x**4 - 6*(-6 + t0*(6 + (-3 + t0)*t0))*sigma**2 + 18*(-1 + t0)*sigma**4 + 6*x**2*(-30 + 5*t0*(6 + (-3 + t0)*t0) - 9*(-1 + t0)*sigma**2))))/(162.*math.exp(t)*sigma**6)





@njit        
def problem_identifier(source_type):
    if source_type[0] == 1:
        problem_type = 'plane_IC'
    elif source_type[1] == 1:
        problem_type = 'square_IC'
    elif source_type[2] == 1:
        problem_type = 'square_source'
    elif source_type[3] == 1:
        problem_type = 'gaussian_IC'
    elif source_type[4] == 1:
        problem_type = 'gaussian_source'

    else:
        problem_type =='none'
    return problem_type


# xs  = np.linspace(-1.5,1.5, 100)
# phi = xs*0
# phi_u = xs*0
# tf = 1.0
# expi1 = xs*0
# expi2 = xs*0

# for i in range(len(xs)):
#     expi1[i] = numba_expi(xs[i])
#     expi2[i] = expi(xs[i])

#     # phi_u[i] = uncollided_square_s2(xs[i], tf, 0.5, tf)
# plt.figure(1)

# plt.plot(xs, expi1, "--")
# plt.plot(xs, expi2, ":")

@njit
def uncollided_su_olson_s2_2(x,t,x0,t0):
    T0 = min(t, t0)
    if t > t0:
        x = abs(x)
        if x0 - math.sqrt(3)*(t-T0)/3.0 <= abs(x) <= x0 + math.sqrt(3)*(t-T0)/3.0:
            arg = max(T0 - (abs(x)-x0)*3/math.sqrt(3),0)
            return s2_F(t,  T0) - s2_F(t, 0) 
        
        
        if abs(x) < x0 - math.sqrt(3)*(t-T0)/3.0:
            arg = t - (x0-x)*3/math.sqrt(3)
            
            if t - (x0-x)*3/math.sqrt(3) <= 0:
                return  2*(s2_F(t,  T0) - s2_F(t, 0))
            
            
            elif t - (x0-x)*3/math.sqrt(3) > 0 and t - (x0-x)*3/math.sqrt(3) < T0:
                return 2*(s2_F(t,  T0) - s2_F(t, arg)) + s2_F(t, arg) - s2_F(t, 0)
            
            

            
        elif x > x0 + math.sqrt(3)*(t-T0)/3.0:
            arg = max(t - (x-x0)*3/math.sqrt(3),0)
            return s2_F(t,  arg) - s2_F(t, 0) 
       
        
    else:
        return 0.0

def test_s2_sol(t = 10, t0 = 10):
    import scipy.integrate as integrate
    
    # xs = np.linspace(0, t+0.5, 500)
    xs = np.linspace(50,60, 500)
    phi = xs*0
    phi_test = xs*0
    phi_exact = xs*0
    x0 = 0.5 
    for ix in range(xs.size):
        phi[ix] = uncollided_su_olson_s2(xs[ix],t, x0, t0)
        # phi_test[ix] = uncollided_su_olson_s2_2(xs[ix],t, x0, t0)
        phi_exact[ix] = integrate.quad(su_olson_s2_integrand, 0, min(t,t0), args = (xs[ix],t,x0,t0))[0]
    
    # plt.plot(xs, phi, '-ob')
    # plt.plot(xs, phi_exact, '-k')
    # plt.plot(xs, phi_test, '-or', mfc = 'none')
    # plt.axvline(x = x0 + math.sqrt(3)*(t-t0)/3.0, color = 'r')
    # t - (x0-x)*3/math.sqrt(3)
    
    print(np.sqrt(np.mean(phi_exact-phi)**2), 'RMSE')
    # show('uncollided_su_olson_s2_t_10')
    # plt.show()

def test_square_sol(t = 31.6228, t0 = 10):
    import scipy.integrate as integrate
    
    xs = np.linspace(0, 20, 1000)
    phi = xs*0
    phi_test = xs*0
    phi_exact = xs*0
    x0 = 0.5
    for ix in range(xs.size):
        phi[ix] = uncollided_square_s2(xs[ix],t, x0, t0)

    
    # plt.plot(xs, phi, '-ob')
    # plt.plot(xs, phi, '-k')
    # plt.plot(xs, phi_test, '-or', mfc = 'none')
    
    # show('uncollided_square_s2t')
    # plt.show()

# def time_step_counter(t, division):
@njit    
def heaviside_vector(x):
    return_array = np.ones(x.size)
    for ix, xx in enumerate(x):
        if xx < 0:
            return_array[ix] = 0.0
    return return_array
    
@njit    
def heaviside_scalar(x):
    returnval = 1.0
    if x < 0:
        returnval = 0.0
    return returnval
    
@njit
def shaper(angles, spaces, M, thermal):
    if thermal == True:
        return np.array([angles + 1 , spaces, M+1])
    else:
        return np.array([angles, spaces, M+1])
    
@njit 

def eval_Tn(n,x):
    if n == 0:
        return  1 + x*0 
    elif n == 1:
        return x
    elif n == 2:
        return -1 + 2*x**2
    elif n == 3:
        return -3*x + 4*x**3
    elif n == 4:
        return 1 - 8*x**2 + 8*x**4
    elif n == 5:
        return 5*x - 20*x**3 + 16*x**5
    elif n == 6:
        return -1 + 18*x**2 - 48*x**4 + 32*x**6
    elif n == 7:
        return -7*x + 56*x**3 - 112*x**5 + 64*x**7
    elif n == 8:
        return 1 - 32*x**2 + 160*x**4 - 256*x**6 + 128*x**8
    elif n == 9:
        return 9*x - 120*x**3 + 432*x**5 - 576*x**7 + 256*x**9
    elif n == 10:
        return -1 + 50*x**2 - 400*x**4 + 1120*x**6 - 1280*x**8 + 512*x**10
    elif n == 11:
        return -11*x + 220*x**3 - 1232*x**5 + 2816*x**7 - 2816*x**9 + 1024*x**11
    elif n == 12:
        return 1 - 72*x**2 + 840*x**4 - 3584*x**6 + 6912*x**8 - 6144*x**10 + 2048*x**12
    elif n > 12:
        print('not implemented to this order yet')
        assert(0)
    else:
        raise ValueError('j must be a positive integer')

@njit('float64(int64)')
def kronecker(i):
    if i == 0:
        return 1.0
    else:
        return 0.0
    
@njit('float64[:](int64, float64[:], float64, float64)')
def normTn(n,x,a=0,b=1.0):
    tmp = 0*x
    norm = (1/ math.sqrt(2))**kronecker(n) * math.sqrt(1/(b-a)) * math.sqrt(2) / math.sqrt(math.pi) 
    for count in range(x.size):
        xx = x[count]
        # tmp[count] = sc.eval_legendre(n,z)*fact
        z = 2/(b - a) * xx - (b + a)/(b - a)

        tmp[count] = norm * eval_Tn(n, z)
    return tmp 
@njit
def normTn_intcell(j, a,b):
    if j ==0:
        return (math.sqrt(1/(-a + b))*(-a**3 + b**3))/(3.*math.sqrt(math.pi))
    elif j == 1:
        return ((a - b)**2*(a + b)*math.sqrt(-(1/(2*a*math.pi - 2*b*math.pi))))/3.
    elif j == 2:
        return (math.sqrt(2)*(a - b)*(a**2 + 3*a*b + b**2)*math.sqrt(-(1/(a*math.pi - b*math.pi))))/15.
    elif j == 3:
        return -0.2*((a - b)**2*(a + b)*math.sqrt(-(1/(2*a*math.pi - 2*b*math.pi))))
    elif j == 4:
        return (math.sqrt(2)*(a - b)*(5*a**2 - 3*a*b + 5*b**2)*math.sqrt(-(1/(a*math.pi - b*math.pi))))/105.



@njit('float64[:](float64[:], float64, float64)')
def weight_func_Tn(x, a, b):
    return (b-a) / np.sqrt((a-x) * (x-b))

@njit 
def angular_deriv(N_ang, angle, mus, V_old, space):

    if angle != 0 and angle != N_ang -1:
        h = (mus[angle+1] - mus[angle]) / (mus[angle]-mus[angle-1])
        right = (1-mus[angle+1]**2)*V_old[angle+1,space,:]
        middle = (1-mus[angle]**2)*V_old[angle,space,:]
        left = (1-mus[angle-1]**2)*V_old[angle-1,space,:]
        dterm =  (right - h**2 * middle - (1-h**2) * left) / (mus[angle+1]-mus[angle]) / (1+h) 
    
    elif angle == 0:
        h = mus[angle+1] - mus[angle]
        right = (1-mus[angle+1]**2)*V_old[angle+1,space,:]
        middle = (1-mus[angle]**2)*V_old[angle,space,:]
        dterm = (right - middle) / h

    elif angle == N_ang - 1:
        h = mus[angle] - mus[angle-1]
        right = (1-mus[angle]**2)*V_old[angle,space,:]
        middle = (1-mus[angle-1]**2)*V_old[angle-1,space,:]
        dterm = (right - middle) / h

    return dterm


@njit
def finite_diff_uneven_2(x, ix, u, left = False, right = False):

    if left == False and right == False:
        h = (x[ix+1] - x[ix]) / (x[ix] - x[ix-1])
        res = (u[ix + 1] - h**2 * u[ix-1] - (1-h**2) * u[ix]) / (x[ix+1]- x[ix]) / (1+h)
    
    elif left == True:
        h = x[ix+1] - x[ix]
        right = u[ix+1]
        middle = u[ix]
        res = (right - middle) / h 
        # xghost = x[ix] - (x[ix+1]-x[ix])
        # h = (x[ix+1] - x[ix]) / (x[ix] - xghost)
        # res = (u[ix + 1] - h**2 * 0 - (1-h**2) * u[ix]) / (x[ix+1]- x[ix]) / (1+h)

    elif right == True:
        h = x[ix] - x[ix-1]
        right = u[ix]
        middle = u[ix-1]
        res = (right - middle) / h 
        # xghost = x[ix] + (x[ix] - x[ix-1])
        # h = (xghost - x[ix]) / (x[ix] - x[ix-1])
        # res = (0 - h**2 * u[ix-1] - (1-h**2) * u[ix]) / (xghost- x[ix]) / (1+h)
    return res

@njit
def finite_diff_uneven_diamond(x, ix, psi, left = False, right = False, origin = False):
    # if left == False and right == False:
    #     h = (x[ix+1] - x[ix]) / (x[ix] - x[ix-1])
    #     res = (u[ix + 1] - h**2 * u[ix-1] - (1-h**2) * u[ix]) / (x[ix+1]- x[ix]) / (1+h)
    
    # elif left == True:
    # if right != True:
        if left == False and right == False:
            psip = 0.5 * (psi[ix+1] + psi[ix])
            psim = 0.5 * (psi[ix] + psi[ix-1])
            # mup = (x[ix+1] - x[ix])*0.5 + x[ix]
            mup = (x[ix+1] + x[ix]) * 0.5
            # mum = (x[ix] - x[ix-1])*0.5 + x[ix-1]
            mum = (x[ix] + x[ix-1]) * 0.5
            deltamu = mup - mum
            res = ((1-mup**2) * psip - (1-mum**2) * psim) / deltamu
        elif left == True:
            psip = 0.5 * (psi[ix+1] + psi[ix])
            # mup = (x[ix+1] - x[ix])*0.5 + x[ix]
            mup = (x[ix+1] + x[ix]) * 0.5
            mum =  -1
            deltamu = mup - mum
            res = ((1-mup**2) * psip ) / deltamu
        elif right == True:
            mup = 1
            # mum =  (x[ix] - x[ix-1])*0.5 + x[ix-1]
            mum = (x[ix] + x[ix-1]) * 0.5
            psim = 0.5 * (psi[ix] + psi[ix-1])
            deltamu = mup - mum
            res = (-(1-mum**2) * psim ) / deltamu
        return res


@njit 
def alpha_difference(alphasp1, alphasm1, w, psionehalf, V_old, left, right, origin):

    res = 1/w * (2 * alphasp1 * V_old - (alphasp1 + alphasm1) * psionehalf)
    return res 



@njit
def finite_diff_uneven_diamond_2(x, ix, psi, alphams, ws, left = False, right = False):
    # if left == False and right == False:
    #     h = (x[ix+1] - x[ix]) / (x[ix] - x[ix-1])
    #     res = (u[ix + 1] - h**2 * u[ix-1] - (1-h**2) * u[ix]) / (x[ix+1]- x[ix]) / (1+h)
    
    # elif left == True:
    # if right != True:
    # ws = 2 * ws
    alpham = alphams[ix]
    alphap = alphams[ix + 1]
    assert(abs(alphap - (alpham - ws[ix]*x[ix])) <= 1e-8)
    if left == False and right == False:
        psip = 0.5 * (psi[ix+1] + psi[ix])
        psim = 0.5 * (psi[ix] + psi[ix-1])
        # mup = (x[ix+1] + x[ix])*0.5 
        # mum = (x[ix] - x[ix-1])*0.5 + x[ix-1] 
    elif left == True:
        #  alpham = 0.0
         assert(alpham == 0.0)
         psip = 0.5 * (psi[ix+1] + psi[ix])
        #  mup = (x[ix+1] + x[ix])*0.5 
        #  mum =  -1
         psim = psi[ix]
    elif right == True:
        #  mup = 1.0
        #  mum =  (x[ix] + x[ix-1])*0.5 
         psim = 0.5 * (psi[ix] + psi[ix-1])
         psip = 0.5 * (0.0 + psi[ix])
    

    res = (2 / ws[ix]) * (alphap * psip - alpham * psim)
    # res = (2 / ws[ix]  ) * (2 * alphap * psi[ix]  - (alphap + alpham) * psim) 
    
    return res 

# @njit
# def gauss_quadrature(integrand, xs_quad, ws_quad, a, b):
#      argument = (b-a)/2*self.xs_quad + (a+b)/2
#     mu = self.mus[ang]
#     self.IC[ang,space,j] = (b-a)/2 * np.sum(self.ws_quad * ic.function(argument, mu) * Tn(j, argument, a, b) * weight_func_Tn(argument, a, b))
        
        
def quadrature(n, name, testing = False):
    ws = np.zeros(n)
    xs = np.zeros(n)
    # roots, weights = roots_legendre(n-1)
    roots = np.zeros(n)
    if name == 'gauss_legendre':
        xs, ws = poly.legendre.leggauss(n)
    elif name == 'chebyshev':
        ws = np.zeros(n)
        xs = np.zeros(n)
        pi = math.pi
        for i in range(1,n+1):
            xs[i-1] = math.cos((2*i-1)/2/n * pi)
            ws[i-1] = pi/n
        xs = np.sort(xs)


    elif name == 'gauss_lobatto':
        if n > 1:
            # brackets = sps.legendre(n-1).weights[:, 0]
            xs_brackets, blanl = poly.legendre.leggauss(n-1)
            brackets = xs_brackets
        else:
            brackets = np.array([-1,1])
        for i in range(n-2):
            # roots[i+1] = bisection(partial(eval_legendre_deriv, n-1), brackets[i], brackets[i+1])
            x0 = (brackets[i]+ brackets[i+1])*0.5
            roots[i+1] =  newtons(x0, partial(eval_legendre_deriv, n-1), partial(eval_second_legendre_deriv, n-1))

        xs = roots
        xs[0] = -1
        xs[-1] = 1
        if n ==2:
            ws[0] = 1
            ws[1] = 1
        else:
            for nn in range(1, n-1):
                inn = nn + 1
                ws[nn] = 2 / (n*(n-1)) / (sps.eval_legendre(n-1, roots[nn]))**2
                ws[0] = 2/ (n*(n-1))
                ws[-1] = 2/ (n*(n-1))
        # if testing == True:
        #     testxs = quadpy.c1.gauss_lobatto(n).points
        #     testws = quadpy.c1.gauss_lobatto(n).weights
        #     np.testing.assert_allclose(testxs, xs)
        #     np.testing.assert_allclose(testws, ws)


        # if testing == True:
        #     # testxs = quadpy.c1.gauss_legendre(n).points
        #     # testws = quadpy.c1.gauss_legendre(n).weights
        #     np.testing.assert_allclose(testxs, xs)
        #     np.testing.assert_allclose(testws, ws)
    return xs, ws    
# @njit
def bisection(f, a, b, tol=1e-14):
    assert np.sign(f(a)) != np.sign(f(b))
    while b-a > tol:
        m = a + (b-a)/2
        fm = f(m)
        if np.sign(f(a)) != np.sign(fm):
            b = m
        else:
            a = m
            
    return m



def eval_legendre_deriv(n, x):
    return (
        (x*sps.eval_legendre(n, x) - sps.eval_legendre(n-1, x))
        /
        ((x**2-1)/n))

def eval_second_legendre_deriv(n, x):
    return (n*(-((1 + x**2)*sps.eval_legendre(n, x)) + 2*x*sps.eval_legendre(n-1, x) + (-1 + x**2)*(x*eval_legendre_deriv(n, x) - eval_legendre_deriv(n-1, x))))/(-1 + x**2)**2

def newtons2(x0, f, fprime, tol = 1e-14):
    old_guess = x0
    new_guess = 1000
    it = 0
    while abs(old_guess-new_guess) > tol:
        new_guess = old_guess - f(old_guess) / fprime(old_guess)
        old_guess = new_guess
    return old_guess

def newtons(x0, f, fprime, tol = 1e-14):
    def iterate(x0, f, fprime):
        return x0 - f(x0) / fprime(x0)
    tol_met = False
    while tol_met == False:
        new_x0 = iterate(x0, f, fprime)
        if abs(new_x0-x0) <= tol:
            tol_met = True
        x0 = new_x0
    return x0

@njit
def sqrt_two_mass_func(i, j):
    rttwo = math.sqrt(2)
    if ((i == 0) and (j == 0)) or ((i != 0) and (j != 0)):
        return 1.0
    elif (i == 0) and (j != 0):
        if j%2 == 0:
            return rttwo
        else:
            return 1/rttwo
    elif (j == 0) and (i!=0):
        if i%2 == 0:
            return rttwo
        else:
            return 1/rttwo
    else:
        assert(0)

@njit
def rttwo_mistake_undoer(i,j):
    if ((i == 0) and (j == 0)) or ((i != 0) and (j != 0)):
        return 1.0
    elif (i == 0) and (j != 0):
        if j%2 == 0:
            return 2.0
        else:
            return 1.0
    elif (j == 0) and (i!=0):
        if i%2 == 0:
            return 2.0
        else:
            return 1.0
    else:
        assert(0)
        

@njit
def converging_r(t_dim, sigma_func):
    # rfront = 0.01 * (-t_dim) ** 0.679502
    rfront = 9.0
    if sigma_func['test1'] == True:
        rfront = 1e-4 * (-t_dim) ** 0.679501
    elif sigma_func['test2'] == True:
        rfront = 0.005 * (-t_dim) ** 0.51765
    elif sigma_func['test3'] == True:
        rfront = 1e-4 * (-t_dim) ** 1.1157536
    elif sigma_func['test4'] == True:
        rfront = (- t_dim) ** 0.462367
    # else:
    #     raise ValueError('no function selected')
    return rfront 

@njit
def xi_converging(rf, r):
    return r / rf

@njit
def W_converging(xi, sigma_func):
    if sigma_func['test1'] == True:
        if 1 <= xi <= 2:
            res =  (xi-1)**0.4057 * (1.521 - 0.3762 * xi + 0.06558 * xi **2)
        elif xi > 2:
            res =  (xi-1)**0.2955 * (1.082 - 0.02718 * xi + 0.001055 * xi **2)
    elif sigma_func['test2'] == True:
        if 1 <= xi <= 2:
            res =  (xi-1)**0.3977 * (1.244 - .1757 * xi + 0.03186 * xi **2)
        elif xi > 2:
            res =  (xi-1)**0.3401 * (1.021 - 0.0007123 * xi + 0.0001726 * xi **2)
    elif sigma_func['test3'] == True:
        if 1 <= xi <= 2:
            res =  (xi-1)**0.3575 * (1.979 - .6195 * xi + 0.1106 * xi **2)
        elif xi > 2:
            res =  (xi-1)**0.2101 * (1.27 - 0.04707 * xi + 0.001797 * xi **2)
    elif sigma_func['test4'] == True:
        if 1 <= xi <= 2:
            res =  (xi-1)**1.141 * (0.2251 + 0.127 * xi + 0.001626 * xi **2)
        elif xi > 2:
            res =  (xi-1)**1.102 * (0.1846 + 0.1505 * xi + 0.00004394 * xi **2)
        else:
            res = 0
    else:
        assert 0
    return res
        

@njit
def ts_converging(t, sigma_func):
    # surface temperature. Units in HeV
    rf = converging_r(t, sigma_func)
    
    if sigma_func['test1'] == 1:
        R = 1e-3
        xi = xi_converging(rf, R)
        res = 1.34503465 * (-t) ** 0.0920519 * W_converging(xi, sigma_func) ** (5/8)
    elif sigma_func['test2'] == 1:
        R = 0.05
        xi = xi_converging(rf, R)
        res = 0.809892 * (-t) ** 0.100238 * W_converging(xi, sigma_func) ** .5
    elif sigma_func['test3'] == 1:
        R = 1e-3
        xi = xi_converging(rf, R)
        res = 1.1982 * (-t) ** 0.027639 * W_converging(xi, sigma_func) ** .5
    elif sigma_func['test4'] == 1:
        R = 10
        xi = xi_converging(rf, R)
        res = .552154 * (-t) ** .242705 * W_converging(xi, sigma_func) ** .25 #KeVc
    return res

@njit
def V_converging(xi, sigma_func):
    if sigma_func['test1'] == True:
        res = 0.4345 * xi ** -2.752 + .2451 * xi ** -1.454
    elif sigma_func['test2'] == 1:
        res = 0.262 * xi ** -3.24 + .2558 * xi ** -1.88
    elif sigma_func['test3'] == 1:
        res = 0.8879 * (xi+1e-14) ** -2.233 + .2278 * (xi+1e-10) ** -1.037
    elif sigma_func['test4'] == 1:
        res = 0.06247 * xi ** -3.836 + 0.3999 * xi **-2.157
    return res 

@njit
def T_bath(t, sigma_func):
    rf = converging_r(t, sigma_func)
    if sigma_func['test1'] == True:
        R = 1e-3
        xi = xi_converging(rf, R)
        LAMBDA = xi * V_converging(xi, sigma_func) * W_converging(xi, sigma_func) ** -1.5
        res = (1 + 0.103502* LAMBDA * (-t) ** -.541423)**0.25 * ts_converging(t, sigma_func)
    elif sigma_func['test2'] == 1:
        R = 0.05  
        xi = xi_converging(rf, R)
        LAMBDA = xi ** 1.2 * V_converging(xi, sigma_func) * W_converging(xi, sigma_func)**-1
        res = (1 + 0.385372 * LAMBDA * (-t) ** -0.579294) ** .25 * ts_converging(t, sigma_func)
    elif sigma_func['test3'] == 1:
        R = 1e-3
        xi = xi_converging(rf, R)
        LAMBDA = xi**.6625 * V_converging(xi, sigma_func) * (W_converging(xi, sigma_func)+1e-13) ** -1
        res = (1+0.075821* LAMBDA * (-t+1e-13) ** -.316092)**.25 * ts_converging(t, sigma_func)

    elif sigma_func['test4'] == 1:
        R = 10
        xi = xi_converging(rf, R)
        LAMBDA = xi * V_converging(xi, sigma_func)
        res = (1 + 0.083391*LAMBDA*(-t)**-.537633)**.25 * ts_converging(t, sigma_func) 
    else:
        assert(0)
    return res



        
    
@njit
def converging_time_function(t, sigma_func):
    t_dim = t/29.98
    t_init = -29.6255
    if sigma_func['test2'] == True:
        t_init = -85.4678 
    elif sigma_func['test1'] == True:
        t_init = -29.625647 
    elif sigma_func['test3'] == True:
        t_init = -7.875848
    elif sigma_func['test4'] == True:
        t_init = -145.47339
    
    return t_dim + t_init

    

# @njit 
# def positivity_enforcer():
# @njit
# def psi_old_maker(vector, xs, mesh):

#     cells = vector[:,0].size()
#     J = vector[:,0].size()
#     res = np.zeros(cells)
#     for ix in range(cells):
#         a = mesh[cells]
#         b = mesh[cells+1]
#         z = (b-a)/2 * xs + (a+b)/2
#         for j in range(j):
#             res[cells] += normTn(j,xs,mesh[cells], mesh[cells+1])
#     return res


@njit 
def make_u_old(vector, edges_old, a, b, xs_quad, ws_quad, M):
    res = np.zeros(M+1)

    z = (b-a)/2 * xs_quad + (a+b)/2

    psi = z*0
    for iz, zz in enumerate(z):

        ie = np.searchsorted(edges_old, zz) 
        if edges_old[ie] > zz:
            ie -= 1
        if edges_old[ie+1] < zz:
            ie -= 1

        for j in range(M+1):
            psi[iz] += normTn(j,z[iz:iz+1],edges_old[ie], edges_old[ie+1])[0] * vector[ie, j]

    
    for i in range(M+1):
        res[i] =  (b-a)/2 * np.sum(ws_quad * psi * normTn(i, z, a, b))
    
    return res
@njit
def mass_lumper(Mass, a, b, invert = True):
            M = Mass[0].size -1 
            mass_lumped = np.zeros((M+1, M+1))
            mass_lumped_inv = np.zeros((M+1, M+1))
            MI = np.zeros((M+1, M+1))
            if M != 1:
                raise ValueError('The lumped mass matrix is sigular for M >1 and there is no reason to lump the M=0 equations')
            for i in range(M+1):
                for j in range(M+1):
                    mass_lumped[i,i] += Mass[i, j]
            if invert == True:
                for i in range(M+1):
                    mass_lumped_inv[i,i] = 1./mass_lumped[i,i]
                return mass_lumped, mass_lumped_inv
            else:
                return mass_lumped, mass_lumped_inv
@njit
def integrate_phi_cell(cs, ws, a, b, M, N_ang):
    # cell_volume = 4 * math.pi * (b**3 - a**3)
    psi = np.zeros(N_ang)
    for l in range(N_ang):
        for j in range(M+1):
            psi[l] += cs[l, j] * normTn_intcell(j, a, b)
    res = np.sum(np.multiply(psi,ws))
    return 4 * math.pi * res #* cell_volume


@njit
def normalize_phi(VV, edges, ws, N_ang, M, N_space, N_groups):
    norm_phi = np.zeros(N_space)
    for ig in range(N_groups):
        for ix in range(N_space):
            norm_phi[ix] += integrate_phi_cell(VV[ig * N_ang: (ig+1) * N_ang, ix, :], ws, edges[ix], edges[ix+1], M, N_ang)
    return np.sum(norm_phi) # * sigma_f * nu * chi