#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:42:00 2022

@author: rmcclarr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:50:11 2022

@author: rmcclarr
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.special import jv
from numpy.linalg import svd
from pydmd import DMD
from tqdm import tqdm
from scipy.interpolate import interp1d as interp
from scipy import integrate
# from .theta_DMD import theta_DMD

Nt = 1000

dt = 0.01
Yminus = np.zeros((2,Nt-1))
Yplus = np.zeros((2,Nt-1))

# dts = np.logspace(-4,-1,Nt)
dts = np.linspace(10e-4, 0.1, Nt)
# dts = dt*np.ones(Nt)
trange = np.cumsum(dts)
yval = np.zeros(2)
yval[0]=1
omega = 1
lam = -1e-1

#set up matrix
A = np.zeros((2,2))
A[0,1] = 1.0
A[1,0] = -omega
A[1,1] = 0
print("A's eigenvalues", np.linalg.eigvals(A))
Yold = np.zeros((2,Nt-1))
#Backward Euler
for i in range(Nt-1):
    dt = dts[i]
    A[1,1] = -1/trange[i+1]
    update = np.linalg.inv(np.identity(2) - dt*A)
    yold = yval.copy() + np.random.rand(2) * 0.05
    yval = np.dot(update,yval)
    Yminus[:,i] = yval.copy()
    Yplus[:,i] = (yval-yold)/dt
    Yold[:, i] = yold.copy()
    # print(Yplus, 'Y+')
# Yminus += np.random.rand(2, Yold[0, :].size) * 0.05
# Yold += np.random.rand(2, Yold[0, :].size) * 0.05
plt.ion()
plt.plot(trange[0:-1],Yminus[0,:])
plt.plot(trange[0:-1],Yold[0,:])
plt.plot(trange,jv(0,trange))
plt.show()


skip = 990

[U,S,V] = svd(Yminus[:,skip:],full_matrices=False)
Sinv = np.zeros(S.size)
Spos = S[S/np.cumsum(S)>1e-18]
Sinv[0:Spos.size] = 1.0/Spos.copy()
tmp=np.dot(U.transpose(),Yplus[:, skip:])
tmp2=np.dot(tmp,V.transpose())
tmp3=np.dot(tmp2,np.diag(Sinv))
deigs = np.linalg.eigvals(tmp3)
#deigs = deigs[deigs>0]
#print(np.log(deigs)/dt)
print(deigs, 'Euler')


# #Crank-Nicolson

# yval = np.zeros(2)
# yval[0]=1
# omega = 1.0
# Yminus = np.zeros((2,Nt-1))
# Yplus = np.zeros((2,Nt-1))

# for i in range(Nt-1):
#     dt = dts[i]
#     A[1,1] = -2/(trange[i+1] + trange[i])
#     update = np.linalg.inv(np.identity(2) - 0.5*dt*A)
#     yold = yval.copy()
#     yval = np.dot(update,yval + 0.5*dt*np.dot(A,yval))  
#     Yminus[:,i] = 0.5*(yval + yold)
#     Yplus[:,i] = (yval-yold)/dt
    
# plt.plot(trange[0:-1],Yminus[0,:],'--')
# [U,S,V] = svd(Yminus[:,skip:],full_matrices=False)
# Sinv = np.zeros(S.size)
# Spos = S[S/np.cumsum(S)>1e-18]
# Sinv[0:Spos.size] = 1.0/Spos.copy()
# tmp=np.dot(U.transpose(),Yplus[:, skip:])
# tmp2=np.dot(tmp,V.transpose())
# tmp3=np.dot(tmp2,np.diag(Sinv))
# deigs = np.linalg.eigvals(tmp3)
# #deigs = deigs[deigs>0]
# #print(np.log(deigs)/dt)
# print(deigs)


def dphi_matrix(ts, J):
    D = np.zeros((J, J))

    for j in tqdm(range(J)):
        for k in range(J):
            a = ts[j]
            b = ts[j+2]
            aa = ts[k]
            bb = ts[k+2]

            x0 = min(a, aa)
            x1 = max(b, bb)
            # x0 = ts[1]
            # x1 = ts[-1]


            phi = lambda t: basis_quadratic(t, a, b)[0]
            dphi = lambda t: basis_quadratic(t, aa, bb)[1]
            integrand = lambda t: phi(t) * dphi(t)
            D[j, k] = integrate.quad(integrand, x0, x1)[0]
    return D

def phi_matrix(ts, J):
    D = np.zeros((J, J))
    for j in tqdm(range(J)):
        for k in range(J):
            a = ts[j]
            b = ts[j+2]
            aa = ts[k]
            bb = ts[k+2]

            x0 = min(a, aa)
            x1 = max(b, bb)
            # x0 = ts[1]
            # x1 = ts[-1]
            phi = lambda t: basis_quadratic(t, a, b)[0]
            phi2 = lambda t: basis_quadratic(t, aa, bb)[0]
            integrand = lambda t: phi(t) * phi2(t)
            D[j, k] = integrate.quad(integrand, x0, x1)[0]
    return D




def find_coeffs(ts, Y, I):
    J = Y[:,0].size
    cs = np.zeros((J, I))
    for j in tqdm(range(J)):
        Y_interp = interp(ts[1:], Y[j,:])
        for i in range(1,I):
            a = ts[i]
            b = ts[i+2]
            phi = lambda t: basis_quadratic(t, a, b)[0]
            phisquared = lambda t: phi(t) * phi(t)
            normalization = integrate.quad(phisquared, a, b)[0]
            # print(normalization)
            integrand = lambda t: phi(t) * Y_interp(t)
            cs[j, i] = integrate.quad(integrand, a, b )[0]#/3 *4 #/ normalization 
    return cs


@njit
def basis_hat(t, a, b):
    """
    Compactly supported hat function. Returns basis value and derivative
    """
    c = (b + a)/2
    phi = 0
    dphi = 0
    if a <=t <= c:
        phi =(t-a)/ (c-a)
        dphi = 1/(c-a)
    elif c <t <=b:
        phi = (b-t)/ (b-c)
        dphi = -1/(b-c)

        # res = [, -1/(b-c)]
    if t == c:
        dphi += 1

    return phi, dphi
    


@njit
def basis_quadratic(t, a, b):
    phi = 0.0
    dphi = 0.0
    if a <=t<=b:
        phi = -(t-a) * (t-b) /(.25 * (a-b)**2) #/(b/6-a/6)
        dphi = (-2*t + a+ b)/(.25 * (a-b)**2)#/(b/6-a/6)
    if t == a:
        dphi += -(t-a) * (t-b) /(.25 * (a-b)**2)  #/(b/6-a/6)
    elif t == b:
        dphi += -(t-a) * (t-b) /(.25 * (a-b)**2) #/(b/6-a/6)
    # return phi, dphi 
    return basis_hat(t,a,b)


def inner_product(t, delta_t, yplus, yn, yminus, psi):
    fplus = psi(t +delta_t)
    f = psi(t)
    fminus = psi(t-delta_t)
    return delta_t * (fplus * yplus + f * yn + fminus * yminus)/3 * 2



def weak_DMD(Y, ts, delta_ts):
    t_index1 = 400
    t_index_2 = 990

    ts_new = ts[t_index1-1:t_index_2]
    Y_new = Y[:, t_index1:t_index_2]
    ts = ts_new
    Y = Y_new

    # Construct new Y+, Y- matrices
    numbasis = ts.size-2 #+ (ts.size-2)/2
    numbasis2 = int((ts.size-2)/2) * 0
    Yplus = np.zeros(( Y[:,0].size, numbasis))
    Yminus = np.zeros((Y[:,0].size, numbasis))
    cs_left = find_coeffs(ts, Y, numbasis)
    


    D = dphi_matrix(ts, numbasis)
    A = phi_matrix(ts, numbasis)
    print(D, 'D')
    plt.spy(A)
    plt.show()
    J = Y[:,0].size
    cs = np.zeros((J, numbasis))

    for j in range(J):
        cs[j, :] = np.linalg.solve(A, cs_left[j,:])
    # cutting off first and last time point to get rid of oscillations by the boundary
    A_new = A[1:-1, 1:-1]
    D_new = D[1:-1, 1:-1]
    Y =Y[:, 1:-1]
    cs_new = cs[:, 1:-1]
    A = A_new
    D = D_new
    cs = cs_new
    
    ts = ts[1:-1]
    Yminus = np.dot(cs, A)
    Yplus = - np.dot(cs, D)

      
   
        
    # kip = 1990
    plt.show()
    print(Yplus, 'Y+')
    print(Yminus, 'Y-')
    # skip = 4900
    skip = 0
    [U,S,V] = svd(Yminus[:,skip:],full_matrices=False)
    Sinv = np.zeros(S.size)
    Spos = S[S/np.cumsum(S)>1e-18]
    Sinv[0:Spos.size] = 1.0/Spos.copy()
    tmp=np.dot(U.transpose(),Yplus[:, skip:])
    tmp2=np.dot(tmp,V.transpose())
    tmp3=np.dot(tmp2,np.diag(Sinv))
    deigs = np.linalg.eigvals(tmp3)
    #deigs = deigs[deigs>0]
    #print(np.log(deigs)/dt)
    print(deigs, 'Weak VDMD')

    # print( (1-1/deigs))

    # dmd = DMD(svd_rank=12)

    # # Fit the DMD model.
    # # X = (n, m) numpy array of time-varying snapshot data.
    # dmd.fit()
    plt.figure(3)
    Y_recon_1 = np.zeros(cs[0, :].size)
    Y_recon_2 = np.zeros(cs[0, :].size)
    for ix in range(1,cs[0, :].size):
        Y_recon_1[ix] = basis_quadratic(ts[ix+1], ts[ix], ts[ix+2])[0] * cs[0, ix]
        Y_recon_2[ix] = basis_quadratic(ts[ix+1], ts[ix], ts[ix+2])[0] * cs[1, ix]
    plt.plot(ts[2:],Y_recon_1, 'r-.' )
    plt.plot(ts[2:],Y_recon_2, 'r--' )
    # plt.plot(ts[:-2], Yplus[0,:])
    # plt.plot(ts[:-2], Yplus[1,:])
    plt.plot(ts[1:], Y[0, :], 'k-.')
    plt.plot(ts[1:], Y[1, :], 'k--')
    # plt.plot(ts[2:],Y_recon_1/ Y[0, 1:], 'b-.')
    # plt.plot(ts[2:],Y_recon_2/ Y[1, 1:], 'b--')
    # plt.plot(ts[1:], Y[1,:], '--')
    # plt.plot(ts[:-2], Yminus[0,:], ':')
    # plt.plot(ts[:-2], Yminus[1,:], ':')
    plt.show()


weak_DMD(Yold, trange, dts)
plt.show()

