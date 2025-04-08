import numpy as np
from numpy.linalg import svd


def VDMD(Y_minus, Y_plus, skip):
    [U,S,V] = svd(Y_minus[:,skip:],full_matrices=False)
    Sinv = np.zeros(S.size)
    Spos = S[S/np.cumsum(S)>1e-18]
    Sinv[0:Spos.size] = 1.0/Spos.copy()
    tmp=np.dot(U.transpose(),Y_plus[:, skip:])
    tmp2=np.dot(tmp,V.transpose())
    tmp3=np.dot(tmp2,np.diag(Sinv))
    deigs = np.linalg.eigvals(tmp3)
    #deigs = deigs[deigs>0]
    #print(np.log(deigs)/dt)
    print(deigs)
