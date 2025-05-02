import numpy as np
from numpy.linalg import svd


def VDMD2(Y_minus, Y_plus, skip):

    [U,S,V] = svd(Y_minus[:,skip:],full_matrices=False)
    print(S, 'singular values vector')
    Sinv = np.zeros(S.size)
    Spos = S[S/np.cumsum(S)>1e-13]
    Sinv[0:Spos.size] = 1.0/Spos.copy()
    tmp=np.dot(U.transpose(),Y_plus[:, skip:])
    tmp2=np.dot(tmp,V.transpose())
    tmp3=np.dot(tmp2,np.diag(Sinv))
    deigs = np.linalg.eigvals(tmp3)
    #deigs = deigs[deigs>0]
    #print(np.log(deigs)/dt)
    # print(Y_minus, 'Y-')
    # print(Y_plus, 'Y+')
    # print('############################')
    # print(deigs, 'eigen values')
    # if (np.real(deigs) >0).any():
        # print('positive eigen val', np.max(np.real(deigs)))
    # print('############################')
    return np.real(deigs)
    # return np.array([0.0])
