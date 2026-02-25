import numpy as np
from numpy.linalg import svd


def VDMD2(Y_minus, Y_plus, skip):

    [U,S,V] = svd(Y_minus[:,skip:],full_matrices=False)
    print(S, 'singular values vector')
    # S = S[:8]
    # U = U[:8]
    # V = V[:8]
    Sinv = np.zeros(S.size)
    Spos = S[S/np.cumsum(S)>1e-15]
    Sinv[0:Spos.size] = 1.0/Spos.copy()
    tmp=np.dot(U.transpose(),Y_plus[:, skip:])
    tmp2=np.dot(tmp,V.transpose())
    tmp3=np.dot(tmp2,np.diag(Sinv))
    deigs = np.linalg.eigvals(tmp3)
    # r = Spos.size
    # Ur  = U[:, :r]               # m x r
    # Sr  = S[:r]                  # r
    # Vr = V.conj().T[:, :r] 

    eigvals, W = np.linalg.eig(tmp3)
    # print('A_tilde eigenvalues', eigvals)
    print(Y_plus[:, skip:].shape)
    print(V.shape)
    print(V.T.shape)
    print(W.shape)
    modes = Y_plus[:, skip:] @ V.conj().transpose() @ np.diag(1.0 / Spos) @ W
    Sigma = np.diag(Spos)
    Atilde = U.T @ Y_plus[:, skip:] @ V @ np.linalg.inv(Sigma)
    nrows, ncols = Atilde.shape
    k = min(U.shape[1], nrows, ncols)
    #deigs = deigs[deigs>0]
    #print(np.log(deigs)/dt)
    # print(Y_minus, 'Y-')
    # print(Y_plus, 'Y+')
    # print('############################')
    # print(deigs, 'eigen values')
    # if (np.real(deigs) >0).any():
        # print('positive eigen val', np.max(np.real(deigs)))
    # print('############################')
    Af = U[:, :k] @ Atilde @ U.conj().T[:k, :] 
    return np.real(deigs), modes, Af
    # return np.array([0.0])
