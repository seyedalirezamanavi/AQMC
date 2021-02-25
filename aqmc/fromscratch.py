#in this module we emplement methods to calculate the Greens function
#from scratch, using G = (I + B_l... B_1)^(-1)

import numpy as cp

eps = 10**(-15)

def from_scratch(cl,n_qr):
    Q,D,T = QR_prime(cl, n_qr)
    G = invers(Q, D, T)
    return G 


def QR(z, I):
    Q = I
    D = cp.diag(I)
    T = I
 
    for i in range(len(z)):
        A = (z[len(z) - i - 1]).dot(Q*D)
        Q, R = cp.linalg.qr(A)
        D = cp.diag(R)
        T_prime = R / (D[:,None]+eps)
        T = T_prime.dot(T)
 
    return Q, D, T

 
def QR_prime(z, n):
    len_z,len_z0,_ = z.shape
    m = len_z // n
    if len_z % n != 0:
        m += 1
    z_0 = cp.empty((m,len_z0,len_z0))
    A0 = cp.eye(len_z0)
    A = A0
    for i in range(len_z):
        A = A.dot(z[i])
        if i%n == n-1:
            z_0[int(i/n),...] = A
            A = A0
    if len(z) % n != 0:
        z_0[int(i/n),...] = A
    usvh = QR(z_0,A0)
    return usvh[0], usvh[1], usvh[2]
 
 
def invers(Q, D, T):
    D_p = cp.where(cp.abs(D)>1,1,D)
    D_m = cp.where(cp.abs(D)<1,1,D)
    A1 = cp.conj(Q.T)/(D_p[:,None]+eps) + D_m[:,None]*T
    A2 = 1/(D_p+eps)
    A3 = cp.conj(Q.T)
    g = (cp.linalg.inv(A1)*A2).dot(A3)
    
    return g
 
