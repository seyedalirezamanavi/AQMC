# in this module we initialize the hamiltonian and related parts
import numpy as np 
import numpy as cp
from scipy.linalg import expm


def make_hopping(n_x, n_y, periodic_x, periodic_y, tunneling):
    hopping_matrix = np.zeros((n_y * n_x, n_y * n_x))
    if n_y == 1:
        periodic_y = 0
    if n_x == 1:
        periodic_x = 0
 
    if n_x == 2:
        periodic_x = 1
    if n_y == 2:
        periodic_y = 1
 
    for j in range(n_x):
        for i in range(n_y):
            neighbour_1 = n_y * ((j + 1) % n_x) + i
            neighbour_2 = n_y * ((j - 1) % n_x) + i
            neighbour_3 = n_y * j + (i + 1) % n_y
            neighbour_4 = n_y * j + (i - 1) % n_y
 
            hopping_matrix[n_y * j + i][neighbour_1] = -1
            hopping_matrix[n_y * j + i][neighbour_2] = -1
            hopping_matrix[n_y * j + i][neighbour_3] = -1
            hopping_matrix[n_y * j + i][neighbour_4] = -1
 
            if i == 0:
                hopping_matrix[n_y * j + i][neighbour_4] *= periodic_y
            if i == n_y - 1:
                hopping_matrix[n_y * j + i][neighbour_3] *= periodic_y
            if j == 0:
                hopping_matrix[n_y * j + i][neighbour_2] *= periodic_x
            if j == n_x - 1:
                hopping_matrix[n_y * j + i][neighbour_1] *= periodic_x
 
    hopping_matrix *= tunneling
    return hopping_matrix


def cluster(h,Bk,sign_U_interact):
    cl_up = cp.empty((h.shape[0],h.shape[1],h.shape[1]))
    cl_dn = cp.empty((h.shape[0],h.shape[1],h.shape[1]))
    for i in range(h.shape[0]):
        cl_up[i] = Bk[i].dot(cp.diag(cp.exp(h[i,:])))
        cl_dn[i] = Bk[i].dot(cp.diag(-sign_U_interact[i]*h[i,:]))
    return cl_up,cl_dn
 
 
def init_trotter(Beta,N_time,U_eff,H0):
    T_hop = np.concatenate(([1/2], np.ones(N_time-2), [1/2])) * (Beta / N_time)
    U_interact = U_eff * np.ones(N_time)
    sign_U_interact = np.array(np.sign(U_interact))
    T_u = np.concatenate(([1], np.ones(N_time - 2), [0])) * (Beta / N_time)
    H0_array = np.array([H0 for i in range(N_time)])
    lamda = np.arccosh(np.exp(np.abs(U_interact)*T_u/2))
    probability = cp.array(np.exp(-(1 - np.sign(U_interact)) * lamda))
    gamma1 = cp.array(np.exp(-2*lamda)-1)
    gamma2 = cp.array(np.exp( 2*lamda*sign_U_interact)-1)
    return cp.array(sign_U_interact), T_hop, H0_array, cp.array(lamda), probability, gamma1, gamma2
 
 
def expmk(H0_array, T_hop): #cpu
    Bk = []
    Bk_inv = []
    for i in range(len(T_hop)):
        Bk.append(expm(-H0_array[i]*T_hop[i]))
        Bk_inv.append(expm(H0_array[i]*T_hop[i]))
    Bk = cp.array(Bk)
    Bk_inv = cp.array(Bk_inv)
    return Bk, Bk_inv
