# measures go here
import cupy as cp

def correlation(GF_up_tmp, GF_dn_tmp, sz_tmp, rho_tmp, n_up_tmp, n_dn_tmp, N_x, N_y):
    corr_z_list = 0
    corr_x_list = 0
    rho_list = 0
    N_s = N_x*N_y
    for nx_1 in range(N_x):
        for ny_1 in range(N_y):
            
            R1_0 = nx_1*N_y + ny_1 # first coordinate of the correlation function
            R2_0 = N_y*cp.concatenate([cp.arange(nx_1+1,N_x) ,cp.arange(0,nx_1+1)]) - N_y + ny_1  # 2nd coordinate of the correlation function
            
            
            # <sz(R1_0)sz(R2_0)> computation
            szsz = sz_tmp[R1_0]*sz_tmp[R2_0] - cp.multiply(GF_up_tmp[R1_0,R2_0],GF_up_tmp[R2_0,R1_0])/4 - cp.multiply(GF_dn_tmp[R1_0,R2_0],GF_dn_tmp[R2_0,R1_0])/4
            szsz[0] = -n_up_tmp[R1_0]*n_dn_tmp[R1_0]/2 + rho_tmp[R1_0]/2
            
            # <sx(R1_0)sx(R2_0)> computation
            sxsx = -cp.multiply(GF_up_tmp[R1_0,R2_0],GF_dn_tmp[R2_0,R1_0])/2
            sxsx[1] = n_up_tmp[R1_0]*(1-n_dn_tmp[R1_0])/2
            
            # <rho(R1_0)\rho(R2_0)> computation
            rho =  (rho_tmp[R1_0]*rho_tmp[R2_0]) - cp.multiply(GF_up_tmp[R1_0,R2_0],GF_up_tmp[R2_0,R1_0])/4 - cp.multiply(GF_dn_tmp[R1_0,R2_0],GF_dn_tmp[R2_0,R1_0])/4
            rho[1] = n_up_tmp[R1_0]*n_dn_tmp[R1_0]/2 + rho_tmp[R1_0]/2

            corr_z_list = cp.add(corr_z_list,szsz)
            corr_x_list = cp.add(corr_x_list,sxsx)
            rho_list = cp.add(rho_list,rho)
    szsz_correlation = corr_z_list/N_s
    sxsx_correlation = corr_x_list/N_s
    rho_correlation = rho_list/N_s

    return szsz_correlation,sxsx_correlation,rho_correlation

#def expectaion_values(G_up, G_dn, sign_partition_accu, N_s, X_dimension, Y_dimension):
    #G_up_m = cp.eye(N_s) - G_up
    #G_dn_m = cp.eye(N_s) - G_dn
    #G_up_mar += (G_up_m)*sign_partition_accu
    #G_dn_mar += (G_dn_m)*sign_partition_accu
    #n_up_tmp = cp.diag(G_up_m)
    #n_dn_tmp = cp.diag(G_dn_m)
    #sz_tmp = (n_up_tmp-n_dn_tmp)/2
    #rho_tmp = (n_up_tmp+n_dn_tmp)/2
    #interaction_mar += cp.sum(n_up_tmp * n_dn_tmp * U)*sign_partition_accu
    #szsz,sxsx,rho = correlation(G_up_m, G_dn_m, sz_tmp, rho_tmp, n_up_tmp, n_dn_tmp, X_dimension, Y_dimension)
    #szsz_mar += szsz*sign_partition_accu
    #sxsx_mar += sxsx*sign_partition_accu
    #rho_mar += rho*sign_partition_accu
    #sign_mar += sign_partition_accu
    #N_measure += 1
    
    #return G_up_mar, G_dn_mar, interaction_mar, szsz_mar, sxsx_mar, rho_mar, sign_mar, N_measure
