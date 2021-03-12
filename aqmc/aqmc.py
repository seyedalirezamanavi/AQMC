# this function is to set aqmc parameters and run the method
from initializing import make_hopping, init_trotter, expmk, cluster
from fromscratch import from_scratch
from update import time_wrap, update_G
from measure import correlation

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

class AQMC:
    def __init__(self, **params):
        self.X_dimension = params["X_dimension"]
        self.Y_dimension = params["Y_dimension"]
        self.N_s = self.X_dimension * self.Y_dimension 

        self.periodic_X = params["periodic_X"]
        self.periodic_Y = params["periodic_Y"]
        self.tunneling = params["tunneling"]
        self.U = params["U"]
        self.chemical_potential = params["chemical_potential"] # half filling 0
        self.N_markov = params["N_markov"]
        self.N_sw_measure = params["N_sw_measure"]
        self.N_warm_up = self.N_sw_measure // 5
        self.N_from_scratch = params["N_from_scratch"]
        self.N_qr = params["N_qr"]  
    
        # Set generalized trotter
        self.Beta = params["Beta"]
        self.N_time = params["N_time"]
        self.U_eff =  self.U
        self.directory = params["directory"]
        if self.N_s<32:
            self.BLOCKSIZE = self.N_s
        else:
            self.BLOCKSIZE = 32
        
    def pin_memory(self):
        memory_pool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(memory_pool.malloc)
        pinned_memory_pool = cp.cuda.PinnedMemoryPool()
        cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)
        
    def run(self):
        # making tunneling Hamiltonian matrix
        H_0 = make_hopping(self.X_dimension, self.Y_dimension, self.periodic_X, self.periodic_Y, self.tunneling)
        H_0_msr = H_0.copy()
        H_0 += np.identity(self.X_dimension * self.Y_dimension) * ((-1) * self.chemical_potential)
    
        I = cp.array(1)
        sign_U_interact, T_hop, H0_array, lamda, probability, gamma1, gamma2 = init_trotter(self.Beta,self.N_time,self.U_eff,H_0)
        Bk, Bk_inv = expmk(H0_array, T_hop)
        

        sign_U_interact_m = cp.empty((self.N_markov,self.N_time))
        hs_m = cp.empty((self.N_markov,self.N_time,self.N_s))
        cl_up_m = cp.empty((self.N_markov,self.N_time,self.N_s,self.N_s))
        cl_dn_m = cp.empty((self.N_markov,self.N_time,self.N_s,self.N_s))
        G_up_m = cp.empty((self.N_markov,self.N_s,self.N_s))
        G_dn_m = cp.empty((self.N_markov,self.N_s,self.N_s))
        Bk_m = cp.empty((self.N_markov,self.N_time,self.N_s,self.N_s))
        Bk_inv_m = cp.empty((self.N_markov,self.N_time,self.N_s,self.N_s))
        probability_m = cp.empty((self.N_markov,self.N_time))
        gamma1_m = cp.empty((self.N_markov,self.N_time))
        gamma2_m = cp.empty((self.N_markov,self.N_time))

        hs_cached = self.N_markov * 1000 
        
        map_streams = []
        mark_list = []
        for i in range(self.N_markov):
            mark_list.append(i)
            map_streams.append(cp.cuda.stream.Stream())
            sign_U_interact_m[i] = sign_U_interact
            hs_m[i] = lamda[:,None]*((-1)**cp.random.randint(2,size=(self.N_time,self.N_s)))
            cl_up_m[i],cl_dn_m[i] = cluster(hs_m[i],Bk_m[i],sign_U_interact_m[i])
            G_up_m[i] = from_scratch(cl_up_m[i],self.N_qr)
            G_dn_m[i] = from_scratch(cl_dn_m[i],self.N_qr)
            Bk_m[i] = Bk.copy()
            Bk_inv_m[i] = Bk_inv.copy()
            probability_m[i] = probability.copy()
            gamma1_m[i] = gamma1.copy()
            gamma2_m[i] = gamma2.copy()

        check = cp.zeros((self.N_sw_measure+self.N_warm_up,self.N_time,self.N_markov))
        n_up_dn = cp.zeros(self.N_markov)
        interaction_mar = cp.zeros(self.N_markov)
        sign_mar = cp.zeros(self.N_markov)
        N_measure = cp.zeros(self.N_markov)
        G_up_mar = cp.zeros((self.N_markov,self.N_s,self.N_s))
        G_dn_mar = cp.zeros((self.N_markov,self.N_s,self.N_s))
        szsz_mar = cp.zeros((self.N_markov,self.X_dimension))
        sxsx_mar = cp.zeros((self.N_markov,self.X_dimension))
        rho_mar = cp.zeros((self.N_markov,self.X_dimension))
        sign_partition_accu = cp.array([cp.linalg.slogdet(G_up_m[i]+G_dn_m[i])[0] for i in range(self.N_markov)],dtype=cp.int32)[:,None]
        HS_list = cp.empty((1, self.N_time, self.N_s)) #keep in mind the size of the array be smaller than the DRAM
        P_list = cp.empty((1, )) #keep in mind the size of the array be smaller than the DRAM
        sign_list = cp.empty((1, )) #keep in mind the size of the array be smaller than the DRAM
        # mark_list, map_streams = self.initialize()
        strt = time.time()
        for msr in range(self.N_sw_measure+self.N_warm_up):
            for l in range(self.N_time):
                for i,stream in zip(mark_list,map_streams):
                    with stream:                
                        randomlist = cp.random.random(size=(self.N_s))
                        
                        update_G((1,),(1,),(cp.eye(self.N_s),cp.eye(self.N_s),G_up_m[i],  G_dn_m[i], hs_m[i], randomlist, self.N_s, self.N_time, sign_partition_accu[i], probability_m[i], gamma1_m[i], gamma2_m[i],self.BLOCKSIZE))
                        G_up_m[i], G_dn_m[i], Bk_m[i], Bk_inv_m[i], hs_m[i], cl_up_m[i], cl_dn_m[i], sign_U_interact_m[i],probability_m[i], gamma1_m[i], gamma2_m[i] = time_wrap(G_up_m[i], G_dn_m[i], Bk_m[i], Bk_inv_m[i], hs_m[i], cl_up_m[i], cl_dn_m[i], sign_U_interact_m[i],probability_m[i], gamma1_m[i], gamma2_m[i])
                        if l%self.N_from_scratch == 0:
                            G_up_m[i] = from_scratch(cl_up_m[i],self.N_qr)
                            G_dn_m[i] = from_scratch(cl_dn_m[i],self.N_qr)
                            
                        if msr >= self.N_warm_up and l==self.N_time-1:
                            G_up_m[i] = cp.eye(self.N_s) - G_up_m[i]
                            G_dn_m[i] = cp.eye(self.N_s) - G_dn_m[i]
                            G_up_mar[i] += (G_up_m[i])*sign_partition_accu[i,0]
                            G_dn_mar[i] += (G_dn_m[i])*sign_partition_accu[i,0]
                            n_up_tmp = cp.diag(G_up_m[i])
                            n_dn_tmp = cp.diag(G_dn_m[i])
                            sz_tmp = (n_up_tmp-n_dn_tmp)/2
                            rho_tmp = (n_up_tmp+n_dn_tmp)/2
                            interaction_mar[i] +=  cp.sum(n_up_tmp * n_dn_tmp * self.U)*sign_partition_accu[i,0]
                            szsz,sxsx,rho = correlation(G_up_m[i], G_dn_m[i], sz_tmp, rho_tmp, n_up_tmp, n_dn_tmp, self.X_dimension, self.Y_dimension)
                            szsz_mar[i] += szsz*sign_partition_accu[i,0]
                            sxsx_mar[i] += sxsx*sign_partition_accu[i,0]
                            rho_mar[i] += rho*sign_partition_accu[i,0]
                            sign_mar[i] += sign_partition_accu[i,0]
                            N_measure[i] += 1
            HS_list = cp.concatenate(HS_list, hs_m)
            si, logP = cp.linalg.slogdet(G_up_m+G_dn_m)
            P_list = cp.concatenate(P_list, cp.exp(logP))
            sign_list =cp.concatenate(sign_list, si)
            
            if msr >= self.N_warm_up or (msr+1) % hs_cached == 0:
                HS_list, P_list, sign_list = self.save_hs(HS_list,  P_list, sign_list, self.directory)
                                
        HS_list, P_list, sign_list = self.save_hs(HS_list,  P_list, sign_list, self.directory)                        
        end = time.time()
        N_msr = cp.mean(N_measure,axis=0)
        sign_msr = sign_mar/N_msr


        G_up_msr = cp.array([G_up_mar[i]/(N_msr*sign_msr[i]) for i in range(self.N_markov)])
        G_dn_msr = cp.array([G_dn_mar[i]/(N_msr*sign_msr[i]) for i in range(self.N_markov)])
        interaction_msr = cp.array([interaction_mar[i]/(N_msr*sign_msr[i]) for i in range(self.N_markov)])

        G_msr = G_up_msr+G_dn_msr
            
        kin = cp.array([cp.trace(cp.array(H_0_msr).dot(cp.eye(self.N_s)-G_msr[i]))/self.N_s for i in range(self.N_markov)])
        intr = cp.array([interaction_msr[i]/self.N_s for i in range(self.N_markov)])
        filling = np.trace(2*cp.eye(self.N_s) - cp.mean(G_msr,axis=0))/self.N_s
        mean_onsite_corr = cp.mean(cp.array([interaction_mar[i]/(self.U*cp.mean(1-cp.diag(G_up_mar[i])*cp.mean(1-cp.diag(G_dn_mar[i])))) for i in range(self.N_markov)]),axis=0)
        totenrgy = kin + intr
        err = 100*cp.std(totenrgy,axis=0)/cp.abs(cp.mean(totenrgy,axis=0))
        energy_mean = cp.mean(totenrgy,axis=0)

        


        
        szsz_msr = cp.array([szsz_mar[i,:]/(N_msr*sign_msr[i]) for i in range(self.N_markov)])
        sxsx_msr = cp.array([sxsx_mar[i,:]/(N_msr*sign_msr[i]) for i in range(self.N_markov)])
        rho_msr = cp.array([rho_mar[i,:]/(N_msr*sign_msr[i]) for i in range(self.N_markov)])
        
        sign_msr = cp.mean(cp.abs(sign_msr),axis=0)

        measures = {"elpsd_time":end-strt,
                    "kinetic_energy_mean":cp.mean(kin),
                    "interaction_energy_mean":cp.mean(intr),
                    "n_mean":filling,
                    "mean_onsite_corr":mean_onsite_corr,
                    "energy_mean":energy_mean,
                    "err_bar":err,
                    "sign_mean":sign_msr,
                    "markov_data":{
                        "G_up_mar":G_up_mar,
                        "G_dn_mar":G_dn_mar,
                        "N_msr":N_msr,
                        "sign_mar":sign_mar,
                        "interaction_mar":interaction_mar,
                        "correlations":{
                            "szsz_mar":szsz_mar,
                            "sxsx_mar":sxsx_mar,
                            "rho_mar":rho_mar
                        }
                                },
                    "model_params":{
                        "N_sw_measure" : self.N_sw_measure,
                        "N_warm_up" : self.N_sw_measure // 5,
                        "N_from_scratch" : self.N_from_scratch,
                        "N_qr" : self.N_qr,
                        "N_markov" : self.N_markov,
                        "chemical_potential" : self.chemical_potential,
                        "U" : self.U,
                        "tunneling" : self.tunneling,
                        "periodic_Y" : self.periodic_Y,
                        "periodic_X" : self.periodic_X,
                        "X_dimension" : self.X_dimension,
                        "Y_dimension" : self.Y_dimension,
                        "N_s" : self.X_dimension * self.Y_dimension,
                        "N_time" : self.N_time,
                        "Beta" : self.Beta,    
                        "directory": self.directory,
                    }
                    }

        print(" time: {}s\n kinetic_energy_mean: {}\n interaction_energy_mean: {}\n n_mean: {}\n mean_onsite_corr: {}\n energy_mean: {}\n err_bar: {}\n sign_mean: {}".format(end - strt,cp.mean(kin),cp.mean(intr),filling,mean_onsite_corr,energy_mean,err,sign_msr))
        plt.plot(cp.asnumpy(szsz_msr[0]))
        return measures
    
    def save_hs(self, hs, p, sign, directory):
        cp.savez_compressed(directory+str(time.time())+".npz", hs = hs, sign = sign, p = p)
        return cp.empty_like(hs), cp.empty_like(p), cp.empty_like(sign) #keep in mind the size of the array be smaller than the DRAM
    
    def load_hs(directory):
    
        data = cp.load(directory)
        hs = data["hs"]
        sign = data["sign"]
        p = data["p"]
        return hs, sign, p

    def calc_var_green(directory, params):
        
        H_0 = make_hopping(self.X_dimension, self.Y_dimension, self.periodic_X, self.periodic_Y, self.tunneling)
        H_0 += np.identity(self.X_dimension * self.Y_dimension) * ((-1) * self.chemical_potential)

        sign_U_interact, T_hop, H0_array, _, _, _, _ = init_trotter(self.Beta,self.N_time,self.U_eff,H_0)
        Bk, Bk_inv = expmk(H0_array, T_hop)
        
        hs_m, sign_m, p_m = load_hs(directory)
        
        G_up_m = cp.zeros((self.N_s, self.N_s))
        G_dn_m = cp.zeros((self.N_s, self.N_s))
                
        for hs, sign, p in zip(hs_m, sign_m, p_m):
        
            cl_up, cl_dn = cluster(hs,Bk,sign_U_interact)
            
            G_up = from_scratch(cl_up, self.N_qr)
            G_dn = from_scratch(cl_dn, self.N_qr)
            
            signp, log_pp = cp.linalg.slogdet(G_up + G_dn)
            
            G_up = cp.eye(self.N_s) - G_up
            G_dn = cp.eye(self.N_s) - G_dn
            
            G_up_m += G_up * (signp * pp)/(sign * p)
            G_dn_m += G_dn * (signp * pp)/(sign * p)
            
            n_up_tmp = cp.diag(G_up_m)
            n_dn_tmp = cp.diag(G_dn_m)
            sz_tmp = (n_up_tmp-n_dn_tmp)/2
            rho_tmp = (n_up_tmp+n_dn_tmp)/2
        
params = {
"N_sw_measure" : 10,
"N_warm_up" : 10 // 5,
"N_from_scratch" : 10,
"N_qr" : 10,
"N_markov" : 2,
"chemical_potential" : 0,
"U" : 4,
"tunneling" : 1,
"periodic_Y" : 1,
"periodic_X" : 1,
"X_dimension" : 16,
"Y_dimension" : 8,
"N_s" : 16*8,
"N_time" : 100,
"Beta" : 4,         
"directory":""
}


a = AQMC(**params)
