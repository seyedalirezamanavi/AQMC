# this function is to set aqmc parameters and run the method
from initializing import make_hopping, init_trotter, expmk, cluster
from fromscratch import from_scratch
import numpy as np
import numpy as cp

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
    
        if self.N_s<32:
            self.BLOCKSIZE = self.N_s
        else:
            self.BLOCKSIZE = 32
    
    def initialize(self):
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


        map_streams = []
        mark_list = []
        for i in range(self.N_markov):
            mark_list.append(i)
            #map_streams.append(cp.cuda.stream.Stream())
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
        return hs_m
        
    def run(self):
        self.initialize()
        
        
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
}

a = AQMC(**params)
