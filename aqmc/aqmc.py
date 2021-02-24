# this function is to set aqmc parameters and run the method
class AQMC:
    __init__(self, **params):
        self.X_dimension = params["X_dimension"]
        self.Y_dimension = params["Y_dimension"]
        self.N_s = X_dimension * Y_dimension 

        self.periodic_X = params["periodic_X"]
        self.periodic_Y = params["periodic_Y"]
        self.tunneling = params["tunneling"]
        self.U = params["U"]
        self.chemical_potential = params["chemical_potential"] # half filling 0
        self.N_markov = params["N_markov"]
        self.N_sw_measure = params["N_sw_measure"]
        self.N_warm_up = N_sw_measure // 5
        self.N_from_scratch = params["N_from_scratch"]
        self.N_qr = params["N_qr"]  
    
        # Set generalized trotter
        self.Beta = params["Beta"]
        self.N_time = params["N_time"]
        self.U_eff =  U
    
        if self.N_s<32:
            self.BLOCKSIZE = self.N_s
        else:
            self.BLOCKSIZE = 32
