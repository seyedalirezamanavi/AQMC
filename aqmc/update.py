# in this module, we emplement the sherman morrison local 
# configuration update and time wrapping. 




src = r'''
 extern "C"{
__global__ void outer(double* G_upp,double* G_dnp,double* G_up,double* G_dn,int p, int N_s, double gamma1, double r_up,double gamma2, double r_dn){
    int l = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    double I11, I21, I12, I22;
    
    int id1 = j*N_s + p;
    int id2 = p*N_s + l;
    int id = j*N_s + l;
    
    I11 =  G_up[id1];
    I12 =  G_dn[id1];

    I21 = G_up[id2];
    I22 = G_dn[id2];

    I11 = - (j==p) + I11;
    I12 = - (j==p) + I12;

    I21 = gamma1/r_up * I21;
    I22 = gamma2/r_dn * I22;

    G_upp[id] = G_up[id] + I11 * I21;
    G_dnp[id] = G_dn[id] + I12 * I22;

}

__global__ void copy(double* G_upp, double* G_dnp, double* G_up, double* G_dn, int N_s){
    int l = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int id = j*N_s + l;
    G_up[id] = G_upp[id];
    G_dn[id] = G_dnp[id];

}


__global__ void update_G(double* G_upp, double* G_dnp,double* G_up, double* G_dn, double* hs,
                         double* randomlist, int N_s, int N_t,
                         int* sign_partition_accu, double* probability_tmp,
                         double* gamma1_tmp, double* gamma2_tmp, int BLOCKSIZE){
              
        double gamma1,gamma2,P;
        int ii ;
        double r, r_up, r_dn;
        int NUMGRID = N_s/BLOCKSIZE;
        dim3 block(BLOCKSIZE,BLOCKSIZE);
        dim3 grid(NUMGRID,NUMGRID);
        //#pragma unroll
        for(int i=0;i<N_s;i++)
        {
            if (hs[N_s*(N_t-1)+i] > 0){
                gamma1 = gamma1_tmp[N_t-1];
                gamma2 = gamma2_tmp[N_t-1];
                P = probability_tmp[N_t-1];
            }
            else{
                gamma1 = gamma2_tmp[N_t-1];
                gamma2 = gamma1_tmp[N_t-1];
                P = 1/probability_tmp[N_t-1];
            }   

            
            ii = i * N_s + i;
            r_up = 1 + gamma1*(1-G_up[ii]);
            r_dn = 1 + gamma2*(1-G_dn[ii]);

            r = r_up * r_dn * P;

            if (abs(r)>randomlist[i]){
                outer<<<grid,block>>>(G_upp,G_dnp,G_up,G_dn, i, N_s, gamma1, r_up, gamma2, r_dn);
                copy<<<grid,block>>>(G_upp,G_dnp,G_up,G_dn,N_s);
                cudaDeviceSynchronize();
                sign_partition_accu[0] = r/abs(r) * sign_partition_accu[0];
                hs[N_s*(N_t-1)+i] = -hs[N_s*(N_t-1)+i];
            }
            
        }
}
}
 '''
 
update = cp.RawModule(code=src,backend='nvcc',options=('-dc',))
update_G = update.get_function('update_G')
 
 
def update(G_up, G_dn, hs,sign_partition_accu,randomlist, N_s, probability_tmp, gamma1, gamma2,I):
 
    # sign_partition_accu = cp.array([1]).astype(cp.int32)
    for i in range(N_s):
        # acceptance ratio
        if hs[-1,i]>0:
            gamma1_tmp = gamma1
            gamma2_tmp = gamma2
            P = probability_tmp
        else:
            gamma1_tmp = gamma2
            gamma2_tmp = gamma1
            P = 1/probability_tmp
 
        r_1 = I + gamma1_tmp*(I-G_up[i,i])
        r_2 = I + gamma2_tmp*(I-G_dn[i,i])
        r = r_1 * r_2 * P
        if cp.abs(r) > randomlist[i]:
        # update greens 
            k_1 = -G_up[:,i][:,None]
            k_1[i] += I
            G_up = G_up - (gamma1_tmp / r_1) * k_1*G_up[i,:]
            k_2 = -G_dn[:,i][:,None]
            k_2[i] += I
            G_dn = G_dn - (gamma2_tmp / r_2) * k_2*G_dn[i,:]
            sign_partition_accu = cp.sign(r) *sign_partition_accu
            hs[-1,i] = -hs[-1,i]
 
    return G_up, G_dn, hs, sign_partition_accu


 
src = r'''
 extern "C"{
 
 __global__ void calc_B (double* B_inv,double* B, double* hu, double* Bk_inv, double* Bk, const int N_s,int signU){
    int l = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    #pragma unroll
    for(int li = l; li<N_s; li += gridDim.x * blockDim.x){
            for(int ji = j; ji<2*N_s; ji += gridDim.y * blockDim.y){
                B_inv[li*2*N_s + ji] = exp(-hu[li]) * Bk_inv[li*2*N_s + ji];
                B_inv[(li+N_s)*2*N_s + ji] = exp(signU*hu[li]) * Bk_inv[(li+N_s)*2*N_s + ji];
            }
    }
    #pragma unroll
    for(int li = l; li<2*N_s; li += gridDim.x * blockDim.x){
            for(int ji = j; ji<N_s; ji += gridDim.y * blockDim.y){    
                B[li*2*N_s + ji] = exp(hu[ji]) * Bk[li*2*N_s + ji];
                B[li*2*N_s + ji + N_s] = exp(-signU*hu[ji]) * Bk[li*2*N_s + ji + N_s];
            }
    }
     
 }
////////////////////////////////////////////////////////////////////////////////
 
const int N = 1 << 10;
const int SHMEM_SIZE = 1 << 10;
 
__global__ void matrixMul(const float *a, const float *b, float *c) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
 
  // Statically allocated shared memory
  __shared__ int s_a[SHMEM_SIZE];
  __shared__ int s_b[SHMEM_SIZE];
 
  // Accumulate in temporary variable
  int tmp = 0;
 
  // Sweep tile across matrix
  #pragma unroll
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = a[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] =
        b[i * N + threadIdx.y * N + col];
 
    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();
 
    // Do matrix multiplication on the small matrix
    for (int j = 0; j < blockDim.x; j++) {
      tmp +=
          s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }
 
    // Wait for all threads to finish using current tiles before loading in new
    // ones
    __syncthreads();
  }
 
  // Write back results
  c[row * N + col] = tmp;
}
 
 
////////////////////////////////////////////////////////////////////////////////
 
 
 
 
/*
 __global__ void time_wrap(double* B_inv,double* B, double* hu, double* Bk, const int N_s, int* signU) 
 {  
    calc_B(B_inv, B, hu, Bk, N_s, signU);
 }*/
 }
 '''
 
 
 
time_wrap_module = cp.RawModule(code=src)
calc_B = time_wrap_module.get_function('calc_B')
# matmul_kernel = time_wrap_module.get_function('matrixMul')
 
 
def time_wrap_partial_kernel(B,B_inv, G, Bk, Bk_inv, h, cl, sign_U_interact,probability, gamma1, gamma2,N_s):
    calc_B((8,8),(16,16),(B_inv,B,h[-1],Bk_inv[-1],Bk[-1],N_s,int(sign_U_interact[-1])))
    G = B.dot(G).dot(B_inv)
    cl[-1]= B
 
    h = cp.roll(h,1,axis = 0)
    Bk = cp.roll(Bk,1,axis = 0)
    Bk_inv = cp.roll(Bk_inv,1,axis = 0)
    cl = cp.roll(cl,1,axis = 0)
    probability = cp.roll(probability,1,axis = 0)
    gamma1 = cp.roll(gamma1,1,axis = 0)
    gamma2 = cp.roll(gamma2,1,axis = 0)
    sign_U_interact = cp.roll(sign_U_interact,1,axis = 0)
    return G, Bk, Bk_inv, h, cl, sign_U_interact, probability, gamma1, gamma2
 
 
 
def time_wrap(G_up, G_dn, Bk, Bk_inv, h, cl_up, cl_dn, sign_U_interact,probability, gamma1, gamma2):
    B_up_inv = cp.diag(cp.exp(-h[-1,:])).dot(Bk_inv[-1])
    B_up = Bk[-1].dot(cp.diag(cp.exp(h[-1,:])))
    G_up = B_up.dot(G_up).dot(B_up_inv)
    cl_up[-1]= B_up

    B_dn_inv = cp.diag(cp.exp(sign_U_interact[-1]*h[-1,:])).dot(Bk_inv[-1])
    B_dn = Bk[-1].dot(cp.diag(cp.exp(-sign_U_interact[-1]*h[-1,:])))
    G_dn = B_dn.dot(G_dn).dot(B_dn_inv)
    cl_dn[-1]= B_dn
 
    h = cp.roll(h,1,axis = 0)
    Bk = cp.roll(Bk,1,axis = 0)
    Bk_inv = cp.roll(Bk_inv,1,axis = 0)
    cl_up = cp.roll(cl_up,1,axis = 0)
    cl_dn = cp.roll(cl_dn,1,axis = 0)
    probability = cp.roll(probability,1,axis = 0)
    gamma1 = cp.roll(gamma1,1,axis = 0)
    gamma2 = cp.roll(gamma2,1,axis = 0)
    sign_U_interact = cp.roll(sign_U_interact,1,axis = 0)
    return G_up, G_dn, Bk, Bk_inv, h, cl_up, cl_dn, sign_U_interact,probability, gamma1, gamma2
