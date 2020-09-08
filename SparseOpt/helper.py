import numpy as np
import scipy

def factor_alternating_min(Sigma, rank, tol=1e-5):
    """ Performing alternating min to get D + LL.T """
    
    assert Sigma.shape[0] == Sigma.shape[1]
    n = Sigma.shape[0]
    d = np.zeros(n)
    
    costs = [1, 2]
    while np.abs(costs[-1] - costs[-2])/np.abs(costs[-1]) > tol:
        u, s, v = np.linalg.svd(Sigma - np.diag(d))
        U = u[:, :rank] * np.sqrt(s[:rank])
        d = np.clip(np.diag(Sigma - U @ U.T),
                           a_min = 1e-3,
                           a_max = None)
        costs.append(np.linalg.norm(Sigma - np.diag(d) - U @ U.T))
    
    return np.diag(d), U, costs
    

def factor_decomp(Sigma, rank=1):
    """ Get factor decomposition of covariance via alternating min """
    
    if rank != -1:
        return factor_alternating_min(Sigma, rank)
    else:
        rank_costs = []
        for rank in range(1,Sigma.shape[0]-1):
            D, L, costs = factor_alternating_min(Sigma, rank)
            rank_costs.append(costs[-1])
        
        return rank_costs
            

def generate_factor_model(m, r):
    """ Generates random factor model structure """
    
    assert r >= 0   
    D = np.diag(np.random.rand(m)) # must be PD
    L = np.zeros((m,m))
    if r > 0:
        L = np.random.randn(m,r)
    p = np.random.randn(m)
    
    return D, L, p

                
def factor_model_to_ls(D, L, sparsity_lvl = 0.2):
    """ Convert factor model structure to LS matrices """
    
    m = D.shape[0]
    X = scipy.linalg.sqrtm(D + L @ L.T)
    
    # generate ground truth model
    beta = 10 * np.random.randn(m)
    indices = np.random.choice(np.arange(m), 
                               replace=False,
                               size=int(m * sparsity_lvl))
    beta[indices] = 0
    
    y = X @ beta + np.random.randn(m)
    
    return X, y
    
    