import itertools
import numpy as np
import cvxpy as cp
import scipy
from sklearn.linear_model import LassoLars, lars_path

def sum_top_k(x, k):
    """ Sum of top k entries of vector x """
    idx = x.argsort()[-k:][::-1]
    return sum(x[idx]), idx

def kbits(n, k):
    """ Returns all sets of n choose k"""
    result = []
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        result.append(s)
    return result


def primalize(Q, p, supp):
    """ Get primal solution based on support """
    
    if not isinstance(supp, list):
        supp = [supp]  
    
    m = Q.shape[0]
    soln_k, costs_k = [], []
    
    for s in supp:
        s.sort()
        Qs = Q[s,:][:,s]
        ps = p[s]
        
        # optimal soln/cost for given support
        xstar = np.zeros(m)
        
        ## should replace with minimum norm solution
        try:
            soln = -np.linalg.inv(Qs) @ ps
        except:
            raise ValueError
            soln = -np.linalg.inv(Qs + 0.01 * np.eye(len(s))) @ ps
        xstar[s] = soln
        cost = 1/2 * ps.T @ soln
        
        soln_k.append(xstar)
        costs_k.append(cost)
    
    return costs_k, soln_k


def l0_bruteforce(Q, p, k):
    """ Brute Force solution to l0 constr QP """
    
    if not isinstance(k, list):
        k = [k]  
        
    supp_k, costs_k = [], []
    
    for kk in k:
        m = len(p)
        
        # will take too long
        if m > 16:
            return [], []
        
        supports = kbits(m,kk)
        costs = []
        counter = 0

        # enumerate over all sets
        for supp in supports:    
            counter += 1
            if counter % 100000 == 0:
                print(counter)
            idx = np.where(np.array(supp) == 1)[0]
            Qtilde = Q[idx,:][:,idx]
            ptilde = p[idx]
            M = np.linalg.inv(Qtilde)
            costs.append(-1/2 * ptilde.T @ M @ ptilde)

        # get minimum cost solution
        idx_min = np.argmin(costs)
        one_hot = np.array(supports[idx_min])
        supp_k.append(np.where(one_hot != 0)[0])
        costs_k.append(costs[idx_min])
    
    return supp_k, costs_k


def lbfgs_dual(D, L, p, k, verbose=False):
    
    if not isinstance(k, list):
        k = [k]  
        
    supp_k, costs_k = [], []
    
    for kk in k:
        m = len(p)
        l = L.shape[1]
        iD = np.linalg.inv(D)
        
        def fun(v):
            # caluclate sum-top-k vector
            t = np.zeros(m)
            for i in range(m):
                t[i] = 1/2 * iD[i,i] * np.linalg.norm(p[i] + L[i,:] @ v) ** 2
            g, idx = sum_top_k(t, kk)
            
            #calculate derivative sum-top-k function
            dg = 0
            for i in idx:
                dg += L[i,:] * iD[i,i] * (p[i] + L[i,:] @ v)
                
            # cost and derivative
            f = 1/2 * np.linalg.norm(nu) ** 2 + g
            df = nu + dg
            return f, df


def cvx_dual(D, L, p, k, verbose=False, solver='ecos'):
    """ Dual solved using cvx """
    
    if not isinstance(k, list):
        k = [k]  
        
    supp_k, costs_k = [], []
    
    for kk in k:
        if verbose:
            print(f'kk = {kk}')
        m = len(p)
        l = L.shape[1]
        iD = np.linalg.inv(D)

        # cvx problem
        nu = cp.Variable(l)
        
        alpha = -1/2 * cp.multiply(1/np.diag(D), (p + L @ nu) ** 2)
        cost = 0.5 * cp.sum_squares(nu) + cp.sum_largest(-alpha, kk)

        prob = cp.Problem(cp.Minimize(cost))
        if solver == 'scs':
            prob.solve(solver=cp.SCS)
        elif solver == 'ecos':
            prob.solve(solver=cp.ECOS, verbose=verbose, abstol=1e-2, reltol=1e-2, feastol=1e-6)

        # recover support from t.value
        support = alpha.value.argsort()[-kk:][::-1]
        support.sort()
        
        supp_k.append(support)
        costs_k.append(-cost.value)

    return supp_k, costs_k

    

def lasso_solve(X, y, k):
    """ Solve sparse QP with l1 penalization -- same as lars solve """

    L1 = LassoLars(alpha=1e-6,fit_intercept=False, normalize=False)
    L1.fit(X,y)
    supp_k, l1_sparsity = [], []
    for c in L1.coef_path_.T:
        supp = np.where(c != 0)[0]
        supp_k.append(supp)
        l1_sparsity.append(len(supp))
    indexes = [l1_sparsity.index(x) for x in set(l1_sparsity)]
    l1_sparsity = np.array(l1_sparsity)[np.array(indexes)[1:]]
    supp_k = [j for i, j in enumerate(supp_k) if i in indexes][1:]
    
        
    return supp_k[:k]
    
    
########### not in use ##############################
def cvx_bidual(D, L, p, k, verbose=False,
           solver=cp.SCS):
    """ Bidual SDP solved using cvx """
    
    m = len(p)
    l = L.shape[1]
    Dhalf = np.linalg.inv(np.sqrt(D))
    
    ## cvx problem
    t = cp.Variable(1)
    x = cp.Variable(m)
    X = cp.Variable((l+1,l+1), PSD=True)

    Dx = cp.diag(x)
    Sigma = Dhalf @ Dx @ Dhalf

    cost = 1/2 * t
    constraints = []
    constraints = [X[0,0] == t + p.T @ Sigma @ p,
                   X[0,1:l+1] == p.T @ Sigma @ L, 
                   X[1:l+1, 0] == L.T @ Sigma @ p,
                   X[1:l+1,1:l+1] == np.eye(l) + L.T @ Sigma @ L,
                   X >> 0,
                   cp.sum(x) <= k, x<= 1, x >= 0]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver, verbose=verbose)
    
    return x.value, 1/2 * t.value[0]


def lars_solve(X, y, k):
    """ Solve LASSO problem and get path """
    
    _, _, coefs = lars_path(X, y, method='lasso', verbose=True)
    supp_k, l1_sparsity = [], []
    for c in coefs.T:
        supp = np.where(c != 0)[0]
        supp_k.append(supp)
        l1_sparsity.append(len(supp))
    indexes = [l1_sparsity.index(x) for x in set(l1_sparsity)]
    l1_sparsity = np.array(l1_sparsity)[np.array(indexes)[1:]]
    supp_k = [j for i, j in enumerate(supp_k) if i in indexes][1:]
    
    return supp_k[:k]

