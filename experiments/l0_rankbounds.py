import sys
sys.path.append('../SparseOpt/')
import sparse_models as sm
from helper import generate_data, solve_l0
import pickle
import numpy as np

# init
np.random.seed(1)
loss = 'l2'
lmbda_list = list(np.logspace(-2, 1, 40))
method_list = ['constrained', 'penalized']
n_list = [10, 100]
m_list = [10, 20]
gamma_list = [1e-1, 1, 10]
r_list = [0.1, 0.2, 0.5, 1]
true_sparsity_list = [0.2, 0.8]
low_rank = 'soft'
l2constr = True


assert l2constr == True
assert low_rank == 'soft'

for method in method_list:
    for n in n_list:
        for m in m_list:
            for gamma in gamma_list:
                for r in r_list:
                    for true_sparsity in true_sparsity_list:
                        r_eff = np.min((n, round(r * m)))
                        params = generate_data(n,
                                               m,
                                               r_eff,
                                               loss=loss,
                                               true_sparsity=true_sparsity,
                                               gamma=gamma,
                                               low_rank=low_rank)
                        supp, pcosts, bdcosts = [], [], []
                        conscosts, optcosts = [], []

                        if method == 'constrained':
                            lmbdas = []
                        else:
                            lmbdas = lmbda_list
                        p_costs, bd_costs, opt_costs = solve_l0(params,
                                                                method=method,
                                                                lmbdas=lmbdas,
                                                                l2constr=l2constr)
                        # store dual  variables so that we can construct
                        # upper and lower bounds on rank approx  via weak duality
                        U, S, Vh = np.linalg.svd(params['Xtrue'], full_matrices=True)
                        params2 = params.copy()
                        nu = []
                        
                        n = len(params['y'])
                        for r in np.arange(1,m+1):
                            Xr = U[:,:r] @ np.diag(S[:r]) @ Vh[:r,:]
                            params2['X'] = Xr
                            dual_cost, dual_var = sm.dual_problem(
                                params2, l2constr=l2constr)
                            nu.append(dual_var[:n]) #only first n entries are relevant

                        res = {
                            'primal': p_costs,
                            'bidual': bd_costs,
                            'primalized': opt_costs,
                            'method': method,
                            'params': params,
                            'lmbdas': lmbdas,
                            'nus' : nu
                        }

                        filename = f'../results/synthetic/l2con/rankbounds/{n}_{m}_{r_eff}_{gamma}_{true_sparsity}_{method}_lowrank_{low_rank}.pkl'

                        pickle.dump(res, open(filename, "wb"))
