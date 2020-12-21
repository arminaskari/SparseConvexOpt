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
                                               gamma=gamma)
                        supp, pcosts, bdcosts = [], [], []
                        conscosts, optcosts = [], []
                        
                        if method == 'constrained':
                            lmbdas = []
                        else:
                            lmbdas = lmbda_list
                        p_costs, bd_costs, opt_costs = solve_l0(params,
                                                                method=method,
                                                                lmbdas=lmbdas)

                        res = {
                            'primal': p_costs,
                            'bidual': bd_costs,
                            'primalized': opt_costs,
                            'method': method,
                            'params': params,
                            'lmbdas': lmbdas
                        }

                        filename = f'../results/synthetic/{n}_{m}_{r_eff}_{gamma}_{true_sparsity}_{method}.pkl'

                        pickle.dump(res, open(filename, "wb"))
