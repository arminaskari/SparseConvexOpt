import sys
sys.path.append('../SparseOpt/')
import sparse_models as sm
from helper import * 
import matplotlib.pyplot as plt

np.random.seed(1)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.figure(figsize=(6,6))
legend = [r'$\phi(k)$: dual lower bound',
          r'$\psi(k): \ell_0$ brute force',
          r'$\tilde{\psi}(k)$: dual primalized',
          r'$\ell_1$-reg: primalized']

# different dimensionality and rank levels to test
m_list = [10, 15]
s_list = [0, 1e-1, 2e-1, 5e-1, 1]

for m in m_list:
    for s in s_list:
        
        r = int(m * s)
        
        # generate synthetic data
        D, L, _ = generate_factor_model(m, r)
        X, y = factor_model_to_ls(D, L, sparsity_lvl = 0.2)
        Q = D + L @ L.T
        
        # sparse model solutions
        supp_l0, cost_l0 = sm.l0_bruteforce(Q, X.T @ y,
                                        list(range(1,m+1)))
        supp_du, cost_du = sm.cvx_dual(D, L, X.T @ y, 
                                       list(range(1,m+1)))
        supp_l1  = sm.lasso_solve(X, y, list(range(1,m+1)))
        
        # primalize based on support recovered
        primalized_cost_du, _ = sm.primalize(Q, X.T @ y, supp_du)
        primalized_cost_l1, _ = sm.primalize(Q, X.T @ y, supp_l1)
        
        # technicality for plotting
        cost_du.insert(0,0)
        cost_l0.insert(0,0)
        primalized_cost_du.insert(0,0)
        primalized_cost_l1.insert(0,0)
                
        # plot results
        plt.figure()
        plt.step(range(m+1), cost_du, 'k-')
        plt.step(range(m+1), cost_l0, 'r-')
        plt.step(range(m+1), primalized_cost_du,'g-')
        plt.step(range(m+1), primalized_cost_l1,'m-')
        
        plt.legend(legend)
        plt.ylabel('Function Value')
        plt.xlabel('Sparsity level (k)')
        plt.savefig(f'../figures/{m}_sparsity_{s}.pdf')
        
        if m == 10 and r == 5:
            import pdb; pdb.set_trace()
        