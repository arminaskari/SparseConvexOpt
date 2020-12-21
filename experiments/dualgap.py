import sys
sys.path.append('../SparseOpt/')
import sparse_models as sm
from helper import * 
import matplotlib.pyplot as plt
import pickle

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
hsv = plt.get_cmap('hsv')
colors = hsv([0,0.6,0.9])

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
legend = [r'$\phi(k)$: dual lower bound',
          r'$\psi(k): \ell_0$ brute force',
          r'$\tilde{\psi}(k)$: dual primalized',
          r'$\ell_1$-reg: primalized']

# different dimensionality and rank levels to test
m_list = [20]
s_list = [2e-2, 1e-1, 2e-1]
sparsity_lvl = 0.8

np.random.seed(1)
beta = None


for m in m_list:
    for s in s_list:
        
        ds_rel_err = [[] for i in range(m)]
        dc_rel_err = [[] for i in range(m)]
        l1_rel_err = [[] for i in range(m)]
        
        r = int(m * s)

        # generate synthetic data
        D, L, _ = generate_factor_model(m, r)
        X, y = factor_model_to_ls(D, L, sparsity_lvl = sparsity_lvl)
        Q = D + L @ L.T
        
        # sparse model solutions
#         supp_l0, cost_l0, betas_l0, gammas_l0 = sm.l0_bruteforce(Q, X.T @ y,
#                                                       list(range(1,m+1)),
#                                                       beta=beta)
        gammas_l0 = list(np.linspace(1e-1,20,50))
        gammas_l0 = list(np.logspace(-1,1.2,50))
#         supp_l0_pen, cost_l0_pen = sm.l0_penalized(supp_l0, cost_l0, gammas_l0)
        
        
        cost_l0pen_bidual = sm.l0_penalized_bidual_cvx(D, L, X.T @ y, gammas_l0,
                                          solver='mosek', verbose=False)
        vals = []
        for k in range(1,m+1):
            _, val = sm.cvx_bidual(D, L, X.T @ y, k, verbose=False)
            vals.append(val)
        _, val2 = sm.cvx_dual(D, L, X.T @ y, list(range(1,m+1)), verbose=False, solver='ecos')
        
        print([np.abs(a-b) for a,b in zip(vals, val2)])
        print('--------------')
        
        import pdb; pdb.set_trace()
        

        
#         supp_dc, cost_dc = sm.penalized_dual(D, L, X.T @ y, 
#                                              gammas_l0,
#                                              solver='scs')

#         print(cost_dc)
#         print(supp_dc)
#         import pdb; pdb.set_trace()
        
#         beta = 1
#         betas_l0 = [beta for i in range(m)]
#         supp_dc, cost_dc = sm.socp_dual(D, L, X.T @ y, 
#                                        list(range(1,m+1)),
#                                        betas_l0,
#                                        solver='mosek')
#         cost_bd = sm.socp_bidual(D, L, X.T @ y, 
#                                        list(range(1,m+1)),
#                                        betas_l0,
#                                        solver='mosek')
#         print(cost_bd)
        
        

        # primalize based on support recovered
#         primalized_cost_dc, _ = sm.primalize(Q, X.T @ y, supp_dc, beta=beta)
        
#         plt.figure()
#         #plt.step(range(m), cost_l0, 'r-', label=r'$p^\ast (k)$')
#         plt.step(range(m), cost_dc, 'k-', label=r'$d^\ast(k)$')
#         plt.step(np.arange(r+2, m), cost_dc[:m-r-2], 'k--', label=r'$d^\ast(k-\bar{k})$')
#         #plt.step(range(m), primalized_cost_dc, 'b-', label=r'$\tilde{p}(k)$')
        
        inv_gammas_l0 = np.array([1/g for g in gammas_l0])
        plt.figure(figsize=(7.5,5))
        #plt.plot(gammas_l0, cost_l0_pen, 'r-', label=r'$p^\ast (\gamma)$')
        plt.plot(inv_gammas_l0, cost_l0pen_bidual, 'k-', label=r'$d^\ast (\gamma)$')
        ub = [c + g * (r+1) for c,g in zip(cost_l0pen_bidual, gammas_l0)]
        plt.plot(inv_gammas_l0, ub, 'k--', label=r'$d^\ast (\gamma) + \gamma (r+1)$')
        
        print(inv_gammas_l0, cost_l0pen_bidual)

        
        plt.legend(frameon=False)
        plt.ylabel('Function Value')
        #plt.xlabel('Sparsity level (k)')
        plt.xlabel(r'$1/\gamma$')
        
        name = f'dualgap_{m}_{s}_penalized_sparsity_{sparsity_lvl}'
        plt.savefig(f'../figures/{name}.pdf')
        plt.close()
        
        data_plt = {'d': cost_l0pen_bidual, 'r': r, 'inv_gamma': inv_gammas_l0,
                    'D': D, 'L': L, 'p': X.T @ y,
                    'true_sparsity_lvl': sparsity_lvl}
        
        if m <= 20:
            data_plt['p'] = cost_l0_pen
            
        with open(f'../results/{name}.pkl', 'wb') as f:
            pickle.dump(data_plt, f)


   