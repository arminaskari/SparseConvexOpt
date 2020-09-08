import sys
sys.path.append('../SparseOpt/')
import sparse_models as sm
from helper import * 
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
hsv = plt.get_cmap('hsv')
colors = hsv([0,0.6])

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
plt.figure(figsize=(6,6))
legend = [r'$\phi(k)$: dual lower bound',
          r'$\psi(k): \ell_0$ brute force',
          r'$\tilde{\psi}(k)$: dual primalized',
          r'$\ell_1$-reg: primalized']

# different dimensionality and rank levels to test
m_list = [10, 15]
s_list = [1e-1, 2e-1, 5e-1, 1]

for m in m_list:
    
    ticks = [str(i+1) for i in range(m)]
    for s in s_list:
        
        dual_rel_err = [[] for i in range(m)]
        l1_rel_err = [[] for i in range(m)]
        
        for j in range(100):
            np.random.seed(j)
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
            
            for jj in range(m):
                dual_rel_err[jj].append(np.abs(primalized_cost_du[jj] - \
                                              cost_l0[jj])/np.abs(cost_l0[jj]))
                l1_rel_err[jj].append(np.abs(primalized_cost_l1[jj] - \
                                              cost_l0[jj])/np.abs(cost_l0[jj]))

        data_plt = [dual_rel_err, l1_rel_err]
        offset = np.linspace(-0.25,0.25,len(data_plt))
        label_str = [r'$\widetilde{\psi}(k)$: dual primalized',
                     r'$\ell_1$-reg: primalized']
        
        off_cnt =0
        plt.figure()
        for model in data_plt:
            print([np.median(z) for z in model])
            print([np.std(z) for z in model])
            # outliers are hidden with sym=''
            bpl = plt.boxplot(model, 
                              positions=np.array(range(len(model)))*2.0-offset[off_cnt], 
                              sym='', 
                              widths=0.25)
            plt.plot([], c=colors[off_cnt], label=label_str[off_cnt])
            set_box_color(bpl, colors[off_cnt])
            off_cnt += 1
#         plt.show()
#         import pdb; pdb.set_trace()
        plt.legend(frameon=False)
        plt.ylabel('Relative Error')
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)
        plt.xlabel('Sparsity level (k)')
        plt.savefig(f'../figures/{m}_averaged_{s}.pdf')


   