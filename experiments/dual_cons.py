import sys
sys.path.append('../SparseOpt/')
import sparse_models as sm
from helper import generate_data
import matplotlib.pyplot as plt
import pickle
import numpy as np

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


# init
np.random.seed(1)
n, m, r = 50, 10, 4
gamma, loss = 1, 'l2'
true_sparsity = 0.2
params = generate_data(n,
                       m,
                       r,
                       loss='l2',
                       true_sparsity=true_sparsity,
                       gamma=gamma)
supp, pcosts, bdcosts = [], [], []
optcosts = []

# solve l0 constrained, dual
for k in range(1, m + 1):
    s, c = sm.l0_constrained(params, k)
    params['target_sparsity'] = k
    opt = sm.sf_primalize(params, method='constrained')
    supp.append(s)
    pcosts.append(c)
    bdcosts.append(params['bidual_cost'])
    optcosts.append(opt)

k_vals = range(1, m + 1)
plt.figure(figsize=(7.5,5))
plt.plot(k_vals, pcosts, 'k-', label= r'$p^{\ast}(k)$')
plt.plot(k_vals, bdcosts, 'b-', label=r'$p^{\ast \ast}(k)$')
plt.plot(k_vals, optcosts, 'r-', label=r'OPT')

lb = []
for k in k_vals:
    if k + r + 2 >= m:
        lb.append(pcosts[-1])
    else:
        lb.append(pcosts[k + r + 2])
plt.plot(k_vals, lb, 'k--',  label=r'$p^\ast (k+r+2)$')
plt.legend(frameon=False)
plt.ylabel('Function Value')
plt.xlabel(r'$k$')
file_name = f'constrained_{n}_{m}_{r}_{true_sparsity}_{gamma}_{loss}'
plt.savefig(f'../figures/{file_name}.pdf')
plt.close()
