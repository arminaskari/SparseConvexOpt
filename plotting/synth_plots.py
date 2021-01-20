import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import glob
import math
from plot_utils import sf_bound

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
hsv = plt.get_cmap('hsv')
colors = hsv([0, 0.6, 0.9])

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 22

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc('lines', linewidth=4) # line thickness

def get_parameters(file_str):
    idx = [x.start() for x in re.finditer('_', file_str)]

    # if file_str contains path to pkl files (eg '../results/*.pkl)
    # start reading file name from the last '/'
    if file_str.rfind('/res') != -1:
        start_idx = file_str.rfind('/res') + 4
    else:
        start_idx = 0

    n = int(file_str[start_idx:idx[0]])
    m = int(file_str[idx[0] + 1:idx[1]])
    r = int(file_str[idx[1] + 1:idx[2]])
    loss = str(file_str[idx[2] + 1:idx[3]])
    sparsity = float(file_str[idx[3] + 1:idx[4]])
    method = str(file_str[idx[4] + 1:idx[5]])
    gamma = float(file_str[idx[5] + 1:-4])

    return n, m, r, loss, sparsity, method, gamma


if __name__ == "__main__":

    # import all data
    l2reg = ['constr', 'pen']
    l0reg = ['constr', 'pen']
    prefix_str = '/Users/aaskari/github-repos/SparseConvexOpt'
    for l2 in l2reg:
        for l0 in l0reg:
            files = glob.glob(
                f'{prefix_str}/results/synthetic/l2{l2}/l0{l0}/*.pkl')
            if len(files) > 0:
                for f in files:

                    # unpack params
                    _, m, r, _, _, low_rank, _ = get_parameters(f)
                    save_str = f'l0{l0}_l2{l2}_{f[f.rfind("/res") + 5:-4]}'
                    res = pickle.load(open(f, "rb"))
                    bidual, opt, lmbdas = res['bd'], res['opt'], res['lmbdas']
                    xvals, bnd, bnd_label = sf_bound(l0, l2, lmbdas, bidual, r)

                    saved_plots = glob.glob(
                        f'{prefix_str}/figures/{save_str}.pdf')
                    # plot results
                    plt.figure()
                    if l0 == 'constr':
                        bd_label = r'$p^{\ast \ast}(k)$'
                    else:
                        bd_label = r'$p^{\ast \ast}(\lambda)$'
                    
                    mean = np.array([np.mean(o) for o in opt])
                    std = np.array([np.std(o) for o in opt])

                    if l0 == 'pen':
                        plt.semilogx(1/lmbdas,
                                     bidual,
                                     'b-',
                                     alpha=0.7,
                                     label=bd_label)
                        plt.semilogx(1/lmbdas, mean, 'r-', alpha=0.7, label=r'OPT')
                        plt.fill_between(1/lmbdas,
                                     mean + std,
                                     mean - std,
                                     color='r',
                                     alpha=0.3)
                        plt.semilogx(1/xvals, bnd, 'k--', label=bnd_label)
                    else:
                        plt.plot(lmbdas,
                             bidual,
                             'b-',
                             alpha=0.7,
                             label=bd_label)
                        plt.plot(lmbdas, mean, 'r-', alpha=0.7, label=r'OPT')
                        plt.fill_between(lmbdas,
                                     mean + std,
                                     mean - std,
                                     color='r',
                                     alpha=0.3)
                        plt.plot(xvals, bnd, 'k--', label=bnd_label)


                    # shaded portion for opt
                    if l0 == 'constr':
                        plt.xlabel(r'$k$')
                    else:
                        plt.xlabel(r'$1/\lambda$')
                    plt.legend()
                    plt.savefig(f'{prefix_str}/figures/{save_str}.pdf')
                    plt.close()

                    # if l2reg = constr, then make numerical rank bound plots
                    if l2 == 'constr' and low_rank == 'soft':
                        plt.figure()
                        plt.plot(
                            lmbdas,
                            res['nr_ub'],
                            'k-',
                            label=r'$\sqrt{\gamma}\|\Delta X\nu_r^\ast\|_2$'
                        )
                        plt.plot(
                            lmbdas,
                            res['nr_lb'],
                            'k--',
                            label=r'$-\sqrt{\gamma}\|\Delta X \nu^\ast\|_2$'
                        )
                        if l0 == 'constr':
                            plt.xlabel(r'$k$')
                        else:
                            plt.xlabel(r'$\lambda$')
                        plt.legend()
                        plt.savefig(
                            f'{prefix_str}/figures/nr_{save_str}.pdf')
                        plt.close()
