import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import glob
import math

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
hsv = plt.get_cmap('hsv')
colors = hsv([0, 0.6, 0.9])


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize


def get_parameters(file_str):
    idx = [x.start() for x in re.finditer('_', file_str)]

    # if file_str contains path to pkl files (eg '../results/*.pkl)
    # start reading file name from the last '/'
    if file_str.rfind('/') != -1:
        start_idx = file_str.rfind('/') + 1
    else:
        start_idx = 0

    n = int(file_str[start_idx:idx[0]])
    m = int(file_str[idx[0] + 1:idx[1]])
    r = int(file_str[idx[1] + 1:idx[2]])
    gamma = float(file_str[idx[2] + 1:idx[3]])
    true_sparsity = float(file_str[idx[3] + 1:idx[4]])
    method = file_str[idx[4] + 1:-4]

    return n, m, r, gamma, true_sparsity, method


if __name__ == "__main__":

    # import all data
    files = glob.glob('../results/synthetic/*.pkl')
    idx_to_keep = []
    for i, f in enumerate(files):
        n, m, r, gamma, true_sparsity, method = get_parameters(f)

        # condition for which files to keep to make plots
        if method == 'constrained' and n == 100 and m == 20 and gamma == 0.1 and true_sparsity == 0.2:
            idx_to_keep.append(i)
            file_name = f'{n}_{m}_X_{gamma}_{true_sparsity}_{method}'

    files_to_plot = [files[i] for i in idx_to_keep]
    files_to_plot.sort()

    # plot data for specific files
    counter = 0
    fig, ax = plt.subplots(nrows=2,
                           ncols=math.ceil(len(files_to_plot) / 2),
                           figsize=(12, 7))
    for row in ax:
        for col in row:
            # since we add extra columns if we have odd number, need the if
            # statement
            if counter <= len(files_to_plot)-1:
                f = files_to_plot[counter]
                res = pickle.load(open(f, "rb"))
                n, m, r, gamma, true_sparsity, method = get_parameters(f)

                primal, bidual, opt = res['primal'], res['bidual'], res[
                    'primalized']
                if method == 'constrained':
                    k_vals = range(1, m + 1)
                    col.plot(k_vals, primal, 'k-', label=r'$p^{\ast}(k)$')
                    col.plot(k_vals, bidual, 'b-', label=r'$p^{\ast \ast}(k)$')
                    col.plot(k_vals, opt, 'r-', label=r'Primalization')

                    lb = []
                    for k in k_vals:
                        if k + r + 2 >= m:
                            lb.append(primal[-1])
                        else:
                            lb.append(primal[k + r + 2])
                    col.plot(k_vals, lb, 'k--', label=r'$p^\ast (k+r+2)$')

                elif method == 'penalized':
                    lmbdas = res['lmbdas']
                    col.plot(lmbdas,
                             primal,
                             'k-',
                             label=r'$p^{\ast}(\lambda)$')
                    col.plot(lmbdas,
                             bidual,
                             'b-',
                             label=r'$p^{\ast \ast}(\lambda)$')
                    col.plot(lmbdas, opt, 'r-', label=r'Primalization')

                    ub = [bd + l * (r + 1) for bd, l in zip(bidual, lmbdas)]
                    col.plot(lmbdas,
                             ub,
                             'k--',
                             label=r'$p^{\ast \ast}(\lambda) + \lambda (r+1)$')
                
                col.legend(frameon=False)
                col.set_title(
                    f'n={n}, m={m}, r={r}, g={gamma}, s={true_sparsity}')

                counter += 1

    if method == 'constrained':
        fig.text(0.5, 0.04, r'$k$', ha='center', va='center')
    else:
        fig.text(0.5, 0.04, r'$\lambda$', ha='center', va='center')

    plt.savefig(f'../figures/{file_name}.pdf')
    plt.close()
