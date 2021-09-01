import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
import glob
import math
import argparse
from plot_utils import sf_bound

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
hsv = plt.get_cmap("hsv")
colors = hsv([0, 0.6, 0.9])

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=BIGGER_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
plt.rc("lines", linewidth=3)


def get_parser():
    """Create argument parser"""

    parser = argparse.ArgumentParser("Sparse Optimization")
    parser.add_argument(
        "--dim",
        "-d",
        action="store",
        type=str,
        default="100,10,1",
        help="n,m,r_eff for generated data (eg. -d 100,10,1)",
    )
    parser.add_argument(
        "--loss",
        "-l",
        action="store",
        choices=["l2", "logistic"],
        required=True,
        help='loss function (either "l2" or "logistic")',
    )
    parser.add_argument(
        "--l0reg",
        "-l0",
        action="store",
        choices=["constr", "pen"],
        required=True,
        help='type of l0 regularization (either "constr" or "pen")',
    )
    parser.add_argument(
        "--sparsity",
        "-s",
        action="store",
        type=float,
        default=0.05,
        help="true sparsity of underlying linear model in synthetic data",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        action="store",
        type=float,
        default=1,
        help="regularaization parameter of l2 regularization",
    )
    parser.add_argument(
        "--param_grid",
        action="store",
        type=str,
        default="5,10,30",
        help='list of ranks if vary is "lambda" and list of "lambda" if vary is rank',
    )
    parser.add_argument(
        "--vary",
        action="store",
        type=str,
        choices=["lmbda", "rank"],
        help="vary either lambda or rank to store data",
    )
    return parser


def get_args(args):
    """Get variables from argument parser"""

    l0reg = args.l0reg
    vary = args.vary
    dim = [int(s.strip()) for s in args.dim.split(",")]
    assert len(dim) == 3
    if l0reg == "pen" and vary == "lmbda":
        params = [float(s.strip()) for s in args.param_grid.split(",")]
    else:
        params = [int(s.strip()) for s in args.param_grid.split(",")]
    assert len(params) == 3
    loss = args.loss
    true_sparsity = args.sparsity
    gamma = args.gamma

    return dim[0], dim[1], dim[2], loss, l0reg, true_sparsity, gamma, params, vary


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
    run(args)


def run(args):

    # import all data
    l0reg = ["constr", "pen"]
    n, m, r, loss, l0reg, sparsity, gamma, hyperparams, vary = get_args(args)
    prefix_str = "/Users/aaskari/github-repos/SparseConvexOpt"
    folder_str = f"{prefix_str}/results/rank_bounds/l0{l0reg}"
    file_pattern = f"res-{n}_{m}_{r}_{loss}_{sparsity}_{gamma}"
    # file_pattern = f'leukemia-res-{loss}_{gamma}'

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.2, hspace=0.25)
    counter = -1
    for row in ax:
        for col in row:
            # if top left, plot spectrum
            if counter == -1:
                X = np.load(f"{folder_str}/X_{n}_{m}_{r}.npy")
                # X = np.load(f'{folder_str}/X_leukemia.npy')
                _, s, _ = np.linalg.svd(X)
                ev = np.cumsum(s ** 2) / np.sum(s ** 2)
                col.plot(range(1, len(s) + 1), ev, alpha=0.7)
                col.set_xlabel("Rank Approximation")
                col.set_ylabel("Explained Variance")

                if vary == "rank":
                    """
                    for i, rr in enumerate(hyperparams):
                        col.plot([rr],
                                 ev[rr],
                                 marker='o',
                                 markersize=5,
                                 color='k')
                        col.text(rr+1.5, ev[rr], f'{i+1}', fontsize=15)
                        col.annotate(f'{i+1}', (rr, ev[rr]))
                    """

            # load sparse solutions
            else:
                res = pickle.load(
                    open(
                        f"{folder_str}/{file_pattern}_{vary}{hyperparams[counter]}.pkl",
                        "rb",
                    )
                )
                bidual, opt, lmbdas = (
                    np.array(res["bd"]),
                    np.array(res["opt"]),
                    res["lmbdas"],
                )
                if vary == "lmbda":
                    lmbdas = range(1, len(bidual) + 1)
                zeta_r, zeta = np.abs(np.array(res["zeta_r"])), np.abs(
                    np.array(res["zeta"])
                )
                xvals, bnd, bnd_label = sf_bound(
                    l0reg, "constr", lmbdas, bidual, hyperparams[counter], rank=True
                )
                if l0reg == "pen":
                    # lower bound

                    col.plot(
                        lmbdas,
                        bidual - zeta_r,
                        "b-",
                        alpha=0.7,
                        label=r"$p^{\ast \ast}(\lambda, X_r) - \zeta_r$",
                    )

                    # OPT value
                    mean = np.array([np.mean(o + z) for o, z in zip(opt, zeta)])
                    std = np.array([np.std(o + z) for o, z in zip(opt, zeta)])

                    col.plot(lmbdas, mean, "r-", alpha=0.7, label=f"OPT $+ \zeta$")
                    col.fill_between(
                        lmbdas, mean + std, mean - std, color="r", alpha=0.3
                    )

                    # upper bound
                    col.plot(xvals, (bnd + zeta), "k--", label=bnd_label)

                    if vary == "rank":
                        col.set_xlabel("$\lambda$")
                    else:
                        col.set_xlabel("Rank")

                    if vary == "rank":
                        col.set_title(f"Rank {hyperparams[counter]}")
                    else:
                        col.set_title(f"$\lambda = {hyperparams[counter]}$")

                    plt.legend()

                    # col.text(0.95, 0.98, f'{counter+1}', size=20, color='k',
                    #        verticalalignment='top', transform=col.transAxes)

            counter += 1

    plt.savefig(
        f"{prefix_str}/figures/nr_{n}_{m}_{r}_{loss}_{sparsity}_{gamma}_{vary}_{args.param_grid}.pdf"
    )
    # plt.savefig(f'{prefix_str}/figures/nr_leukemia_{loss}_{gamma}_{vary}_{args.param_grid}.pdf')
    plt.close()


if __name__ == "__main__":
    main()
