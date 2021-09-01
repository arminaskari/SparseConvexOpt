"""
This module generates synthetic data by calling helper.py in SparseOpt package
and then runs the primalization procedure described in the paper to generate
sparse weight vectors that satisfy the bounds given in Proposition 5.2 in the paper.

Sample Usage
-------------
python synthetic_expt.py --dim 100,30,10 --loss l2 --l2reg constr --l0reg constr 
    --sparsity 0.1 --gamma 1 --lowrank soft --iterate_over rank 
    --param_grid 1,5,10  --save

the above generates
    - X \in R{100 x 30} (100 data points, 30 features) with X having a rank of
        10 (--lowrank soft means that X is approximately rank 10 and the
        remaining singular values do not explain much of the variance in X).
    - l2 loss means our loss function is ||Xw - y||_2^2
    - l2reg constr means we have a constraint ||w||_2^2 \leq \gamma
    - l0reg constr means we have a constraint ||w||_0 \leq \lambda where we
        vary \lambda
    - sparsity 0.1 means that the true weight vector w^\ast has 30 * 0.1 = 3
        non-zero entries
    - gamma is the l2 regularization value
    - iterate_over rank means that we fix values of lambda (based on param_grid)
        and then fit SparseModels with X varying in rank
    - param_grid (in conjunction with iterate_over) means to try lambda
        values of 1,5, and 10 and for each value, vary the rank of X from 1 to 30
        (since rank(X) = min(100,30) = 30)
    - save means we save a pickle of all the data and results
"""


import argparse
import os
import pickle

import numpy as np

from SparseOpt.helper import generate_data, svd_r
from SparseOpt.SparseModel import SparseModel


def get_parser():
    """Create argument parser"""

    parser = argparse.ArgumentParser("Sparse Optimization")
    parser.add_argument(
        "--dim",
        "-d",
        action="store",
        type=str,
        required=True,
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
        "--l2reg",
        "-l2",
        action="store",
        choices=["constr", "pen"],
        required=True,
        help='type of l2 regularization (either "constr" or "pen")',
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
        "--lowrank",
        "-lr",
        action="store",
        choices=["soft", "hard"],
        default="soft",
        help='type of synthetic low rank model to generate (either "soft" or "hard',
    )
    parser.add_argument(
        "--random_seed",
        "-rs",
        action="store",
        type=int,
        default=1,
        help="random seed for generating data",
    )
    parser.add_argument(
        "--param_grid",
        action="store",
        type=str,
        default=None,
        help="list of ranks if --iterate_over=lambda or list of lambdas if --iterate_over=rank",
    )
    parser.add_argument(
        "--iterate_over",
        action="store",
        type=str,
        default="lambda",
        choices=["lambda", "rank"],
        help="vary either lambda or rank of data matrix",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--outdir",
        default=None,
        required=False,
    )
    return parser


def get_args(args):
    """
    Get variables from argument parser

    Inputs:
      args (ArgumentParser)

    Outputs:
      n (int): number of data points
      m (int): number of features
      r (int): effective rank, value between 0 and min(n,m)
      loss (str): 'l2', 'hinge' or 'logistic'
      l0reg (str): 'constr' or 'pen'
      l2reg (str): 'constr' or 'pen'
      true_sparsity (double): desired sparsity level for synthetic linear model
      low_rank (str): 'soft' or 'hard' for generating synthetic data matrix n x m
        that either has exactly rank r (for 'hard' case) or generates matrix
        using scipy make_low_rank_matrix (for 'soft' case)
      gamma (double): value of l2 regularizaiton

    """

    # extract number of data points (n), number of features (m), and effective rank (r)
    dim = [int(s.strip()) for s in args.dim.split(",")]
    assert len(dim) == 3
    n, m, r_eff = dim[0], dim[1], dim[2]
    loss = args.loss
    l0reg = args.l0reg
    l2reg = args.l2reg
    true_sparsity = args.sparsity
    low_rank = args.lowrank
    gamma = args.gamma
    iterate_over = args.iterate_over

    # if we are doing l0 penalization and we are iterating over the rank,
    # we interpret the parameters as the different regularization values (floats).
    # Otherwise if l0reg is constr, then the lambda values are read as ints.
    # Similarly, it args.iterate_over == "lambda", then the param_grid
    # represents the different ranks to try out and is interpreted as ints
    if l0reg == "pen" and iterate_over == "rank":
        param_grid = [float(s.strip()) for s in args.param_grid.split(",")]
    else:
        param_grid = [int(s.strip()) for s in args.param_grid.split(",")]

    return (
        n,
        m,
        r_eff,
        loss,
        l0reg,
        l2reg,
        true_sparsity,
        low_rank,
        gamma,
        param_grid,
        iterate_over,
    )


def run(args):
    """
    Generate data and run synthetic experiment

    Input:
        args (ArgumentParser)

    Outputs:
        None, if args.save is True, saves pickle of solution sparsity, opt costs,
        bidual costs, the theoretical upper and lower bounds, and lambda values
        (l0 regularization values). Pickles are saved in args.outdir if specified,
        else in results/ directory in the same folder the module was called from

    """

    # parse arguments and generate data
    np.random.seed(args.random_seed)
    (
        n_pts,
        m_feats,
        r_eff,
        loss,
        l0reg,
        l2reg,
        sparsity,
        low_rank,
        gamma,
        param_grid,
        iterate_over,
    ) = get_args(args)
    params = generate_data(
        n=n_pts, m=m_feats, r=r_eff, loss=loss, sparsity=sparsity, low_rank=low_rank
    )

    if args.save:
        save_dir = "results/synthetic/l2%s/l0%s" % (l2reg, l0reg)
        filename = "X_%s_%s_%s" % (n_pts, m_feats, r_eff)
        save_object_pkl(params["Xtrue"], save_dir, filename, outdir=args.outdir)

    if iterate_over == "lambda":
        iterate_over_lambda(
            args, params, param_grid, loss, l0reg, l2reg, sparsity, gamma
        )
    else:
        iterate_over_rank(args, params, param_grid, loss, l0reg, l2reg, sparsity, gamma)


def generate_lambdas(l0reg, m):
    """generate grid of lambdas depending on type of regularization"""
    if l0reg == "constr":
        return np.array(list(range(1, m + 1)))
    return np.logspace(-6, -1, 30)


def save_object_pkl(obj, save_dir, filename, outdir=None):
    """save obj into a pickle file"""

    # if outdir specified, save data there, else save in current folder
    if outdir:
        path = os.path.join(outdir, save_dir)
    else:
        path = os.path.join(os.getcwd(), save_dir)

    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, filename + ".pkl"), "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fit_model(Xr_svd, y, loss, l0reg, l2reg, lmbda, gamma, Xtrue, delta_Xr):
    """
    Fit SparseModel

    Input:
        Xr_svd (dict<str>): dict with keys "U" "S" "V" such that Xr = U @ S @ V.T
            where Xr is a low rank approximation of Xtrue
        y (np.ndarray): (n, ) array
        loss (str): "l2" or "logistic"
        l0reg (str): "constr" or "pen"
        l2reg (str): "constr" or "pen"
        lmbda (double): l0 regularization value
        gamma (double): l2 regularization value
        Xtrue (np.ndarray): n x m array; true data matrix that has not been
            perturbed
        delta_Xr (np.ndarray): n x m array; constitutes "remaining" portion of
            Xr, that is Xr + delta_Xr = Xtrue

    Output:
        opt (double): primalized value OPT
        bd (double): value of bidual
        soln_sparsity (int): sparsity level of primalized solution
        zeta_r (double): quantity used to compute bounds in Proposition 5.2
        zeta (double): quantity used to compute bounds in Proposition 5.2
    """

    Xr = Xr_svd["U"] @ Xr_svd["S"] @ Xr_svd["V"].T

    # fit model
    clf = SparseModel(loss=loss, l0reg=l0reg, l2reg=l2reg, lmbda=lmbda, gamma=gamma)
    clf.fit(Xr_svd, y)

    # store results
    opt = clf.costs_
    bd = clf.bdcost_
    soln_sparsity = len(np.where(clf.coef_ != 0)[0])

    # if l2reg == "constr" compute theoretical bounds of Section 5 in paper
    # since the bounds only apply if we have an l2 constraint.
    if l2reg == "constr":
        d_ast, nu_r = clf.dual_problem(Xr)
        nu = clf.dual_problem(Xtrue)[1]
        assert np.abs(d_ast - clf.bdcost_) / np.abs(clf.bdcost_) <= 5e-4

        zeta_r = np.linalg.norm(delta_Xr.T @ nu_r) * np.sqrt(gamma)
        zeta = -np.linalg.norm(delta_Xr.T @ nu) * np.sqrt(gamma)
    # else theoretical bounds dont hold so return nans
    else:
        zeta_r = np.nan
        zeta = np.nan

    return opt, bd, soln_sparsity, zeta_r, zeta


def iterate_over_rank(args, params, lambda_values, loss, l0reg, l2reg, sparsity, gamma):
    """
    For different lambda_values, fit SparseModel by iterating over all possible
    low rank approximations of X

    Input:
        args (Argparse)
        params (dict): dict containing problem data
        rank_values (list<ints>): list containing different values of rank
        loss (str): "l2" or "logistic"
        l0reg (str): "constr" or "pen"
        l2reg (str): "constr" or "pen"
        sparsity (double): sparsity level of true linear model
        gamma (double): l2 regularization value

    Output:
        None
    """

    n_pts, m_feats = params["Xtrue"].shape[0], params["Xtrue"].shape[1]
    r_eff = params["rank"]

    rank_values = range(1, np.min((n_pts, m_feats)))

    # fit models by fixing a value of lambda and then iterating over all
    # low-rank versions of X by performing rank-r SVD of X
    for lam in lambda_values:
        opt_costs, bd_costs = [], []
        soln_sparsity = []
        zeta_r, zeta = [], []

        for rank in rank_values:
            # make rank-r approximation of X
            _, delta_Xr, Ur, Sr, Vrt = svd_r(params["Xtrue"].copy(), rank=rank)
            Xr_svd = {"U": Ur, "S": Sr, "V": Vrt.T}

            # fit model
            opt, bd, sparsity, z_r, z = fit_model(
                Xr_svd=Xr_svd,
                y=params["y"],
                loss=loss,
                l0reg=l0reg,
                l2reg=l2reg,
                lmbda=lam,
                gamma=gamma,
                Xtrue=params["Xtrue"],
                delta_Xr=delta_Xr,
            )

            opt_costs.append(opt)
            bd_costs.append(bd)
            soln_sparsity.append(sparsity)
            zeta_r.append(z_r)
            zeta.append(z)

        # pickle results
        if args.save:
            res = {
                "soln_sparsity": soln_sparsity,
                "opt": opt_costs,
                "bd": bd_costs,
                "zeta_r": zeta_r,
                "zeta": zeta,
                "ranks": rank_values,
            }
            save_dir = "results/synthetic/l2%s/l0%s" % (l2reg, l0reg)
            filename = "res-%s_%s_%s_%s_%s_%s_lmbda%s" % (
                n_pts,
                m_feats,
                r_eff,
                loss,
                sparsity,
                gamma,
                lam,
            )

            save_object_pkl(res, save_dir, filename, outdir=args.outdir)


def iterate_over_lambda(args, params, rank_values, loss, l0reg, l2reg, sparsity, gamma):
    """
    For different rank_values, create rank approximations to X and then fit Sparse
    Model by iterating over all values of lambda

    Input:
        args (Argparse)
        params (dict): dict containing problem data
        rank_values (list<ints>): list containing different values of rank
        loss (str): "l2" or "logistic"
        l0reg (str): "constr" or "pen"
        l2reg (str): "constr" or "pen"
        sparsity (double): sparsity level of true linear model
        gamma (double): l2 regularization value

    Output:
        None
    """

    n_pts, m_feats = params["Xtrue"].shape[0], params["Xtrue"].shape[1]
    r_eff = params["rank"]

    # solve sparse problem for different values of lambda
    lambda_values = generate_lambdas(l0reg=l0reg, m=m_feats)

    # fit models by fixing a value of the rank, truncating X to have that rank,
    # and then iterating over a grid of lambdas
    for rank in rank_values:
        # make rank-r approximation of X
        _, delta_Xr, Ur, Sr, Vrt = svd_r(params["Xtrue"].copy(), rank=rank)
        Xr_svd = {"U": Ur, "S": Sr, "V": Vrt.T}

        opt_costs, bd_costs = [], []
        soln_sparsity = []
        zeta_r, zeta = [], []

        # now for fixed rank version of X, iterate over all possible lambdas
        for lam in lambda_values:
            # fit model
            opt, bd, sparsity, zr, z = fit_model(
                Xr_svd=Xr_svd,
                y=params["y"],
                loss=loss,
                l0reg=l0reg,
                l2reg=l2reg,
                lmbda=lam,
                gamma=gamma,
                Xtrue=params["Xtrue"],
                delta_Xr=delta_Xr,
            )

            opt_costs.append(opt)
            bd_costs.append(bd)
            soln_sparsity.append(sparsity)
            zeta_r.append(zr)
            zeta.append(z)

        # pickle results
        if args.save:
            res = {
                "soln_sparsity": soln_sparsity,
                "opt": opt_costs,
                "bd": bd_costs,
                "zeta_r": zeta_r,
                "zeta": zeta,
                "lambdas": lambda_values,
            }
            save_dir = "results/synthetic/l2%s/l0%s" % (l2reg, l0reg)
            filename = "res-%s_%s_%s_%s_%s_%s_rank%s" % (
                n_pts,
                m_feats,
                r_eff,
                loss,
                sparsity,
                gamma,
                rank,
            )
            save_object_pkl(res, save_dir, filename, outdir=args.outdir)


def main(args=None):
    """Collect arguments and call run(args)"""

    parser = get_parser()
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    main()
