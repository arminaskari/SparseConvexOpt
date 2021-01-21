import sys
import pandas as pd
import argparse
sys.path.append('../SparseOpt/')
from SparseModel import SparseModel
from helper import generate_data, svd_r
import pickle
import pdb
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import scale
import multiprocessing


def get_parser():
    """ Create argument parser """

    parser = argparse.ArgumentParser('Sparse Optimization for real data')
    parser.add_argument(
        '--l0reg',
        '-l0',
        action='store',
        choices=['constr', 'pen'],
        required=True,
        help='type of l0 regularization (either "constr" or "pen")')
    parser.add_argument('--gamma',
                        '-g',
                        action='store',
                        type=float,
                        default=1,
                        help='regularaization parameter of l2 regularization')
    parser.add_argument('--randomseed',
                        '-rs',
                        action='store',
                        type=int,
                        default=1,
                        help='random seed for generating data')
    parser.add_argument(
        '--param_grid',
        action='store',
        type=str,
        default='5,10,30',
        help=
        'list of ranks if vary is "lambda" and list of "lambda" if vary is rank'
    )
    parser.add_argument('--vary',
                        action='store',
                        type=str,
                        default='lambda',
                        choices=['lambda', 'rank'],
                        help='vary either lambda or rank to store data')
    parser.add_argument('--cpus',
                        action='store',
                        type=int,
                        default=1,
                        help='number of cpus to use for parallelization')
    return parser


def get_args(args):
    """ Get variables from argument parser """

    l0reg = args.l0reg
    vary = args.vary
    if l0reg == 'pen' and vary == 'rank':
        params = [float(s.strip()) for s in args.param_grid.split(',')]
    else:
        params = [int(s.strip()) for s in args.param_grid.split(',')]
    gamma = args.gamma
    cpus = args.cpus

    return l0reg, gamma, params, vary, cpus


def vary_lambda(params, ranks, loss, l0reg, gamma):
    """ Vary lambda for different fixed values of rank """

    # solve sparse problem for each value of l
    n, m, r_eff = params['Xtrue'].shape[0], params['Xtrue'].shape[1], params[
        'rank']
    if l0reg == 'constr':
        lmbdas = np.array([i for i in range(1, m + 1)])
    elif l0reg == 'pen':
        lmbdas = np.logspace(-6, -1, 30)

    # iterate over ranks
    for r in ranks:
        # make rank-r approximation of X
        U, S, Vt = svd_r(params['Xtrue'].copy(), rank=r)
        X = U @ S @ Vt
        dX = params['Xtrue'].copy() - X

        opt_costs, bd_costs = [], []
        soln_sparsity = []
        zeta_r, zeta = [], []

        for l in lmbdas:
            # fit model
            clf = SparseModel(loss=loss,
                              l0reg=l0reg,
                              l2reg='constr',
                              lmbda=l,
                              gamma=gamma,
                              threads=1)
            clf.fit(params['Xtrue'], params['y'], rank=r)

            # store results
            opt_costs.append(clf.costs_)
            bd_costs.append(clf.bdcost_)
            soln_sparsity.append(len(np.where(clf.coef_ != 0)[0]))
            print(clf.bdcost_)

            # compute bounds on numerical rank approximation
            d_ast, nu_r = clf._dual_problem(X)
            nu = clf._dual_problem(params['Xtrue'])[1]
            #print(d_ast, clf.bdcost_)
            #print(np.abs(d_ast - clf.bdcost_) / np.abs(clf.bdcost_))
            assert np.abs(d_ast - clf.bdcost_) / np.abs(clf.bdcost_) <= 5e-4

            zeta_r.append(np.linalg.norm(dX.T @ nu_r) * np.sqrt(gamma))
            zeta.append(np.linalg.norm(dX.T @ nu) * np.sqrt(gamma))

        # pickle results
        res = {
            'soln_sparsity': soln_sparsity,
            'opt': opt_costs,
            'bd': bd_costs,
            'zeta_r': zeta_r,
            'zeta': zeta,
            'lmbdas': lmbdas
        }
        pickle.dump(res, open(f"leukemia-res-{loss}_{gamma}_rank{r}.pkl",
                              "wb"))


def vary_rank(params, lmbdas, loss, l0reg, gamma):
    """ Vary rank for different fixed values of lmbda """

    U, S, Vt = np.linalg.svd(params['Xtrue'])
    S = np.diag(S)

    # iterate over lmbdas
    for l in lmbdas:
        opt_costs, bd_costs = [], []
        soln_sparsity = []
        zeta_r, zeta = [], []

        ## solve for nu once
        clf_nu = SparseModel(loss=loss,
                             l0reg=l0reg,
                             l2reg='constr',
                             lmbda=l,
                             gamma=gamma,
                             n_trials=20,
                             threads=1)
        clf_nu.y = params['y']
        nu = clf_nu._dual_problem(params['Xtrue'])[1]

        for r in range(1, np.min(params['Xtrue'].shape)):
            # make rank-r approximation of X
            UU = U[:, :r]
            SS = S[:r, :r]
            VVt = Vt[:r, :]
            X = UU @ SS @ VVt
            dX = params['Xtrue'] - X

            # fit model
            clf = SparseModel(loss=loss,
                              l0reg=l0reg,
                              l2reg='constr',
                              lmbda=l,
                              gamma=gamma,
                              n_trials=20,
                              threads=1)
            clf.fit({'U': UU, 'S': SS, 'V': VVt.T}, params['y'])

            # store results
            opt_costs.append(clf.costs_)
            bd_costs.append(clf.bdcost_)
            soln_sparsity.append(len(np.where(clf.coef_ != 0)[0]))
            print(clf.bdcost_)

            # compute bounds on numerical rank approximation
            d_ast, nu_r = clf._dual_problem(X)

            assert np.abs(d_ast - clf.bdcost_) / np.abs(clf.bdcost_) <= 8e-3

            zeta_r.append(np.linalg.norm(dX.T @ nu_r) * np.sqrt(gamma))
            zeta.append(np.linalg.norm(dX.T @ nu) * np.sqrt(gamma))

        # pickle results
        res = {
            'soln_sparsity': soln_sparsity,
            'opt': opt_costs,
            'bd': bd_costs,
            'zeta_r': zeta_r,
            'zeta': zeta,
            'lmbdas': lmbdas
        }
        pickle.dump(res, open(f"leukemia-res-{loss}_{gamma}_lmbda{l}.pkl",
                              "wb"))


def vary_rank_parallel(params, lmbdas, loss, l0reg, gamma, cpus):
    """ Vary rank for different fixed values of lmbda """

    U, S, Vt = np.linalg.svd(params['Xtrue'])
    S = np.diag(S)

    # iterate over lmbdas
    for l in lmbdas:

        ## solve for nu once
        clf = SparseModel(loss=loss,
                          l0reg=l0reg,
                          l2reg='constr',
                          lmbda=l,
                          gamma=gamma,
                          n_trials=20,
                          threads=1)
        clf.y = params['y']
        nu = clf._dual_problem(params['Xtrue'])[1]

        dXs, Xdicts = [], []

        for r in range(1, np.min(params['Xtrue'].shape) + 1):
            # make rank-r approximation of X
            UU = U[:, :r]
            SS = S[:r, :r]
            VVt = Vt[:r, :]
            X = UU @ SS @ VVt
            dX = params['Xtrue'] - X

            # fit model
            dXs.append(dX)
            Xdicts.append({'U': UU, 'S': SS, 'V': VVt.T})

        zipped_list = zip([clf] * len(dXs), dXs, [nu] * len(dXs), Xdicts,
                          [params['y']] * len(dXs))
        #cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(cpus)
        res = pool.starmap(parallel_solve, zipped_list)
        pool.close()
        pool.join()

        opt_costs = [x[0] for x in res]
        bd_costs = [x[1] for x in res]
        soln_sparsity = [x[2] for x in res]
        zeta_r = [x[3] for x in res]
        zeta = [x[4] for x in res]

        # pickle results
        res = {
            'soln_sparsity': soln_sparsity,
            'opt': opt_costs,
            'bd': bd_costs,
            'zeta_r': zeta_r,
            'zeta': zeta,
            'lmbdas': lmbdas
        }
        pickle.dump(res, open(f"leukemia-res-{loss}_{gamma}_lmbda{l}.pkl",
                              "wb"))


def parallel_solve(SparseModel, dX, nu, Xdict, y):
    """ Do parallel solve on Sparse Model """

    SparseModel.fit(Xdict, y)
    X = Xdict['U'] @ Xdict['S'] @ Xdict['V'].T
    d_ast, nu_r = SparseModel._dual_problem(X)

    opt_cost = SparseModel.costs_
    bd_cost = SparseModel.bdcost_
    soln_sparsity = len(np.where(SparseModel.coef_ != 0)[0])

    zeta_r = np.linalg.norm(dX.T @ nu_r) * np.sqrt(SparseModel.gamma)
    zeta = np.linalg.norm(dX.T @ nu) * np.sqrt(SparseModel.gamma)

    return opt_cost, bd_cost, soln_sparsity, zeta_r, zeta


def run(args):
    """ Generate data and run synthetic experiment """

    # parse arguments and generate data
    l0reg, gamma, hyperparams, vary, cpus = get_args(args)
    np.random.seed(args.randomseed)

    # load leuekemia data
    df = pd.read_csv('../data/leukemia_small.csv')
    X = scale(df.values.T)
    y = np.array([1 if 'ALL' in txt else -1 for txt in df.columns.values])
    loss = 'logistic'

    params = {'Xtrue': X, 'y': y}

    print('fitting models....')
    if vary == 'lambda':
        vary_lambda(params, hyperparams, loss, l0reg, gamma)
    else:
        vary_rank_parallel(params, hyperparams, loss, l0reg, gamma, cpus)


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main()
