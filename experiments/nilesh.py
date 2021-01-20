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

    return l0reg, gamma, params, vary


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
                              gamma=gamma)
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
        pickle.dump(
            res,
            open(
                f"leukemia-res-{loss}_{gamma}_rank{r}.pkl",
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

        for r in range(1, np.min(params['Xtrue'].shape), 5):
            # make rank-r approximation of X
            UU = U[:, :r] 
            SS = S[:r, :r]
            VVt = Vt[:r,:]
            X = UU @ SS @ VVt
            dX = params['Xtrue'] - X

            # fit model
            clf = SparseModel(loss=loss,
                              l0reg=l0reg,
                              l2reg='constr',
                              lmbda=l,
                              gamma=gamma,
                              n_trials=10)
            clf.fit({'U': UU, 'S': SS, 'V': VVt.T},
                    params['y'])

            # store results
            opt_costs.append(clf.costs_)
            bd_costs.append(clf.bdcost_)
            soln_sparsity.append(len(np.where(clf.coef_ != 0)[0]))
            print(clf.bdcost_)
            
            # compute bounds on numerical rank approximation
            d_ast, nu_r = clf._dual_problem(X)
            if r == 1:
                nu = clf._dual_problem(params['Xtrue'])[1]


            print(d_ast, clf.bdcost_)
            print(np.abs(d_ast - clf.bdcost_) / np.abs(clf.bdcost_))
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
        pickle.dump(
            res,
            open(
                f"leukemia-res-{loss}_{gamma}_lmbda{l}.pkl",
                "wb"))


def run(args):
    """ Generate data and run synthetic experiment """

    # parse arguments and generate data
    l0reg, gamma, hyperparams, vary = get_args(args)
    np.random.seed(args.randomseed)

    # load leuekemia data
    df = pd.read_csv('../data/leukemia_small.csv')
    X = scale(df.values.T)
    y = np.array([1 if 'ALL' in txt else -1 for txt in df.columns.values])

    params = {'Xtrue': X, 'y': y} 
    loss = 'logistic'

    print('fitting models....')
    if vary == 'lambda':
        vary_lambda(params, hyperparams, loss, l0reg, gamma)
    else:
        vary_rank(params, hyperparams, loss, l0reg, gamma)


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main()
