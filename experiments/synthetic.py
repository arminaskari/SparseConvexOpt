import sys
import argparse
sys.path.append('../SparseOpt/')
from SparseModel import SparseModel
from helper import generate_data, svd_r
import pickle
import pdb
import numpy as np


def get_parser():
    """ Create argument parser """

    parser = argparse.ArgumentParser('Sparse Optimization')
    parser.add_argument('--dim',
                        '-d',
                        action='store',
                        type=str,
                        required=True,
                        help='n,m,r_eff for generated data (eg. -d 100,10,1)')
    parser.add_argument('--loss',
                        '-l',
                        action='store',
                        choices=['l2', 'logistic'],
                        required=True,
                        help='loss function (either "l2" or "logistic")')
    parser.add_argument(
        '--l2reg',
        '-l2',
        action='store',
        choices=['constr', 'pen'],
        required=True,
        help='type of l2 regularization (either "constr" or "pen")')
    parser.add_argument(
        '--l0reg',
        '-l0',
        action='store',
        choices=['constr', 'pen'],
        required=True,
        help='type of l0 regularization (either "constr" or "pen")')
    parser.add_argument(
        '--sparsity',
        '-s',
        action='store',
        type=float,
        default=0.05,
        help='true sparsity of underlying linear model in synthetic data')
    parser.add_argument('--gamma',
                        '-g',
                        action='store',
                        type=float,
                        default=1,
                        help='regularaization parameter of l2 regularization')
    parser.add_argument(
        '--lowrank',
        '-lr',
        action='store',
        choices=['soft','hard'],
        default='soft',
        help=
        'type of synthetic low rank model to generate (either "soft" or "hard')
    parser.add_argument('--randomseed',
                        '-rs',
                        action='store',
                        type=int,
                        default=1,
                        help='random seed for generating data')
    return parser


def get_args(args):
    """ Get variables from argument parser """

    dim = [int(s.strip()) for s in args.dim.split(',')]
    assert len(dim) == 3
    loss = args.loss
    l0reg = args.l0reg
    l2reg = args.l2reg
    true_sparsity = args.sparsity
    gamma = args.gamma
    low_rank = args.lowrank

    return dim[0], dim[1], dim[
        2], loss, l0reg, l2reg, true_sparsity, low_rank, gamma


def run(args):
    """ Generate data and run synthetic experiment """

    # parse arguments and generate data
    n, m, r_eff, loss, l0reg, l2reg, sparsity, low_rank, gamma = get_args(args)
    np.random.seed(args.randomseed)
    params = generate_data(n,
                           m,
                           r_eff,
                           loss=loss,
                           sparsity=sparsity,
                           low_rank=low_rank)



    # solve sparse problem for each value of l
    opt_costs, bd_costs = [], []
    soln_sparsity = []
    nr_lb, nr_ub = [], []
    if l0reg == 'constr':
        lmbdas = np.array([i for i in range(1, m + 1)])
    elif l0reg == 'pen':
        lmbdas = np.logspace(-6, -1, 30)

    print('fitting models....')
    for l in lmbdas:
        print(l)
        # fit model
        clf = SparseModel(loss=loss,
                          l0reg=l0reg,
                          l2reg=l2reg,
                          lmbda=l,
                          gamma=gamma)
        clf.fit(params['X'], params['y'], rank=r_eff)

        # store results
        opt_costs.append(clf.costs_)
        bd_costs.append(clf.bdcost_)
        soln_sparsity.append(len(np.where(clf.coef_ != 0)[0]))
        print(clf.bdcost_)
        # compute bounds on numerical rank approximation
        if l2reg == 'constr' and low_rank == 'soft':
            dX = params['dX']
            d_ast, nu_r = clf._dual_problem(params['X'])
            nu = clf._dual_problem(params['Xtrue'])[1]
            print(d_ast, clf.bdcost_)
            print(np.abs(d_ast - clf.bdcost_)/np.abs(clf.bdcost_))
            assert np.abs(d_ast - clf.bdcost_)/np.abs(clf.bdcost_) <= 5e-4
            
            nr_ub.append(np.linalg.norm(dX.T @ nu_r) * np.sqrt(gamma))
            nr_lb.append(-np.linalg.norm(dX.T @ nu) * np.sqrt(gamma))

        #l0coef_, l0cost = clf._l0fit(params['X'], params['y'], k=lstar)
        #l1coef_ = lars_solve(params['X'], params['y'], k=lstar)

    # pickle results
    res = {
        'soln_sparsity': soln_sparsity,
        'opt': opt_costs,
        'bd': bd_costs,
        'nr_lb': nr_lb,
        'nr_ub': nr_ub,
        'lmbdas': lmbdas
    }
    pickle.dump(
        res,
        open(
            f"/Users/aaskari/github-repos/SparseConvexOpt/results/synthetic/l2{l2reg}/l0{l0reg}/res-{n}_{m}_{r_eff}_{loss}_{sparsity}_{low_rank}_{gamma}.pkl",
            "wb"))


def main(args=None):

    parser = get_parser()
    args = parser.parse_args(args)
    run(args)


if __name__ == '__main__':
    main()
