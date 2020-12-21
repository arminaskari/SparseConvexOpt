import numpy as np
import scipy
import sparse_models as sm
from sklearn.datasets import make_low_rank_matrix


def solve_l0(params, method='constrained', lmbdas=[], l2constr=False):
    """ Solve l0 constrained or penalized problem given params """

    cons_costs, pen_costs = [], []
    opt_costs, bd_costs = [], []
    supp = []
    m = params['X'].shape[1]

    # solve constrained problem
    for k in range(1, m + 1):
        s, c = sm.l0_constrained(params, k, l2constr=l2constr)
        cons_costs.append(c)
        supp.append(s)

    # if constrained, change k, else chamge lmbda
    if method == 'constrained':
        hyperparams = range(1, m + 1)
    elif method == 'penalized':
        hyperparams = lmbdas

    # solve l0 problem and dual
    for h in hyperparams:
        if method == 'constrained':
            params['target_sparsity'] = k
        elif method == 'penalized':
            c = sm.l0_penalized(h, supp, cons_costs)
            pen_costs.append(c)
            params['lmbda'] = h
        opt = sm.sf_primalize(params, method=method, l2constr=l2constr)
        bd_costs.append(params['bidual_cost'])
        opt_costs.append(opt)

    if method == 'constrained':
        return cons_costs, bd_costs, opt_costs
    return pen_costs, bd_costs, opt_costs


def generate_data(n,
                  m,
                  r,
                  loss='l2',
                  true_sparsity=0.2,
                  gamma=1,
                  low_rank='hard'):
    """ Generate random data for regression """

    assert np.min((m, n)) >= r

    # truncate to rank r
    if low_rank == 'hard':
        Xtrue = np.random.randn(n, m)

    # make data that is numerically low rank
    elif low_rank == 'soft':
        Xtrue = make_low_rank_matrix(n_samples=n,
                                 n_features=m,
                                 effective_rank=r,
                                 tail_strength=0.5,
                                 random_state=None)
    U, S, Vt = np.linalg.svd(Xtrue)
    U = U[:, :r]
    S = np.diag(S[:r])
    Vt = Vt[:r, :]

    X = U @ S @ Vt

    if low_rank == 'hard':
        Xtrue = X
    

    # generate synthetic data
    if loss == 'l2':
        beta_true = 5 * np.random.randn(m)
        idx = np.random.choice(m, int(m * true_sparsity), replace=False)
        u = np.zeros(m)
        u[idx] = 1
        beta_true = beta_true * u
        y = X @ beta_true + np.random.randn(n)
    else:
        raise NotImplementedError

    params = {
        'Xtrue': Xtrue,
        'X': X,
        'U': U,
        'S': S,
        'V': Vt.T,
        'gamma': gamma,
        'rank': r,
        'loss': loss,
        'y': y,
        'beta': beta_true,
        'true_sparsity': true_sparsity,
        'low_rank': low_rank
    }

    return params
