import numpy as np
import scipy
import sparse_models as sm
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model import lars_path

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


def svd_r(X, rank=None):
    """ Truncate matrix and make low rank """

    U, S, Vt = np.linalg.svd(X)

    return U[:, :rank], np.diag(S[:rank]), Vt[:rank, :]


def generate_data(n,
                  m,
                  r,
                  ntest=100,
                  loss='l2',
                  true_sparsity=0.2,
                  gamma=1,
                  low_rank='hard'):
    """ Generate random data for regression """

    assert np.min((m, n)) >= r

    # truncate to rank r
    if low_rank == 'hard':
        Xtrue = np.random.randn(n, m)
        Xtest = np.random.randn(ntest,m)

    # make data that is numerically low rank
    elif low_rank == 'soft':
        Xfull = make_low_rank_matrix(n_samples=n+ntest,
                                 n_features=m,
                                 effective_rank=r,
                                 tail_strength=0.5,
                                 random_state=None)
        Xtrue = Xfull[:n, :]
        Xtest = Xfull[n:, :]

    U, S, Vt = svd_r(Xtrue.copy(), rank=r)
    X = U @ S @ Vt

    if low_rank == 'hard':
        Xtrue = X
        UU, SS, VVt = svd_r(Xtest.copy(), rank=r)
        Xtest = UU @ SS @ VVt
        
    

    # generate synthetic data
    if loss == 'l2':
        beta_true = 10 * np.random.randn(m)
        idx = np.random.choice(m, int(m * true_sparsity), replace=False)
        u = np.zeros(m)
        u[idx] = 1
        beta_true = beta_true * u
        y = X @ beta_true + np.random.randn(n)
        ytest = Xtest @ beta_true + np.random.randn(ntest)
    else:
        raise NotImplementedError

    params = {
        'Xtrue': Xtrue,
        'Xtest': Xtest,
        'X': X,
        'U': U,
        'S': S,
        'V': Vt.T,
        'gamma': gamma,
        'rank': r,
        'loss': loss,
        'y': y,
        'ytest': ytest,
        'beta': beta_true,
        'true_sparsity': true_sparsity,
        'low_rank': low_rank
    }

    return params

def lars_solve(X, y, k):
    """ Solve LASSO problem and get path """
    
    # solve lasso along path
    _, _, coefs = lars_path(X, y, method='lasso', verbose=True)
    supp_k, l1_sparsity = [], []
    for c in coefs.T:
        supp = np.where(c != 0)[0]
        supp_k.append(supp)
        l1_sparsity.append(len(supp))
    
    # extract support of solution)
    print(l1_sparsity)
    idx = np.where(np.array(l1_sparsity) == k)[0]
    assert len(idx) > 0
    costs, wstar = [], []
    for i in idx:
        c, w = ls_solve(X, y, supp_k[i])
        costs.append(c)
        wstar.append(w)
    idx_min = np.argmin(costs)
    return wstar[idx_min]


def ls_solve(X, y, idx=[]):
    """ Solve LS problem with subset of variables """
    
    w = np.zeros(X.shape[1])
    Xs = X[:, idx]
    w[idx] = np.linalg.inv(Xs.T @ Xs) @ Xs.T @ y
    cost = 1/(2*X.shape[0]) * np.linalg.norm(X @ w - y) ** 2

    return cost, w


