import numpy as np
import scipy
import sparse_models as sm
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model import lars_path

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def svd_r(X, rank=None):
    """ Truncate matrix and make low rank """

    U, S, Vt = np.linalg.svd(X)

    return U[:, :rank], np.diag(S[:rank]), Vt[:rank, :]


def generate_data(n,
                  m,
                  r,
                  ntest=100,
                  loss='l2',
                  sparsity=0.2,
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
        Xfull = 10 * make_low_rank_matrix(n_samples=n+ntest,
                                 n_features=m,
                                 effective_rank=r,
                                 tail_strength=1e-5,
                                 random_state=1)
        Xtrue = Xfull[:n, :]
        Xtest = Xfull[n:, :]

    U, S, Vt = svd_r(Xtrue.copy(), rank=r)
    X = U @ S @ Vt
    dX = Xtrue - X

    if low_rank == 'hard':
        Xtrue = X
        dX = np.zeros(Xtrue.shape)
        UU, SS, VVt = svd_r(Xtest.copy(), rank=r)
        Xtest = UU @ SS @ VVt 
    
    # generate synthetic data
    beta_true = 5 * np.random.randn(m)
    idx = np.random.choice(m, int(m * sparsity), replace=False)
    u = np.zeros(m)
    u[idx] = 1
    beta_true = beta_true * u

    if loss == 'l2': 
        y = X @ beta_true + np.random.randn(n)
        ytest = Xtest @ beta_true + np.random.randn(ntest)
    elif loss == 'logistic':
        y = 2 * np.round(sigmoid(X @ beta_true + np.random.randn(n))) - 1
        ytest = 2 * np.round(sigmoid(Xtest @ beta_true + np.random.randn(ntest))) - 1
    else:
        raise NotImplementedError

    params = {
        'Xtrue': Xtrue,
        'Xtest': Xtest,
        'dX': dX,
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
        'sparsity': sparsity,
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


