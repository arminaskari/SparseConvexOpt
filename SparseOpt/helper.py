import numpy as np
import scipy
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model import lars_path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def svd_r(X, rank=None):
    """Truncate matrix and make low rank"""

    U, S, Vt = np.linalg.svd(X)
    Ur, Sr, Vtr = U[:, :rank], np.diag(S[:rank]), Vt[:rank, :]

    Xr = Ur @ Sr @ Vtr
    delta_X = X - Xr

    return Xr, delta_X, Ur, Sr, Vtr


def generate_data(
    n, m, r, ntest=100, loss="l2", sparsity=0.2, gamma=1, low_rank="hard"
):
    """
    Generate synthetic data for experiments

    Input:
        n (int): number of data points
        m (int): number of features
        r (int): effective rank, value between 0 and min(n,m)
        ntest (int): number of test points
        loss (str): 'l2', 'hinge' or 'logistic'
        sparsity (double): floor(sparsity * m) = number of non-zero entires in weight vector for linear model
        gamma (double): value of l2 regularization
        low_rank (str): 'soft' or 'hard' for generating synthetic data matrix n x m that either has
            exactly rank r (for 'hard' case) or generates matrix using scipy make_low_rank_matrix
            (for 'soft' case)

    Output:
        params (dict): Dictionary containing
          "Xtrue": training data before being decomposed into X (rank r) and dX (low rank part)
          "Xtest": test data
          "dX": matrix that satisfies X + dX = Xtrue
          "X": rank r svd of Xtrue
          "U", "S", "V": SVD decomposition of X
          "rank": rank of data matrix
          "loss": "l2", "hinge" or "logistic"
          "y": targets for supervised learning problem
          "ytest": targets for test data
          "beta": ground truth coefficient vector
          "sparsity": number of non-zero entries of beta
          "low_rank": "soft" or "hard"
    """

    assert np.min((m, n)) >= r

    # truncate to rank r
    if low_rank == "hard":
        Xtrue = np.random.randn(n, m)
        Xtest = np.random.randn(ntest, m)

    # make data that is numerically low rank
    elif low_rank == "soft":
        Xfull = 10 * make_low_rank_matrix(
            n_samples=n + ntest,
            n_features=m,
            effective_rank=r,
            tail_strength=1e-5,
            random_state=1,
        )
        Xtrue = Xfull[:n, :]
        Xtest = Xfull[n:, :]

    X, dX, U, S, Vt = svd_r(Xtrue.copy(), rank=r)

    if low_rank == "hard":
        Xtrue = X
        dX = np.zeros(Xtrue.shape)
        Xtest, _, _, _, _ = svd_r(Xtest.copy(), rank=r)
        Xtest = UU @ SS @ VVt

    # generate synthetic data
    beta_true = 5 * np.random.randn(m)
    idx = np.random.choice(m, int(m * sparsity), replace=False)
    u = np.zeros(m)
    u[idx] = 1
    beta_true = beta_true * u

    if loss == "l2":
        y = X @ beta_true + np.random.randn(n)
        ytest = Xtest @ beta_true + np.random.randn(ntest)
    elif loss == "logistic":
        y = 2 * np.round(sigmoid(X @ beta_true + np.random.randn(n))) - 1
        ytest = 2 * np.round(sigmoid(Xtest @ beta_true + np.random.randn(ntest))) - 1
    else:
        raise NotImplementedError

    params = {
        "Xtrue": Xtrue,
        "Xtest": Xtest,
        "dX": dX,
        "X": X,
        "U": U,
        "S": S,
        "V": Vt.T,
        "gamma": gamma,
        "rank": r,
        "loss": loss,
        "y": y,
        "ytest": ytest,
        "beta": beta_true,
        "sparsity": sparsity,
        "low_rank": low_rank,
    }

    return params


"""
def lars_solve(X, y, k):
    # Solve LASSO problem and get path

    # solve lasso along path
    _, _, coefs = lars_path(X, y, method="lasso", verbose=True)
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
    # Solve LS problem with subset of variables

    w = np.zeros(X.shape[1])
    Xs = X[:, idx]
    w[idx] = np.linalg.inv(Xs.T @ Xs) @ Xs.T @ y
    cost = 1 / (2 * X.shape[0]) * np.linalg.norm(X @ w - y) ** 2

    return cost, w
"""
