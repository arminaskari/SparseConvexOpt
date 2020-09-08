import numpy as np
from pmlb import fetch_data, regression_dataset_names
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import sys
sys.path.append('../SparseOpt/')
import sparse_models as sm
from helper import factor_decomp
import matplotlib.pyplot as plt

dataset_names = regression_dataset_names
data_idx = [16, 26, 28, 31, 46, 53, 54, 56, 58, 79, 81, 82, 91, 98, 104, 109]

start = 15

for idx in data_idx:
    X, y = fetch_data(dataset_names[idx], return_X_y=True)
    X = scale(X)
    n, m = X.shape[0], X.shape[1]
    
    train_l0, train_du, train_l1 = [], [], []
    test_l0, test_du, test_l1 = [], [], []


    for rand_state in [43]:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, 
                             random_state=rand_state)


        # perform factor decomp for dual problem
        Sigma = X_train.T @ X_train
        factor_costs = factor_decomp(Sigma, rank=-1)
        D, L, _ = factor_decomp(Sigma, rank=8)
        Q = D + L @ L.T

        # sparse model solutions
        supp_l0, cost_l0 = sm.l0_bruteforce(Sigma, 
                                            -X_train.T @ y_train,
                                            list(range(1,m+1)))
        supp_du, cost_du = sm.cvx_dual(D, L, -X_train.T @ y_train, 
                                       list(range(1,m+1)),
                                       verbose=False)
        supp_l1  = sm.lasso_solve(X_train, y_train, list(range(1,m+1)))

        # primalize based on support recovered
        _, w_l0 = sm.primalize(Sigma, -X_train.T @ y_train, supp_l0)
        _, w_du = sm.primalize(Sigma, -X_train.T @ y_train, supp_du)
        _, w_l1 = sm.primalize(Sigma, -X_train.T @ y_train, supp_l1)

        # calculate train cost
        err = lambda X, y, w: 1/(2*n) * np.linalg.norm(X @ w - y) ** 2 
        train_l0.append([err(X_train, y_train, w) for w in w_l0])
        train_du.append([err(X_train, y_train, w) for w in w_du])
        train_l1.append([err(X_train, y_train, w) for w in w_l1])


        test_l0.append([err(X_test, y_test, w) for w in w_l0])
        test_du.append([err(X_test, y_test, w) for w in w_du])
        test_l1.append([err(X_test, y_test, w) for w in w_l1])


    plt.figure()
    plt.subplot(2,1,1)
    plt.title('Train')
    plt.plot(train_du[0],'r-')
    plt.plot(train_l1[0],'b-')
    plt.legend(['Dual', 'Lasso primalize'])
    plt.subplot(2,1,2)
    plt.title('Test')
    plt.plot(test_du[0],'r-')
    plt.plot(test_l1[0],'b-')
    plt.legend(['Dual', 'Lasso primalize'])
    plt.xlabel('Sparsity (k)')
    plt.ylabel('MSE')


    plt.figure()
    plt.plot(factor_costs)
    plt.xlabel('Rank')
    plt.ylabel('Cost')
    plt.show()



    import pdb; pdb.set_trace()