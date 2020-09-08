import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
import sys
sys.path.append('../SparseOpt/')
import sparse_models as sm
import scipy
from helper import factor_decomp
import matplotlib.pyplot as plt
import pickle

## load data - we do as on page 24 of https://projecteuclid.org/download/pdfview_1/euclid.aos/1458245736
# zero mean features/response and then normalize to have unit norm
df = pd.read_csv('../data/leukemia_small.csv')
X = scale(df.values.T)
y = np.array([1 if 'ALL' in txt else 0 for txt in df.columns.values])
########################

rank = int(sys.argv[1])

n, m = X.shape[0], X.shape[1]
    
train_du, train_l1, train_l1f = [], [], []
test_du, test_l1, test_l1f = [], [], []

## train models
for i in range(20):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, 
                         random_state=i)


    # perform factor decomp for dual problem
    Sigma = X_train.T @ X_train
    D, L, _ = factor_decomp(Sigma, rank=rank)
    Q = D + L @ L.T
    
    Xtilde = scipy.linalg.sqrtm(Q)
    ytilde = scipy.linalg.solve(Xtilde.T, X_train.T @ y_train)
    
    assert np.linalg.norm(Xtilde.T @ ytilde - X_train.T @ y_train) <= 1e-10
    
    # sparse model solutions
    max_k = 50
    supp_du, cost_du = sm.cvx_dual(D, L, -X_train.T @ y_train, 
                                  list(range(1,max_k)),
                                  verbose=False)
    supp_l1  = sm.lasso_solve(X_train, y_train, max_k)
    supp_l1f  = sm.lasso_solve(Xtilde, ytilde, max_k)


    # primalize based on support recovered
    _, w_du = sm.primalize(Sigma, -X_train.T @ y_train, supp_du)
    _, w_l1 = sm.primalize(Sigma, -X_train.T @ y_train, supp_l1)
    _, w_l1f = sm.primalize(Sigma, -X_train.T @ y_train, supp_l1f)
    

    
    # calculate train cost
    err = lambda X, y, w: 1/(2*n) * np.linalg.norm(X @ w - y) ** 2 
    train_du.append([err(X_train, y_train, w) for w in w_du])
    train_l1.append([err(X_train, y_train, w) for w in w_l1])
    train_l1f.append([err(X_train, y_train, w) for w in w_l1f])


    test_du.append([err(X_test, y_test, w) for w in w_du])
    test_l1.append([err(X_test, y_test, w) for w in w_l1])
    test_l1f.append([err(X_test, y_test, w) for w in w_l1f])


#     plt.figure()
#     plt.subplot(2,1,1)
#     plt.title('Train')
#     plt.plot(train_du[0],'r-')
#     plt.plot(train_l1[0],'b-')
#     plt.plot(train_l1f[0],'k-')
#     plt.legend(['Dual', 'Lasso primalize', 'Lasso factor primalize'])
#     plt.subplot(2,1,2)
#     plt.title('Test')
#     plt.plot(test_du[0],'r-')
#     plt.plot(test_l1[0],'b-')
#     plt.plot(test_l1f[0],'k-')
#     plt.legend(['Dual', 'Lasso primalize', 'Lasso factor primalize'])
#     plt.xlabel('Sparsity (k)')
#     plt.ylabel('MSE')
#     plt.show()
    
    
#     import pdb; pdb.set_trace()


## save data
train_res = {'du': train_du, 'l1': train_l1, 'l1f': train_l1f}
test_res = {'du': test_du, 'l1': test_l1, 'l1f': test_l1f}
res = {'train': train_res, 'test': test_res}
filename = '../results/leukemia_' + str(rank) + '.pkl'
with open(filename, 'wb') as handle:
    pickle.dump(res, handle)

