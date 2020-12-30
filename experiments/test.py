import sys
sys.path.append('../SparseOpt/')
from SparseModel import SparseModel
from helper import generate_data, lars_solve
import pickle
import numpy as np

# init
np.random.seed(1)
ntest = 500
n, m, r_eff = 1000, 200, 10
loss = 'l2'
true_sparsity = 0.05
lstar = int(m * true_sparsity)

params = generate_data(n,
                       m,
                       r_eff,
                       loss=loss,
                       ntest=ntest,
                       true_sparsity=true_sparsity,
                       low_rank='soft')

clf = SparseModel(loss=loss,
                  l0reg='constr',
                  l2reg='constr',
                  lmbda=lstar,
                  gamma=2 * np.linalg.norm(params['beta'])**2)
clf.fit(params['X'], params['y'], rank=r_eff)
l0coef_, l0cost = clf._l0fit(params['X'], params['y'], k=lstar)
l1coef_ = lars_solve(params['X'], params['y'], k=lstar)
## NOTE: should also solve primal in closed form and see what the loss is
print(clf.coef_)
print(l1coef_)
print(l0coef_)
print(params['beta'])
print('cost of OPT: ' + str(clf.fval(np.ones(m), clf.coef_)))
print('l1 cost: ' + str(1/(2*n) * np.linalg.norm(params['X'] @ l1coef_ - params['y']) ** 2))
print('l0 closed form cost: ' + str(l0cost))
print('cost of ground truth: ' + str(clf.fval(np.ones(m), params['beta'])))


ff = lambda w: 1/(2*ntest) * np.linalg.norm(params['Xtest'] @ w - params['ytest']) ** 2

print('-----------------------------------------')
print('test cost OPT: ' + str(ff(clf.coef_)))
print('test cost l1: ' + str(ff(l1coef_)))
print('l0 closed form test cost: ' + str(ff(l0coef_)))
print('test cost ground truth: ' + str(ff(params['beta'])))
