import itertools
import numpy as np
import cvxpy as cp
import scipy
from sklearn.base import BaseEstimator
from scipy.optimize import minimize


class SparseModel(BaseEstimator):
    """ SparseModel
    Parameters:
    ----------
    loss : loss function ('l2', 'hinge', or 'logistic')
    l0reg: 'constr' or 'pen' for constr/penalized l0 loss
    lmbda: value of l0reg
    l2reg: 'constr' or 'pen' for constr/penalized l2 loss
    gamma: value of l2reg

    """
    def __init__(self,
                 loss='l2',
                 l0reg='constr',
                 lmbda=1,
                 l2reg='constr',
                 gamma=1):
        self.loss = loss
        self.l0reg = l0reg
        self.lmbda = lmbda
        self.l2reg = l2reg
        self.gamma = gamma
        self.rank = 'full'

        assert (loss in ['l2', 'hinge', 'logistic'])
        assert (l0reg in ['constr', 'pen'])
        assert (l2reg in ['constr', 'pen'])
        assert lmbda > 0
        assert gamma > 0

    def _fcvx(self, X, y, w):
        """ Returns cost function for cvx problem """
        
        n = X.shape[0]

        # add cost based on l2reg
        if self.l2reg == 'constr':
            l2cost = 0
        else:
            l2cost = self.gamma/2 * cp.sum_squares(w)

        # compute objectives
        if self.loss == 'l2':
            return 1/(2*n) * cp.sum_squares(X @ w - y) + l2cost
        else:
            raise NotImplementedError

    def _constrcvx(self, w):
        """ Returns l2 constr for cvx problem """
        
        if self.l2reg == 'constr':
           return cp.sum_squares(w) <= self.gamma
        return 

    def fval(self, u, v):
        """ Returns objective function value for primal variables u, v """

        # if l2reg is pen, add to objective, else check feasibility
        l2term = v.T @ np.diag(u) @ v
        if self.l2reg == 'pen':
            l2cost = self.gamma * l2term
        else:
            l2cost = 0
            assert l2term <= self.gamma

        # if l0reg is pen, add to objective, else check feasibility
        l0term = np.sum(u)
        if self.l0reg == 'pen':
            l0cost = self.lmbda * l0term
        else:
            l0cost = 0
            # assert l0term <= self.lmbda

        # compute objective function value
        w = np.diag(u) @ v
        n = self.X.shape[0]
        if self.loss == 'l2':
            obj = 1 / (2 * n) * np.linalg.norm(self.X @ w - self.y, 2)**2
        elif self.loss == 'hinge':
            obj = 1 / n * np.sum(np.maximum(0, 1 - self.y * (self.X @ w)))
        elif self.loss == 'logistic':
            obj = 1 / n * np.sum(np.log(1 + np.exp(-self.y * (self.X @ w))))

        return obj + l2cost + l0cost

    def fit(self, X, y, rank='full'):

        self.X, self.y = X, y
        U, s, Vh = scipy.linalg.svd(self.X)
        self.U = U
        self.S = np.diag(s)
        self.V = Vh.T
        
        if rank != 'full':
            assert isinstance(rank, int)
            self.rank = np.minimum(rank, np.min(self.X.shape))

            self.U = U[:, :self.rank]
            self.S = np.diag(s[:self.rank])
            self.V = Vh[:self.rank, :].T

            self.X = self.U @ self.S @ self.V.T

        # solve bidual and perform primalization
        utilde, vtilde = self._sparse_fit()
        self.coef_ = utilde * vtilde
        print(f'cost: {self.fval(utilde,vtilde)}')

    def predict(self, X):

        if self.loss == 'l2':
            return X @ self.coef_
        elif self.loss == 'hinge':
            return np.sign(X @ self.coef_)
        elif self.loss == 'logistic':
            return np.log(1 + np.exp(-self.X @ self.coef_))

    def _dual_problem(self):
        """ Solve dual problem based on fenchel conjugate """

        n, m = self.X.shape[0], self.X.shape[1]
        constr = []

        # create cost and constr of dual problem
        if self.l2reg == 'pen':
            zeta = cp.Variable(n)
            if self.l0reg == 'constr':
                cost = -1 / (2 * self.gamma) * cp.sum_largest(
                    (self.X.T @ zeta)**2, self.lmbda)
            elif self.l0reg == 'pen':
                cost = cp.sum(
                    cp.minimum(
                        np.zeros(n), self.lmbda - 1 / (2 * self.gamma) *
                        (self.X.T @ zeta)**2))
        elif self.l2reg == 'constr':
            # cant handle (X.T @ zeta)/eta directly so use slack variable (t)
            zeta = cp.Variable(n)
            eta = cp.Variable(1)
            t = cp.Variable(m)
            for i in range(m):
                constr.append(
                    t[i] >= cp.quad_over_lin((self.X.T @ zeta)[i], eta))

            if self.l0reg == 'constr':
                cost = -1 / 2 * cp.sum_largest(
                    t, self.lmbda) - eta * self.gamma / 2
            elif self.l0reg == 'pen':
                cost = cp.sum(cp.minimum(np.zeros(n), self.lmbda - 1 / 2 * t))

        # add on fenchel conjugate term
        if self.loss == 'l2':
            cost += -n / 2 * cp.sum_squares(zeta) - zeta.T @ self.y
        else:
            raise NotImplementedError

        # solve
        prob = cp.Problem(cp.Maximize(cost), constr)
        prob.solve(solver=cp.MOSEK)

        return cost.value, zeta.value

    def _bidual_problem(self):
        """ Solve bidual problem """

        m, n = self.X.shape[1], self.X.shape[0]

        # bidual
        vp = cp.Variable(m)
        up = cp.Variable(m)
        tp = cp.Variable(m)

        constr = [0 <= up, up <= 1]
        # socp reformulation of l2 term
        for i in range(m):
            constr.append(tp[i] >= cp.quad_over_lin(vp[i], up[i]))

        ### construct cost and add objectives based on l0reg, l2reg
        if self.loss == 'l2':
            cost = 1 / (2 * n) * cp.sum_squares(self.X @ vp - self.y)
        else:
            raise NotImplementedError

        # update cost/constr based on l2reg/l0reg
        if self.l2reg == 'constr':
            constr.append(cp.sum(tp) <= self.gamma)
        elif self.l2reg == 'pen':
            cost += self.gamma / 2 * cp.sum(tp)
        if self.l0reg == 'constr':
            constr.append(cp.sum(up) <= self.lmbda)
        elif self.l0reg == 'pen':
            cost += self.lmbda * cp.sum(up)

        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.MOSEK)

        # primal variables via non-convex reformulation of bidual
        u = up.value
        v = np.linalg.pinv(np.diag(u)) @ vp.value
        
        #print(self._dual_problem()[0])
        #print(self.fval(u,v))
        return u, v

    def _primalization_lp(self, u, v, z):
        """ solve LP primalization via IPM """

        m, n = self.X.shape[1], self.X.shape[0]
        t = self.fval(u, v)

        # construct lp
        u_lp = cp.Variable(m)
        constr = [0 <= u_lp, u_lp <= 1]
        constr.append(self.S @ self.V.T @ cp.diag(u_lp) @ v == z)

        # modify bidual based on l0/l2reg
        if self.loss == 'l2':
            tt = 1 / (2 * n) * np.linalg.norm(self.U @ z - self.y)**2
        else:
            raise NotImplementedError

        if self.l2reg == 'constr' and self.l0reg == 'constr':
            constr.append(cp.sum(cp.multiply(u_lp, v**2)) <= self.gamma)
            constr.append(cp.sum(u_lp) <= self.lmbda)
        elif self.l2reg == 'constr' and self.l0reg == 'pen':
            constr.append(cp.sum(cp.multiply(u_lp, v**2)) <= self.gamma)
            constr.append(self.lmbda * cp.sum(u_lp) == t - tt)
        elif self.l2reg == 'pen' and self.l0reg == 'constr':
            constr.append(self.gamma / 2 * u_lp.T @ (v**2) == t - tt)
            constr.append(cp.sum(u_lp) <= self.lmbda)
        elif self.l2reg == 'pen' and self.l0reg == 'pen':
            constr.append(self.gamma / 2 * u_lp.T @ (v**2) +
                          self.lmbda * cp.sum(u_lp) == t - tt)

        # set random cost and solve lp
        cost = np.random.rand(m).T @ u_lp
        cost = cp.norm(u_lp, 1)
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.ECOS, feastol=1e-9, abstol=1e-6)

        return u_lp.value

    def _sparse_fit(self):
        """ Solve the bidual and then primalize solution """

        # sovle bidual and use self.rank + SVD to make z
        u, v = self._bidual_problem() 
        z = self.S @ self.V.T @ np.diag(u) @ v

        # shapley folkman primalization via LP
        ubar = self._primalization_lp(u, v, z)
        ubar[np.where(np.abs(ubar) <= 1e-7)[0]] = 0
        #print(f'ubar: {ubar}')

        # primalize
        S, Sc = [], []
        m = len(ubar)
        for i in range(m):
            if np.abs(ubar[i]) <= 1e-7 or np.abs(ubar[i] - 1) <= 1e-7:
                Sc.append(i)
            else:
                S.append(i)

        vtilde = v
        utilde = ubar
        for i in range(m):
            if i in S:
                vtilde[i] = ubar[i] * v[i]
                utilde[i] = 1

        return utilde, vtilde
    
    @staticmethod
    def _kbits(n, k):
        """ Returns all sets of n choose k"""
        result = []
        for bits in itertools.combinations(range(n), k):
            s = [0] * n
            for bit in bits:
                s[bit] = 1
            result.append(s)
        return result

    def l0fit(self, X, y, lmbda):
        """ Brute force enumreate over all supports and find min cost """

        m = X.shape[0]
        costs, wstar = [], []
        for k in range(1,m+1):
            w, cost = self._l0fit(self, X, y, k)
            wstar.append(w)
            costs.append(cost)

        return


    def _l0fit(self, X, y, k):
        """ Solve original non-convex l0 constr problem for given k"""

        m = X.shape[1]
        if m > 20:
            return np.zeros(m), 0
        supports = self._kbits(m, k)
        costs, wstar = [], []
        
        # enumerate over all possibilities
        for supp in supports:
            idx = np.where(np.array(supp) == 0)[0]
            
            w = cp.Variable(m)
            constr = [w[idx] == 0, self._constrcvx(w)]
            cost = self._fcvx(X, y, w) 
            prob = cp.Problem(cp.Minimize(cost), constr)
            prob.solve(solver=cp.MOSEK)
            costs.append(cost.value)
            wstar.append(w.value)
        
        idx_min = np.argmin(costs)
        one_hot = np.array(supports[idx_min])
        supp = np.where(one_hot != 0)[0]
        cost = costs[idx_min]
        wast = wstar[idx_min]

        return wast, cost

