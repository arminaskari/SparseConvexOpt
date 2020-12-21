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
    - TO DO
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
            assert l0term <= self.lmbda

        # compute objective function value
        beta = np.diag(u) @ v
        n = self.X.shape[0]
        if self.loss == 'l2':
            obj = 1 / (2 * n) * np.linalg.norm(self.X @ beta - self.y, 2)**2
        elif self.loss == 'hinge':
            obj = 1 / n * np.sum(np.maximum(0, 1 - self.y * (self.X @ beta)))
        elif self.loss == 'logistic':
            obj = 1 / n * np.sum(np.log(1 + np.exp(-self.y * (self.X @ beta))))

        return obj + l2cost + l0cost

    def fit(self, X, y, rank='full'):
        self.X, self.y = X, y
        if rank != 'full':
            assert isinstance(rank, int)
            self.rank = np.minimum(self.rank, np.min(self.X.shape))
        self._sparse_fit()

    def dual_objective(params, var, method='constrained', l2constr=False):
        """ Return dual objective for cvx for generic f """

        # unpack parameters
        X = params['X']
        r = params['rank']
        gamma, y = params['gamma'], params['y']
        loss = params['loss']

        # create cvx variables and objective
        if l2constr == False:
            l = var
            if method == 'constrained':
                k = params['target_sparsity']
                cost = -1 / (2 * gamma) * cp.sum_largest((X.T @ l)**2, k)
            elif method == 'penalized':
                cost = cp.sum(
                    cp.minimum(
                        np.zeros(len(l)),
                        params['lmbda'] - 1 / (2 * gamma) * (X.T @ l)**2))
        else:
            n = len(y)
            l = var[:n]
            phi = var[n:-1]
            eta = var[-1]

            if method == 'constrained':
                k = params['target_sparsity']
                cost = -1 / 2 * cp.sum_largest(phi, k) - eta * gamma / 2
            elif method == 'penalized':
                cost = cp.sum(
                    cp.minimum(
                        np.zeros(len(z)),
                        params['lmbda'] - 1 / 2 * cp.quad_over_lin(
                            (X.T @ lmbda), eta)))

        # add on the fenchel conjugate term
        if loss == 'l2':
            cost += -1 / 2 * cp.sum_squares(l) - l.T @ y
        else:
            raise NotImplementedError

        return cost

    def dual_problem(params, method='constrained', l2constr=False):
        """ Solve dual problem based on fenchel conjugate """

        # create dual problem
        constr = []
        n = len(params['y'])
        if l2constr:
            var = cp.Variable(2 * n + 1)  #[zeta, phi, eta]
            l = var[:n]
            phi = var[n:-1]
            eta = var[-1]
            constr.append(eta >= 0)
            for i in range(n):
                constr.append(
                    phi[i] >= cp.quad_over_lin((params['X'].T @ l)[i], eta))
        else:
            var = cp.Variable(n)
        cost = dual_objective(params, var, method=method, l2constr=l2constr)
        prob = cp.Problem(cp.Maximize(cost), constr)
        prob.solve(solver=cp.ECOS)

        return cost.value, var.value

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
            cost = 1 / (2 * n) * cp.sum_squares(X @ vp - y)
        else:
            raise NotImplementedError

        # if l2reg is constr, add constr, else add l2cost
        if self.l2reg == 'constr':
            constr.append(cp.sum(tp) <= gamma)
        else:
            cost += gamma / 2 * cp.sum(tp)

        # modify bidual based on if constrained or penalized primal
        if self.l0reg == 'constr':
            constr.append(cp.sum(up) <= self.lmbda)
        else:
            cost += self.lmbda * cp.sum(up)

        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.MOSEK)

        # primal variables via non-convex reformulation of bidual
        u = up.value
        v = np.linalg.pinv(np.diag(u)) @ vp.value

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
            constr.append(lmbda * cp.sum(u_lp) == t - tt)
        elif self.l2reg == 'pen' and self.l0reg == 'constr':
            constr.append(self.gamma / 2 * u_lp.T @ (v**2) == t - tt)
            constr.append(cp.sum(u_lp) <= self.lmbda)
        elif self.l2reg == 'pen' and self.l0reg == 'pen':
            constr.append(self.gamma / 2 * u_lp.T @ (v**2) +
                          self.lmbda * cp.sum(u_lp) == t - tt)
        
        # set random cost and solve lp
        cost = np.random.rand(m).T @ u_lp
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.ECOS, verbose=True, feastol=1e-7, abstol=1e-4)

        return u_lp.value

    def _sparse_fit(self):
        """ Solve the bidual and then primalize solution """

        # sovle bidual and use self.rank + SVD to make z
        u, v = self._bidual()
        U, s, Vh = scipy.linalg.svd(self.X)
        self.U = U[:, :self.rank]
        self.S = np.diag(s[:self.rank])
        self.V = Vh[:self.rank, :].T
        z = self.S @ self.V.T @ np.diag(u) @ v

        # shapley folkman primalization via LP
        ubar = self._primalization_lp(u, v, z)

        # primalize
        S, Sc = [], []
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

        # return cost of OPT
        if l2constr:
            if params['loss'] == 'l2':
                cost = 1 / 2 * np.linalg.norm(X @ np.diag(utilde) @ vtilde -
                                              y)**2
            else:
                raise NotImplementedError

        else:
            if params['loss'] == 'l2':
                cost = 1 / 2 * np.linalg.norm(
                    X @ np.diag(utilde) @ vtilde -
                    y)**2 + gamma / 2 * vtilde.T @ np.diag(utilde) @ vtilde
            else:
                raise NotImplementedError

        if method == 'constrained':
            return cost
        else:
            return cost + params['lmbda'] * np.sum(utilde)
