"""
This module sets up and solves optimization problem described in the paper. 
The module solves the bidual problem and performs the primalization procedure.

"""

import itertools

import cvxpy as cp
import numpy as np
import scipy
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

from .helper import sigmoid


GLOBAL_ZERO_TOL = 5e-5  # tolerance for how small to consider number 0

# documentation on what the mosek tolerances are changing
#   https://docs.mosek.com/9.1/pythonapi/parameters.html#doc-dparam-list
#   https://docs.mosek.com/9.1/pythonapi/solving-conic.html#adjusting-optimality-criteria
MOSEK_PARAMS = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12}
MOSEK_PARAMS_LIGHT = {
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-9,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
}


class SparseModel(BaseEstimator):
    """SparseModel
    Parameters:
    ----------
    loss : loss function ('l2', 'hinge', or 'logistic')
    l0reg: 'constr' or 'pen' for constr/penalized l0 loss
    lmbda: value of l0reg
    l2reg: 'constr' or 'pen' for constr/penalized l2 loss
    gamma: value of l2reg

    """

    def __init__(
        self,
        loss="l2",
        l0reg="constr",
        lmbda=1,
        l2reg="constr",
        gamma=1,
        n_trials=20,
        threads=1,
    ):
        self.loss = loss
        self.l0reg = l0reg
        self.lmbda = lmbda
        self.l2reg = l2reg
        self.gamma = gamma
        self.rank = "full"
        self.n_trials = n_trials

        assert loss in ["l2", "hinge", "logistic"]
        assert l0reg in ["constr", "pen"]
        assert l2reg in ["constr", "pen"]
        assert lmbda > 0
        assert gamma > 0

    def _obj_cvx(self, X, y, w):
        """
        Returns cost function for cvx problem

        Input:
            X (np.ndarray): n x m array
            y (np.ndarray): (n, ) array
            w (np.ndarray): (m, ) array

        Output:
            cvxpy.problems.objective
        """

        n = X.shape[0]

        # add cost based on l2reg
        if self.l2reg == "constr":
            l2cost = 0
        else:
            l2cost = self.gamma / 2 * cp.sum_squares(w)

        # compute objectives
        if self.loss == "l2":
            return 1 / (2 * n) * cp.sum_squares(X @ w - y) + l2cost
        if self.loss == "logistic":
            return 1 / n * cp.sum(cp.logistic(-cp.multiply(y, X @ w))) + l2cost
        else:
            raise NotImplementedError("loss function is not implemented")

    def _constr_cvx(self, w):
        """Returns l2 constr for cvx problem"""

        if self.l2reg == "constr":
            return cp.sum_squares(w) <= self.gamma
        return

    def obj_val(self, u, v, check_feasibility=False):
        """Returns objective function value for primal variables u, v"""

        Du = np.diag(u)

        # if l2reg is pen, add to objective, else check feasibility
        l2term = v.T @ Du @ v
        if self.l2reg == "pen":
            l2cost = self.gamma / 2 * l2term
        else:
            l2cost = 0
            if check_feasibility:
                # some slack due to inaccuracy of solvers
                assert l2term <= self.gamma * (1 + 1e-5)

        # if l0reg is pen, add to objective, else check feasibility
        l0term = np.sum(u)
        if self.l0reg == "pen":
            l0cost = self.lmbda * l0term
        else:
            l0cost = 0
            if check_feasibility:
                assert l0term <= self.lmbda

        # compute objective function value
        w = Du @ v
        n = self.X.shape[0]
        if self.loss == "l2":
            obj = 1 / (2 * n) * np.linalg.norm(self.X @ w - self.y, 2) ** 2
        elif self.loss == "hinge":
            obj = 1 / n * np.sum(np.maximum(0, 1 - self.y * (self.X @ w)))
        elif self.loss == "logistic":
            obj = 1 / n * np.sum(np.log(1 + np.exp(-self.y * (self.X @ w))))

        return obj + l2cost + l0cost

    def fit(self, X, y, rank="full"):
        """
        solve bidual and perform primalization

        Input:
            X (np.ndarray or dict<str>): either n x m array or dict
                with keys "U", "S", "V" which constitute an SVD decomposition
                of X
            y (np.ndarray): (n, ) array
            rank (str or int): either str "full" in which case we keep X
                to be the full matrix and do not perform any rank truncation
                or an int specifying a rank-r approximation to take from X
                (i.e. a rank-r SVD of X)
        Output:
            None
        """

        self.y = y
        # if SVD of X is passed in, compute X from U, S, V
        if isinstance(X, dict):
            self.U = X["U"]
            self.S = X["S"]
            self.V = X["V"]

            self.X = self.U @ self.S @ self.V.T
        # perform SVD of X to compute U, S, V
        else:
            U, s, Vh = scipy.linalg.svd(X)
            self.U = U
            self.S = np.diag(s)
            self.V = Vh.T
            # if rank is full, take X to be the entire matrix
            if rank == "full":
                self.X = X
            # if rank is not full and is an integer, then truncate
            # the SVD of X and take a rank-r SVD of X instead
            if rank != "full":
                assert isinstance(rank, int)
                self.rank = np.minimum(rank, np.min(X.shape))

                self.U = U[:, : self.rank]
                self.S = np.diag(s[: self.rank])
                self.V = Vh[: self.rank, :].T

                self.X = self.U @ self.S @ self.V.T

        # solve bidual and perform primalization
        self._sparse_fit()

    def predict(self, X):
        """perform prediction on X based on computed self.coef_"""

        if self.loss == "l2":
            return X @ self.coef_
        elif self.loss == "hinge":
            return np.sign(X @ self.coef_)
        elif self.loss == "logistic":
            return 2 * sigmoid(self.X @ self.coef_) - 1

    def dual_problem(self, X):
        """Solve dual problem based on fenchel conjugate"""

        n, m = X.shape[0], X.shape[1]
        constr = []
        zeta = cp.Variable(n)

        # create cost and constr of dual problem
        if self.l2reg == "pen":
            if self.l0reg == "constr":
                cost = (
                    -1
                    / (2 * self.gamma)
                    * cp.sum_largest((X.T @ zeta) ** 2, self.lmbda)
                )
            elif self.l0reg == "pen":
                cost = cp.sum(
                    cp.minimum(
                        np.zeros(n),
                        self.lmbda - 1 / (2 * self.gamma) * (X.T @ zeta) ** 2,
                    )
                )
        elif self.l2reg == "constr":
            # cant handle (X.T @ zeta)/eta directly so use slack variable (t)
            eta = cp.Variable(1)
            t = cp.Variable(m)
            for i in range(m):
                constr.append(t[i] >= cp.quad_over_lin((X.T @ zeta)[i], eta))

            if self.l0reg == "constr":
                cost = -1 / 2 * cp.sum_largest(t, self.lmbda) - eta * self.gamma / 2
            elif self.l0reg == "pen":
                cost = (
                    cp.sum(cp.minimum(np.zeros(m), self.lmbda - 1 / 2 * t))
                    - eta * self.gamma / 2
                )

        # add on fenchel conjugate term
        if self.loss == "l2":
            cost += -n / 2 * cp.sum_squares(zeta) - zeta.T @ self.y
        elif self.loss == "logistic":
            # from https://web.stanford.edu/~boyd/papers/pdf/l1_logistic_reg_aaai.pdf
            yz = cp.multiply(self.y, zeta)
            cost += 1 / n * cp.sum(cp.entr(n * yz) + cp.entr(1 - n * yz))
            # constr.extend([0 <= zeta, zeta <= 1])
        else:
            raise NotImplementedError("dual problem not implemented in cvx")

        # solve
        prob = cp.Problem(cp.Maximize(cost), constr)
        prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS_LIGHT)

        # instability in cp.entr and zeta.value becoming slightly negative
        if (cost.value == -np.inf or cost.value == np.nan) and self.loss == "logistic":
            print("manually adjusting cost...")
            cost = (
                np.sum(
                    np.minimum(
                        np.zeros(m),
                        self.lmbda - 1 / 2 * (X.T @ zeta.value) ** 2 / eta.value,
                    )
                )
                - eta.value * self.gamma / 2
            )
            yz = self.y * zeta.value
            idx = np.where(yz > 0)[0]
            nyz = (n * yz)[idx]
            cost += 1 / n * np.sum(-nyz * np.log(nyz) - (1 - nyz) * np.log(1 - nyz))

            return cost, zeta.value

        return cost.value, zeta.value

    def _bidual_problem(self):
        """Solve bidual problem"""

        m, n = self.X.shape[1], self.X.shape[0]

        # bidual
        vp = cp.Variable(m)
        up = cp.Variable(m)
        tp = cp.Variable(m)

        constr = [up >= 0, up <= 1]
        # socp reformulation of l2 term
        for i in range(m):
            constr.append(tp[i] >= cp.quad_over_lin(vp[i], up[i]))

        # construct cost and add objectives based on l0reg, l2reg
        if self.loss == "l2":
            cost = 1 / (2 * n) * cp.sum_squares(self.X @ vp - self.y)
        elif self.loss == "logistic":
            cost = 1 / n * cp.sum(cp.logistic(-cp.multiply(self.y, self.X @ vp)))
        else:
            raise NotImplementedError("loss not implemented in cvx")

        # update cost/constr based on l2reg/l0reg
        if self.l2reg == "constr":
            constr.append(cp.sum(tp) <= self.gamma)
        elif self.l2reg == "pen":
            cost += self.gamma / 2 * cp.sum(tp)
        if self.l0reg == "constr":
            constr.append(cp.sum(up) <= self.lmbda)
        elif self.l0reg == "pen":
            cost += self.lmbda * cp.sum(up)

        prob = cp.Problem(cp.Minimize(cost), constr)
        if "MOSKEK" in cp.installed_solvers():
            prob.solve(solver=cp.MOSEK, mosek_params=MOSEK_PARAMS)
        else:
            prob.solve(solver=cp.ECOS, feastol=1e-12, abstol=1e-8)
        self.bdcost_ = cost.value

        # primal variables via non-convex reformulation of bidual/soln polish
        u = up.value
        u[np.where(u <= GLOBAL_ZERO_TOL)[0]] = 0
        v = np.linalg.pinv(np.diag(u)) @ vp.value
        self.bdcost_ = self.obj_val(u, v)
        return u, v

    def _primalization_lp(self, u, v, z):
        """solve LP primalization via IPM"""

        m, n = self.X.shape[1], self.X.shape[0]
        t = self.obj_val(u, v)
        assert np.abs(t - self.bdcost_) <= 1e-7

        self.slack_ = np.max((0, v.T @ (u * v) - self.gamma))
        # construct lp
        u_lp = cp.Variable(m)
        constr = [0 <= u_lp, u_lp <= 1]

        SVt = self.S @ self.V.T
        scales = np.array([np.linalg.norm(r) for r in np.multiply(SVt, v)])
        # scales = np.ones(len(scales))
        constr.append(
            cp.multiply(SVt @ cp.diag(u_lp) @ v, 1 / scales) == 1 / scales * z
        )

        # modify bidual based on l0/l2reg
        if self.loss == "l2":
            tt = 1 / (2 * n) * np.linalg.norm(self.U @ z - self.y) ** 2
        elif self.loss == "logistic":
            tt = 1 / n * np.sum(np.log(1 + np.exp(-self.y * (self.U @ z))))
        else:
            raise NotImplementedError

        vnorm = np.linalg.norm(v ** 2)
        if self.l2reg == "constr" and self.l0reg == "constr":
            constr.append(u_lp.T @ (v ** 2) / vnorm <= self.gamma / vnorm)
            constr.append(cp.sum(u_lp) <= self.lmbda)
        elif self.l2reg == "constr" and self.l0reg == "pen":
            constr.append(u_lp.T @ (v ** 2) / vnorm <= self.gamma / vnorm)
            constr.append(self.lmbda * cp.sum(u_lp) == t - tt)
        elif self.l2reg == "pen" and self.l0reg == "constr":
            constr.append(self.gamma / 2 * u_lp.T @ (v ** 2) == (t - tt))
            constr.append(cp.sum(u_lp) <= self.lmbda)
        elif self.l2reg == "pen" and self.l0reg == "pen":
            constr.append(
                (
                    self.gamma / 2 * u_lp.T @ (v ** 2) + self.lmbda * cp.sum(u_lp)
                    == (t - tt)
                )
            )

        # set random cost and solve lp
        cost = np.random.randn(m).T @ u_lp
        prob = cp.Problem(cp.Minimize(cost), constr)
        if "MOSEK" in cp.installed_solvers():
            prob.solve(solver=cp.MOSEK, bfs=True, mosek_params=MOSEK_PARAMS_LIGHT)
        else:
            prob.solve(solver=cp.ECOS, feastol=1e-8, abstol=1e-3)

        return u_lp.value

    def _sparse_fit(self):
        """Solve the bidual and then primalize solution"""

        # sovle bidual and use self.rank + SVD to make z
        u, v = self._bidual_problem()
        z = self.S @ self.V.T @ np.diag(u) @ v

        self.coefs_, self.costs_ = [], []
        # shapley folkman primalization via LP
        for i in range(self.n_trials):
            ubar = self._primalization_lp(u.copy(), v.copy(), z.copy())
            ubar[np.where(np.abs(ubar) <= GLOBAL_ZERO_TOL)[0]] = 0

            # primalize
            S, Sc = [], []
            m = len(ubar)
            for i in range(m):
                if (
                    np.abs(ubar[i]) <= GLOBAL_ZERO_TOL
                    or np.abs(ubar[i] - 1) <= GLOBAL_ZERO_TOL
                ):
                    Sc.append(i)
                else:
                    S.append(i)

            vtilde = v.copy()
            utilde = ubar
            for i in range(m):
                if i in S:
                    vtilde[i] = ubar[i] * v.copy()[i]
                    utilde[i] = 1

            self.coefs_.append(utilde * vtilde)
            self.costs_.append(self.obj_val(utilde, vtilde))

        # recover lowest cost solutions
        idx_min = np.argmin(self.costs_)
        self.coef_ = self.coefs_[idx_min]
        self.cost_ = self.costs_[idx_min]

    # everything below is unused
    @staticmethod
    def _kbits(n, k):
        """Returns all sets of n choose k"""
        result = []
        for bits in itertools.combinations(range(n), k):
            s = [0] * n
            for bit in bits:
                s[bit] = 1
            result.append(s)
        return result

    def l0fit(self, X, y, lmbda):
        """Brute force enumreate over all supports and find min cost"""

        m = X.shape[0]
        costs, wstar = [], []
        for k in range(1, m + 1):
            w, cost = self._l0fit(self, X, y, k)
            wstar.append(w)
            costs.append(cost)

        return

    def _l0fit(self, X, y, k):
        """Solve original non-convex l0 constr problem for given k"""

        m = X.shape[1]
        if m > 20:
            return np.zeros(m), 0
        supports = self._kbits(m, k)
        costs, wstar = [], []

        # enumerate over all possibilities
        for supp in supports:
            idx = np.where(np.array(supp) == 0)[0]

            w = cp.Variable(m)
            constr = [w[idx] == 0, self._constr_cvx(w)]
            cost = self._obj_cvx(X, y, w)
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
