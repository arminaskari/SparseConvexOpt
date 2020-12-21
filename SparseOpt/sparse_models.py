import itertools
import numpy as np
import cvxpy as cp
import scipy
from sklearn.linear_model import LassoLars, lars_path
from scipy.optimize import minimize
from scipy.optimize import linprog


def sum_top_k(x, k):
    """ Sum of top k entries of vector x """
    idx = x.argsort()[-k:][::-1]
    return sum(x[idx]), idx


def kbits(n, k):
    """ Returns all sets of n choose k"""
    result = []
    for bits in itertools.combinations(range(n), k):
        s = [0] * n
        for bit in bits:
            s[bit] = 1
        result.append(s)
    return result


def l0_constrained(params, k, l2constr=False):
    """ Solve l0 constrained problem """

    # unpack parameters
    if params['loss'] == 'l2':

        m = len(params['beta'])
        K = params['X'].T @ params['X'] + params['gamma'] * np.eye(m)
        p = params['X'].T @ params['y']

        # generate all possible supports
        supports = kbits(m, k)
        costs = []
        assert m <= 20

        # enumerate over all sets
        for supp in supports:
            idx = np.where(np.array(supp) == 1)[0]

            if l2constr == False:
                Ktilde = K[idx, :][:, idx]
                ptilde = p[idx]

                iK = np.linalg.inv(Ktilde)
                costs.append(-1 / 2 * ptilde.T @ iK @ ptilde +
                             1 / 2 * np.linalg.norm(params['y'])**2)
            else:
                w = cp.Variable(m)
                constr = [w[idx] == 0, cp.sum_squares(w) <= params['gamma']]
                cost = cp.sum_squares(params['X'] @ w - params['y'])
                prob = cp.Problem(cp.Minimize(cost), constr)
                prob.solve(solver=cp.MOSEK)
                costs.append(cost.value)

        # get minimum cost solution
        if len(costs) == 0:
            import pdb
            pdb.set_trace()
        idx_min = np.argmin(costs)
        one_hot = np.array(supports[idx_min])
        supp = np.where(one_hot != 0)[0]
        cost = costs[idx_min]

    return supp, cost


def l0_penalized(lmbda, supp, cost):
    """ Given penalty parameter lmbda, and the optimal
    cost for each support, determine the new penalized
    cost """

    pen_costs = [c + lmbda * len(s) for c, s in zip(cost, supp)]
    return np.min(pen_costs)


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
        var = cp.Variable(2*n + 1) #[zeta, phi, eta]
        l = var[:n]
        phi = var[n:-1]
        eta = var[-1]
        constr.append(eta >= 0)
        for i in range(n):
            constr.append(phi[i] >= cp.quad_over_lin((params['X'].T @ l)[i], eta))
    else:
        var = cp.Variable(n)
    cost = dual_objective(params, var, method=method, l2constr=l2constr)
    prob = cp.Problem(cp.Maximize(cost), constr)
    prob.solve(solver=cp.ECOS)

    return cost.value, var.value


def bidual_problem(params, method='constrained', l2constr=False):
    """ Solve bidual problem """

    # unpack parameters
    X, y = params['X'], params['y']
    U, S, V = params['U'], params['S'], params['V']
    gamma = params['gamma']
    m = X.shape[1]

    # bidual
    vp = cp.Variable(m)
    up = cp.Variable(m)
    tp = cp.Variable(m)

    constr = [0 <= up, up <= 1]

    if params['loss'] == 'l2':
        cost = 1 / 2 * cp.sum_squares(X @ vp - y)
    else:
        raise ValueError

    # if problem is l2 constrained, add constr, else add l2 into
    # objective
    if l2constr:
        constr.append(cp.sum(tp) <= gamma)
    else:
        cost += gamma / 2 * cp.sum(tp)

    for i in range(m):
        constr.append(tp[i] >= cp.quad_over_lin(vp[i], up[i]))

    # modify bidual based on if constrained or penalized primal
    if method == 'constrained':
        k = params['target_sparsity']
        constr.append(cp.sum(up) <= k)
    elif method == 'penalized':
        lmbda = params['lmbda']
        cost += lmbda * cp.sum(up)
    else:
        raise ValueError

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.MOSEK)

    # non-convex formulation of bidual optimal variables
    u = up.value
    v = np.linalg.pinv(np.diag(u)) @ vp.value
    z = S @ V.T @ np.diag(u) @ v
    bdcost = 1 / 2 * np.linalg.norm(X @ np.diag(u) @ v -
                                    y)**2
    if not l2constr:
        bdcost += gamma / 2 * v.T @ np.diag(u) @ v


    if method == 'constrained':
        return bdcost, v, z, u
    else:
        return bdcost + lmbda * np.sum(u), v, z, u


"""
def primal_lp_simplex(params, v, z):
    # Solve lp for primalization via simplex 
    ## NOTE: this has only been coded for the constrained
    # case, not the penalized case. This code should not be used
    # anyways since the IPM solver is much more reliable

    # unpack params
    U, S, V = params['U'], params['S'], params['V']
    gamma = params['gamma']
    y, t = params['y'], params['bidual_cost']
    m, k = V.shape[0], params['target_sparsity']

    # construct and sovle LP via simplex
    Aeq = np.vstack((gamma / 2 * (v**2), S @ V.T * v))
    if params['loss'] == 'l2':
        beq = np.hstack((t - 1 / 2 * np.linalg.norm(U @ z - y)**2, z))
    else:
        raise NotImplementedError

    Aineq = (np.ones(m)).reshape((1, m))
    bineq = np.array([k])
    c = np.random.randn(m)
    bnds = list(((0, 1), ) * m)
    res = linprog(c,
                  A_ub=Aineq,
                  b_ub=bineq,
                  A_eq=Aeq,
                  b_eq=beq,
                  bounds=bnds,
                  method='revised simplex',
                  options={
                      'tol': 1e-4,
                      'autoscale': True
                  })

    if res.success:
        return res.x
    else:
        print(res)
        raise ValueError
"""


def primal_lp_ipm(params, v, z, method='constrained', l2constr=False):
    """ solve LP primalization via IPM """

    # unpack params
    U, S, V = params['U'], params['S'], params['V']
    gamma = params['gamma']
    y, t = params['y'], params['bidual_cost']
    m = V.shape[0]

    # solve lp
    u = cp.Variable(m)
    constr = [0 <= u, u <= 1]
    constr.append(S @ V.T @ cp.diag(u) @ v == z)

    # modify bidual based on if constrained or penalized primal
    if l2constr:
        # add norm constr
        constr.append(cp.sum(cp.multiply(u, v**2)) <= gamma)
        if method == 'constrained':
            k = params['target_sparsity']
            constr.append(cp.sum(u) <= k)

            # add epigraph constraint
            if params['loss'] == 'l2':
                pass
            else:
                raise ValueError
        elif method == 'penalized':
            lmbda = params['lmbda']

            # add epigraph constraint
            if params['loss'] == 'l2':
                constr.append(lmbda * cp.sum(u) == t -
                              1 / 2 * np.linalg.norm(U @ z - y)**2)
            else:
                raise ValueError
    else:
        if method == 'constrained':
            k = params['target_sparsity']
            constr.append(cp.sum(u) <= k)

            # add epigraph constraint
            if params['loss'] == 'l2':
                 constr.append(gamma / 2 * u.T @ (v**2) == t -
                              1 / 2 * np.linalg.norm(U @ z - y)**2)
            else:
                raise ValueError

        elif method == 'penalized':
            lmbda = params['lmbda']

            # add epigraph constraint
            if params['loss'] == 'l2':
                constr.append(lmbda * cp.sum(u) +
                              gamma / 2 * u.T @ (v**2) == t -
                              1 / 2 * np.linalg.norm(U @ z - y)**2)
            else:
                raise ValueError

    cost = np.random.rand(m).T @ u
    # cost = cp.sum(u)
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.ECOS, verbose=True, feastol=1e-7, abstol=1e-4)

    return u.value


def sf_primalize(params, method='constrained', l2constr=False):
    """ primalize solution from bidual """

    # unpack parameters
    X, y = params['X'], params['y']
    m, gamma = X.shape[1], params['gamma']

    # bidual
    bdcost, v, z, _ = bidual_problem(params, method=method, l2constr=l2constr)
    params['bidual_cost'] = bdcost

    # shapley folkman primalization via LP
    # ubar = primal_lp_simplex(params, v, z)
    ubar = primal_lp_ipm(params, v, z, method=method, l2constr=l2constr)

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
            cost = 1 / 2 * np.linalg.norm(X @ np.diag(utilde) @ vtilde - y)**2
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
