"""

def primalize(Q, p, supp, beta=None):
    """ Get primal solution based on support """

    if not isinstance(supp, list):
        supp = [supp]

    m = Q.shape[0]
    soln_k, costs_k = [], []

    for s in supp:
        s.sort()
        Qs = Q[s, :][:, s]
        ps = p[s]

        if beta == None:
            # optimal soln/cost for given support
            xstar = np.zeros(m)

            ## should replace with minimum norm solution
            try:
                soln = -np.linalg.inv(Qs) @ ps
            except:
                raise ValueError
                soln = -np.linalg.inv(Qs + 0.01 * np.eye(len(s))) @ ps
            xstar[s] = soln
            cost = 1 / 2 * ps.T @ soln
        else:
            cost, xstar = cvx_qp_constr(Qs, ps, beta)

        soln_k.append(xstar)
        costs_k.append(cost)

    return costs_k, soln_k


def cvx_qp_constr(Q, p, beta, verbose=False):
    """ Solve quadratic program with infinity norm constraints """

    x = cp.Variable(len(p))
    cost = 1 / 2 * cp.quad_form(x, Q) + x.T @ p
    constr = [-beta <= x, x <= beta]
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver=cp.ECOS,
               verbose=verbose,
               abstol=abstol,
               reltol=reltol,
               feastol=feastol)

    return cost.value, x.value


def l0_penalized_atom(supp, cost, g):
    """ Given optimal support and cost and a given gamma for l0
    penalty, determine the optimal solution for penalized problem"""

    pen_costs = np.array([p + len(k) * g for k, p in zip(supp, cost)])
    idx = np.where(pen_costs == np.min(pen_costs))[0]

    if len(idx) > 1:
        raise ValueError

    return supp[idx[0]], pen_costs[idx[0]]


def l0_penalized(supp, cost, g):
    """ Compute optimla l0 penalized solution baseed on the constrained 
    version """
    if not isinstance(g, list):
        g = [g]

    supp_g, costs_g = [], []

    for gg in g:
        s, c = l0_penalized_atom(supp, cost, gg)
        costs_g.append(c)
        supp_g.append(s)

    return supp_g, costs_g


def l0_penalized_bidual_cvx(D, L, p, g, verbose=False, solver='ecos'):
    """ Bidual for l0 penalized problem """

    if not isinstance(g, list):
        g = [g]

    supp_k, costs_k = [], []

    for gg in g:
        m = len(p)
        l = L.shape[0]

        # cvx problem
        x = cp.Variable((m, 3))
        t = cp.Variable((m, 3))
        w_x = cp.Variable(m)
        w_t = cp.Variable(m)

        # constraints
        constr = []
        c = np.sum(np.divide(p**2, 2 * np.diag(D)))
        cost = 0.5 * cp.sum_squares(L.T @ cp.sum(x, 1)) + cp.sum(cp.sum(
            t, 1)) - 4 * gg * m + 0 * c
        for i in range(m):
            fac = np.sqrt(2 * gg / D[i, i])
            constr.extend([
                t[i, 0] >=
                1 / 2 * D[i, i] * cp.square(x[i, 0]) + p[i] * x[i, 0] + gg,
                x[i, 0] <= -fac,
                t[i, 1] >= p[i] * x[i, 1] + D[i, i] * cp.abs(x[i, 1]) * fac,
                -fac <= x[i, 1], x[i, 1] <= fac, t[i, 2] >=
                1 / 2 * D[i, i] * cp.square(x[i, 2]) + p[i] * x[i, 2] + gg,
                x[i, 2] >= fac
            ])
            #w_x[i] == x[i,0] + x[i,1] + x[i,2],
            #w_t[i] == t[i,0] + t[i,1] + t[i,2] - 4 * gg])

        prob = cp.Problem(cp.Minimize(cost), constr)
        if solver == 'scs':
            prob.solve(solver=cp.SCS, eps=scs_eps)
        elif solver == 'ecos':
            prob.solve(solver=cp.ECOS,
                       verbose=verbose,
                       abstol=abstol,
                       reltol=reltol,
                       feastol=feastol)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK, verbose=verbose)

        # recover support from alpha.value
        costs_k.append(cost.value)

    return costs_k


def l0_bruteforce(Q, p, k, beta=None, verbose=False):
    """ Brute Force solution to l0 constr QP """

    if not isinstance(k, list):
        k = [k]

    supp_k, costs_k, betas_k = [], [], []

    for kk in k:
        m = len(p)

        # will take too long
        if m > 20:
            return [], []

        supports = kbits(m, kk)
        costs = []
        betas = []
        counter = 0

        # enumerate over all sets
        for supp in supports:
            counter += 1
            if counter % 100000 == 0:
                print(counter)
            idx = np.where(np.array(supp) == 1)[0]
            Qtilde = Q[idx, :][:, idx]
            ptilde = p[idx]

            # if no infinity norm constraint
            if beta == None:
                M = np.linalg.inv(Qtilde)
                costs.append(-1 / 2 * ptilde.T @ M @ ptilde)
                betas.append(np.linalg.norm(-M @ ptilde, np.inf))
            else:
                cost, _ = cvx_qp_constr(Qtilde, ptilde, beta)
                costs.append(cost)
                betas.append(beta)

        # get minimum cost solution
        idx_min = np.argmin(costs)
        one_hot = np.array(supports[idx_min])
        supp_k.append(np.where(one_hot != 0)[0])
        costs_k.append(costs[idx_min])
        betas_k.append(betas[idx_min])


#     gammas_k = list(np.flip(np.cumsum(-np.diff(costs_k)[::-1])) + 1e-3)
    gammas_k = list(-np.diff(costs_k) + 1e-3)
    gammas_k.append(1e-3)

    return supp_k, costs_k, betas_k, gammas_k


def socp_bidual(D, L, p, k, beta, verbose=False, solver='ecos'):
    """ SOCP bidual solved using cvx"""

    if not isinstance(k, list):
        k = [k]
        beta = [beta]

    supp_k, costs_k = [], []

    for b, kk in zip(beta, k):
        if verbose:
            print(f'kk = {kk}')
        m = len(p)
        l = L.shape[1]

        # cvx problem
        x = cp.Variable(m)
        z = cp.Variable(l)

        # constraints
        constr = [
            L.T @ x == z,
            cp.norm(x, 1) <= kk * b,
            cp.norm(x, 'inf') <= b
        ]

        # cost
        cost = 1 / 2 * cp.quad_form(x, D) + 1 / 2 * cp.sum_squares(z) + p.T @ x

        prob = cp.Problem(cp.Minimize(cost), constr)
        if solver == 'scs':
            prob.solve(solver=cp.SCS, eps=scs_eps)
        elif solver == 'ecos':
            prob.solve(solver=cp.ECOS,
                       verbose=verbose,
                       abstol=abstol,
                       reltol=reltol,
                       feastol=feastol)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)

        costs_k.append(cost.value)

    return costs_k


def socp_dual(D, L, p, k, beta, verbose=False, solver='ecos'):
    """ SOCP dual solved using cvx"""

    if not isinstance(k, list):
        k = [k]
        beta = [beta]

    supp_k, costs_k = [], []

    for b, kk in zip(beta, k):
        if verbose:
            print(f'kk = {kk}')
        m = len(p)
        l = L.shape[1]
        Dhalf = np.diag(np.sqrt(np.diag(D)))

        # cvx problem
        mu = cp.Variable(m + 2)
        chi = cp.Variable(l + 2)

        # constraints
        constr = [
            cp.SOC(mu[-1], mu[:-1]),
            cp.SOC(chi[-1], chi[:-1]), mu[-1] - mu[-2] == 2,
            chi[-1] - chi[-2] == 2
        ]

        # cost
        alpha = cp.abs(2 * p - L @ chi[:l] - Dhalf @ mu[:m])
        cost = -b * cp.sum_largest(
            alpha, kk) - (mu[-1] + mu[-2] + chi[-1] + chi[-2]) / 2

        prob = cp.Problem(cp.Maximize(cost), constr)
        if solver == 'scs':
            prob.solve(solver=cp.SCS, eps=scs_eps)
        elif solver == 'ecos':
            prob.solve(solver=cp.ECOS,
                       verbose=verbose,
                       abstol=abstol,
                       reltol=reltol,
                       feastol=feastol)
        elif solver == 'mosek':
            prob.solve(solver=cp.MOSEK)

        # recover support from alpha.value
        support = alpha.value.argsort()[-kk:][::-1]
        support.sort()

        supp_k.append(support)
        costs_k.append(cost.value / 2)

        if verbose:
            print(alpha.value)

    return supp_k, costs_k


def penalized_dual(D, L, p, gamma, verbose=False, solver='ecos'):
    """ Solve dual of l0 penalized primal """

    if not isinstance(gamma, list):
        gamma = [gamma]

    supp_k, costs_k = [], []

    for g in gamma:
        if verbose:
            print(f'gamma = {g}')
        m = len(p)
        l = L.shape[1]

        # cvx problem
        nu = cp.Variable(l)

        alpha = -1 / 2 * cp.multiply(1 / np.diag(D), (p - L @ nu)**2)
        cost = 1 / 2 * cp.sum_squares(nu) - cp.sum(cp.min(alpha + g, 0))

        prob = cp.Problem(cp.Minimize(cost))
        if solver == 'scs':
            prob.solve(solver=cp.SCS)
        elif solver == 'ecos':
            prob.solve(solver=cp.ECOS,
                       verbose=verbose,
                       abstol=abstol,
                       reltol=reltol,
                       feastol=feastol)

        # recover support from alpha.value
        support = np.where(alpha.value + g <= 0)[0]
        support.sort()

        supp_k.append(support)
        costs_k.append(-cost.value)

    return supp_k, costs_k


def cvx_dual(D, L, p, k, verbose=False, solver='ecos'):
    """ Dual solved using cvx """

    if not isinstance(k, list):
        k = [k]

    supp_k, costs_k = [], []

    for kk in k:
        if verbose:
            print(f'kk = {kk}')
        m = len(p)
        l = L.shape[1]
        iD = np.linalg.inv(D)

        # cvx problem
        nu = cp.Variable(l)

        alpha = -1 / 2 * cp.multiply(1 / np.diag(D), (p + L @ nu)**2)
        cost = 0.5 * cp.sum_squares(nu) + cp.sum_largest(-alpha, kk)

        prob = cp.Problem(cp.Minimize(cost))
        if solver == 'scs':
            prob.solve(solver=cp.SCS)
        elif solver == 'ecos':
            prob.solve(solver=cp.ECOS,
                       verbose=verbose,
                       abstol=abstol,
                       reltol=reltol,
                       feastol=feastol)

        # recover support from t.value
        support = (-alpha).value.argsort()[-kk:][::-1]
        support.sort()

        supp_k.append(support)
        costs_k.append(-cost.value)

    return supp_k, costs_k


def lasso_solve(X, y, k):
    """ Solve sparse QP with l1 penalization -- same as lars solve """

    L1 = LassoLars(alpha=1e-6, fit_intercept=False, normalize=False)
    L1.fit(X, y)
    supp_k, l1_sparsity = [], []
    for c in L1.coef_path_.T:
        supp = np.where(c != 0)[0]
        supp_k.append(supp)
        l1_sparsity.append(len(supp))
    indexes = [l1_sparsity.index(x) for x in set(l1_sparsity)]
    l1_sparsity = np.array(l1_sparsity)[np.array(indexes)[1:]]
    supp_k = [j for i, j in enumerate(supp_k) if i in indexes][1:]

    return supp_k


########### not in use ##############################
def cvx_bidual(D, L, p, k, verbose=False, solver=cp.SCS):
    """ Bidual SDP solved using cvx """

    m = len(p)
    l = L.shape[1]
    Dhalf = np.linalg.inv(np.sqrt(D))

    ## cvx problem
    t = cp.Variable(1)
    x = cp.Variable(m)
    X = cp.Variable((l + 1, l + 1), PSD=True)

    Dx = cp.diag(x)
    Sigma = Dhalf @ Dx @ Dhalf

    cost = 1 / 2 * t
    constraints = []
    constraints = [
        X[0, 0] == t + p.T @ Sigma @ p, X[0, 1:l + 1] == p.T @ Sigma @ L,
        X[1:l + 1, 0] == L.T @ Sigma @ p,
        X[1:l + 1, 1:l + 1] == np.eye(l) + L.T @ Sigma @ L, X >> 0,
        cp.sum(x) <= k, x <= 1, x >= 0
    ]

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=solver, verbose=verbose)

    return x.value, 1 / 2 * t.value[0]


def lars_solve(X, y, k):
    """ Solve LASSO problem and get path """

    _, _, coefs = lars_path(X, y, method='lasso', verbose=True)
    supp_k, l1_sparsity = [], []
    for c in coefs.T:
        supp = np.where(c != 0)[0]
        supp_k.append(supp)
        l1_sparsity.append(len(supp))
    indexes = [l1_sparsity.index(x) for x in set(l1_sparsity)]
    l1_sparsity = np.array(l1_sparsity)[np.array(indexes)[1:]]
    supp_k = [j for i, j in enumerate(supp_k) if i in indexes][1:]

    return supp_k[:k]


def l0_penalized_bidual_bfgs(D, L, p, g, veerbose=False):
    """ Bidual for l0 penalized problem """

    if not isinstance(g, list):
        g = [g]

    supp_k, costs_k = [], []

    for gg in g:
        m = len(p)
        c = np.sum(np.divide(p**2, 2 * np.diag(D)))

        def primal(x):
            f = np.linalg.norm(L.T @ x)**2
            df = 2 * L @ L.T @ x
            for i in range(m):
                fac = np.sqrt(2 * gg / D[i, i])
                if np.abs(x[i]) <= fac:
                    f += 2 * p[i] * x[i] + 2 * D[i, i] * np.abs(x[i]) * fac
                    df[i] += 2 * p[i] + 2 * D[i, i] * np.sign(x[i]) * fac
                else:
                    f += D[i, i] * x[i]**2 + 2 * p[i] * x[i] + gg
                    df[i] += 2 * D[i, i] * x[i] + 2 * p[i]

            return f - 4 * gg * m, df

        x0 = np.ones(m)
        options = {'disp': False}
        res = minimize(primal,
                       x0,
                       jac=True,
                       method='L-BFGS-B',
                       options=options)

        costs_k.append(res.fun / 2)

    return costs_k

 
def factor_alternating_min(Sigma, rank, tol=1e-5):
    """ Performing alternating min to get D + LL.T """
    
    assert Sigma.shape[0] == Sigma.shape[1]
    n = Sigma.shape[0]
    d = np.zeros(n)
    
    costs = [1, 2]
    while np.abs(costs[-1] - costs[-2])/np.abs(costs[-1]) > tol:
        u, s, v = np.linalg.svd(Sigma - np.diag(d))
        U = u[:, :rank] * np.sqrt(s[:rank])
        d = np.clip(np.diag(Sigma - U @ U.T),
                           a_min = 1e-3,
                           a_max = None)
        costs.append(np.linalg.norm(Sigma - np.diag(d) - U @ U.T))
    
    return np.diag(d), U, costs
    

def factor_decomp(Sigma, rank=1):
    """ Get factor decomposition of covariance via alternating min """
    
    if rank != -1:
        return factor_alternating_min(Sigma, rank)
    else:
        rank_costs = []
        for rank in range(1,Sigma.shape[0]-1):
            D, L, costs = factor_alternating_min(Sigma, rank)
            rank_costs.append(costs[-1])
        
        return rank_costs
            

def generate_factor_model(m, r):
    """ Generates random factor model structure """
    
    assert r >= 0   
    D = np.diag(np.random.rand(m)) # must be PD
    L = np.zeros((m,m))
    if r > 0:
        L = np.random.randn(m,r)
    p = np.random.randn(m)
    
    return D, L, p

                
def factor_model_to_ls(D, L, sparsity_lvl = 0.2):
    """ Convert factor model structure to LS matrices """
    
    m = D.shape[0]
    X = scipy.linalg.sqrtm(D + L @ L.T)
    
    # generate ground truth model
    beta = 10 * np.random.randn(m)
    indices = np.random.choice(np.arange(m), 
                               replace=False,
                               size=int(m * sparsity_lvl))
    beta[indices] = 0
    
    y = X @ beta + np.random.randn(m)
    
    return X, y
    
 
def factor_alternating_min(Sigma, rank, tol=1e-5):
    """ Performing alternating min to get D + LL.T """
    
    assert Sigma.shape[0] == Sigma.shape[1]
    n = Sigma.shape[0]
    d = np.zeros(n)
    
    costs = [1, 2]
    while np.abs(costs[-1] - costs[-2])/np.abs(costs[-1]) > tol:
        u, s, v = np.linalg.svd(Sigma - np.diag(d))
        U = u[:, :rank] * np.sqrt(s[:rank])
        d = np.clip(np.diag(Sigma - U @ U.T),
                           a_min = 1e-3,
                           a_max = None)
        costs.append(np.linalg.norm(Sigma - np.diag(d) - U @ U.T))
    
    return np.diag(d), U, costs
    

def factor_decomp(Sigma, rank=1):
    """ Get factor decomposition of covariance via alternating min """
    
    if rank != -1:
        return factor_alternating_min(Sigma, rank)
    else:
        rank_costs = []
        for rank in range(1,Sigma.shape[0]-1):
            D, L, costs = factor_alternating_min(Sigma, rank)
            rank_costs.append(costs[-1])
        
        return rank_costs
            

def generate_factor_model(m, r):
    """ Generates random factor model structure """
    
    assert r >= 0   
    D = np.diag(np.random.rand(m)) # must be PD
    L = np.zeros((m,m))
    if r > 0:
        L = np.random.randn(m,r)
    p = np.random.randn(m)
    
    return D, L, p

                
def factor_model_to_ls(D, L, sparsity_lvl = 0.2):
    """ Convert factor model structure to LS matrices """
    
    m = D.shape[0]
    X = scipy.linalg.sqrtm(D + L @ L.T)
    
    # generate ground truth model
    beta = 10 * np.random.randn(m)
    indices = np.random.choice(np.arange(m), 
                               replace=False,
                               size=int(m * sparsity_lvl))
    beta[indices] = 0
    
    y = X @ beta + np.random.randn(m)
    
    return X, y
    
 
"""
