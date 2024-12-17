# Extracted from
# https://github.com/networkx/networkx/blob/main/networkx/linalg/algebraicconnectivity.py

import numpy as np
import scipy as sp
import networkx as nx
import cupy as cp

GPU = 0
try:
    cp.cuda.Device().compute_capability
    GPU = 1
    print("Using GPU")
except cp.cuda.runtime.CUDARuntimeError:
    print("CUDA GPU not available")


class _PCGSolver:
    """Preconditioned conjugate gradient method.

    To solve Ax = b:
        M = A.diagonal() # or some other preconditioner
        solver = _PCGSolver(lambda x: A * x, lambda x: M * x)
        x = solver.solve(b)

    The inputs A and M are functions which compute
    matrix multiplication on the argument.
    A - multiply by the matrix A in Ax=b
    M - multiply by M, the preconditioner surrogate for A

    Warning: There is no limit on number of iterations.
    """

    def __init__(self, A, M):
        self._A = A
        self._M = M
        self.solve = self._solve_gpu if GPU else self._solve_cpu

    def _solve_cpu(self, B, tol):
        # Densifying step - can this be kept sparse?
        B = np.asarray(B)
        X = np.ndarray(B.shape, order="F")
        for j in range(B.shape[1]):
            X[:, j] = self._solve(B[:, j], tol)
        return X

    def _solve_gpu(self, B, tol):
        with cp.cuda.Device(0):
            B = cp.asarray(B)
            X = cp.ndarray(B.shape, order="F")
            for j in range(B.shape[1]):
                X[:, j] = self._solve(B[:, j], tol)
            return X

    def _solve(self, b, tol):
        xp = cp.get_array_module(b)
        A = self._A
        M = self._M
        tol *= xp.linalg.norm(b, ord=1)
        # Initialize.
        x = xp.zeros(b.shape)
        r = b.copy()
        z = M(r)
        rz = xp.dot(r, z)
        p = z.copy()
        # Iterate.
        while True:
            Ap = A(p)
            alpha = rz / xp.dot(p, Ap)
            x += alpha * p
            r += -alpha * Ap
            if xp.linalg.norm(r, ord=1) < tol:
                return x
            z = M(r)
            beta = xp.dot(r, z)
            beta, rz = beta / rz, beta
            p = beta * p + z


def _tracemin_fiedler(L, X, normalized, tol):
    """Compute the Fiedler vector of L using the TraceMIN-Fiedler algorithm.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph. This function starts with the Laplacian L, not the Graph.

    Parameters
    ----------
    L : Laplacian of a possibly weighted or normalized, but undirected graph

    X : Initial guess for a solution. Usually a matrix of random numbers.
        This function allows more than one column in X to identify more than
        one eigenvector if desired.

    normalized : bool
        Whether the normalized Laplacian matrix is used.

    tol : float
        Tolerance of relative residual in eigenvalue computation.
        Warning: There is no limit on number of iterations.

    method : string
        Should be 'tracemin_pcg' or 'tracemin_lu'.
        Otherwise exception is raised.

    Returns
    -------
    sigma, X : Two NumPy arrays of floats.
        The lowest eigenvalues and corresponding eigenvectors of L.
        The size of input X determines the size of these outputs.
        As this is for Fiedler vectors, the zero eigenvalue (and
        constant eigenvector) are avoided.
    """

    n = X.shape[0]

    if normalized:
        # Form the normalized Laplacian matrix and determine the eigenvector of
        # its nullspace.
        e = np.sqrt(L.diagonal())
        # TODO: rm csr_array wrapper when spdiags array creation becomes available
        D = sp.sparse.csr_array(sp.sparse.spdiags(1 / e, 0, n, n, format="csr"))
        L = D @ L @ D
        e *= 1.0 / np.linalg.norm(e, 2)

    if normalized:

        def project(X):
            """Make X orthogonal to the nullspace of L."""
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= (X[:, j] @ e) * e

    else:

        def project(X, xp):
            """Make X orthogonal to the nullspace of L."""
            X = xp.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j].sum() / n

    D = L.diagonal().astype(float)
    if GPU:
        L = cp.sparse.csr_matrix(L)
        D = cp.asarray(D)
        X = cp.asarray(X)
    xp = cp.get_array_module(X)
    solver = _PCGSolver(lambda x: L @ x, lambda x: D * x)

    # Initialize.
    Lnorm = abs(L).sum(axis=1).flatten().max()
    project(X, xp)
    W = xp.ndarray(X.shape, order="F")

    while True:
        # Orthonormalize X.
        X = xp.linalg.qr(X)[0]
        # Compute iteration matrix H.
        W[:, :] = L @ X
        H = X.T @ W
        sigma, Y = xp.linalg.eigh(H)
        # Compute the Ritz vectors.
        X = X @ Y
        # Test for convergence exploiting the fact that L * X == W * Y.
        res = xp.linalg.norm(W @ Y[:, 0] - sigma[0] * X[:, 0], ord=1) / Lnorm
        if res < tol:
            break
        # Compute X = L \ X / (X' * (L \ X)).
        # L \ X can have an arbitrary projection on the nullspace of L,
        # which will be eliminated.
        W[:, :] = solver.solve(X, tol)
        X = (xp.linalg.inv(W.T @ X) @ W.T).T  # Preserves Fortran storage order.
        project(X, xp)

    if GPU:
        X = cp.asnumpy(X)
    return sigma, np.asarray(X)


def find_fiedler(L, x, normalized, tol, seed):
    q = 1
    X = np.asarray(seed.normal(size=(q, L.shape[0]))).T
    sigma, X = _tracemin_fiedler(L, X, normalized, tol)
    return sigma[0], X[:, 0]


def _preprocess_graph(G, weight):
    """Compute edge weights and eliminate zero-weight edges."""
    if G.is_directed():
        H = nx.MultiGraph()
        H.add_nodes_from(G)
        H.add_weighted_edges_from(
            ((u, v, e.get(weight, 1.0)) for u, v, e in G.edges(data=True) if u != v),
            weight=weight,
        )
        G = H
    if not G.is_multigraph():
        edges = (
            (u, v, abs(e.get(weight, 1.0))) for u, v, e in G.edges(data=True) if u != v
        )
    else:
        edges = (
            (u, v, sum(abs(e.get(weight, 1.0)) for e in G[u][v].values()))
            for u, v in G.edges()
            if u != v
        )
    H = nx.Graph()
    H.add_nodes_from(G)
    H.add_weighted_edges_from((u, v, e) for u, v, e in edges if e != 0)
    return H


def fiedler_vector(
    G, weight="weight", normalized=False, tol=1e-8, method="tracemin_pcg", seed=None
):
    """Returns the Fiedler vector of a connected undirected graph.

    The Fiedler vector of a connected undirected graph is the eigenvector
    corresponding to the second smallest eigenvalue of the Laplacian matrix
    of the graph.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    fiedler_vector : NumPy array of floats.
        Fiedler vector.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    NetworkXError
        If G has less than two nodes or is not connected.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix

    Examples
    --------
    Given a connected graph the signs of the values in the Fiedler vector can be
    used to partition the graph into two components.

    >>> G = nx.barbell_graph(5, 0)
    >>> nx.fiedler_vector(G, normalized=True, seed=1)
    array([-0.32864129, -0.32864129, -0.32864129, -0.32864129, -0.26072899,
            0.26072899,  0.32864129,  0.32864129,  0.32864129,  0.32864129])

    The connected components are the two 5-node cliques of the barbell graph.
    """
    import numpy as np

    if len(G) < 2:
        raise nx.NetworkXError("graph has less than two nodes.")
    G = _preprocess_graph(G, weight)
    if not nx.is_connected(G):
        raise nx.NetworkXError("graph is not connected.")

    if len(G) == 2:
        return np.array([1.0, -1.0])

    L = nx.laplacian_matrix(G)
    x = None
    sigma, fiedler = find_fiedler(L, x, normalized, tol, seed)
    return fiedler


def spectral_ordering(
    G,
    weight="weight",
    normalized=False,
    tol=1e-8,
    method="tracemin_pcg",
    seed=np.random,
):
    """Compute the spectral_ordering of a graph.

    The spectral ordering of a graph is an ordering of its nodes where nodes
    in the same weakly connected components appear contiguous and ordered by
    their corresponding elements in the Fiedler vector of the component.

    Parameters
    ----------
    G : NetworkX graph
        A graph.

    weight : object, optional (default: None)
        The data key used to determine the weight of each edge. If None, then
        each edge has unit weight.

    normalized : bool, optional (default: False)
        Whether the normalized Laplacian matrix is used.

    tol : float, optional (default: 1e-8)
        Tolerance of relative residual in eigenvalue computation.

    method : string, optional (default: 'tracemin_pcg')
        Method of eigenvalue computation. It must be one of the tracemin
        options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
        or 'lobpcg' (LOBPCG).

        The TraceMIN algorithm uses a linear system solver. The following
        values allow specifying the solver to be used.

        =============== ========================================
        Value           Solver
        =============== ========================================
        'tracemin_pcg'  Preconditioned conjugate gradient method
        'tracemin_lu'   LU factorization
        =============== ========================================

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    spectral_ordering : NumPy array of floats.
        Spectral ordering of nodes.

    Raises
    ------
    NetworkXError
        If G is empty.

    Notes
    -----
    Edge weights are interpreted by their absolute values. For MultiGraph's,
    weights of parallel edges are summed. Zero-weighted edges are ignored.

    See Also
    --------
    laplacian_matrix
    """
    if len(G) == 0:
        raise nx.NetworkXError("graph is empty.")
    G = _preprocess_graph(G, weight)

    order = []
    for component in nx.connected_components(G):
        size = len(component)
        if size > 2:
            L = nx.laplacian_matrix(G, component)
            x = None
            sigma, fiedler = find_fiedler(L, x, normalized, tol, seed)
            sort_info = zip(fiedler, range(size), component)
            order.extend(u for x, c, u in sorted(sort_info))
        else:
            order.extend(component)

    return order
