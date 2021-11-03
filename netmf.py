#!/usr/bin/env python
# encoding: utf-8
# File Name: eigen.py
# Author: Jiezhong Qiu
# Create Time: 2017/07/13 16:05

import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import argparse
import logging

logger = logging.getLogger(__name__)

def load_adjacency_matrix(file, variable_name="network"):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]

def deepwalk_filter(evals, window):
    for i in range(len(evals)):
        x = evals[i]
        evals[i] = 1. if x >= 1 else x*(1-x**window) / (1-x) / window
    evals = np.maximum(evals, 0)
    logger.info("After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals), np.min(evals))
    return evals

def approximate_normalized_graph_laplacian(A, rank, which="LA"):
    n = A.shape[0]
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} W D^{-1/2}
    X = sparse.identity(n) - L
    logger.info("Eigen decomposition...")
    #evals, evecs = sparse.linalg.eigsh(X, rank,
    #        which=which, tol=1e-3, maxiter=300)
    evals, evecs = sparse.linalg.eigsh(X, rank, which=which)
    logger.info("Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals))
    logger.info("Computing D^{-1/2}U..")
    D_rt_inv = sparse.diags(d_rt ** -1)
    D_rt_invU = D_rt_inv.dot(evecs)
    return evals, D_rt_invU

def approximate_deepwalk_matrix(evals, D_rt_invU, window, vol, b):
    import aesara
    from aesara import tensor as T
    evals = deepwalk_filter(evals, window=window)
    X = sparse.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
    m = T.matrix()
    mmT = T.dot(m, m.T) * (vol/b)
    f = aesara.function([m], T.log(T.maximum(mmT, 1)))
    Y = f(X.astype(aesara.config.floatX))
    logger.info("Computed DeepWalk matrix with %d non-zero elements",
            np.count_nonzero(Y))
    return sparse.csr_matrix(Y)

def svd_deepwalk_matrix(X, dim):
    u, s, v = sparse.linalg.svds(X, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sparse.diags(np.sqrt(s)).dot(u.T).T


def netmf_large(args):
    logger.info("Running NetMF for a large window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    vol = float(A.sum())
    # perform eigen-decomposition of D^{-1/2} A D^{-1/2}
    # keep top #rank eigenpairs
    evals, D_rt_invU = approximate_normalized_graph_laplacian(A, rank=args.rank, which="LA")

    # approximate deepwalk matrix
    deepwalk_matrix = approximate_deepwalk_matrix(evals, D_rt_invU,
            window=args.window,
            vol=vol, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)

    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)


def direct_compute_deepwalk_matrix(A, window, b):
    import aesara
    from aesara import tensor as T
    n = A.shape[0]
    vol = float(A.sum())
    L, d_rt = csgraph.laplacian(A, normed=True, return_diag=True)
    # X = D^{-1/2} A D^{-1/2}
    X = sparse.identity(n) - L
    S = np.zeros_like(X)
    X_power = sparse.identity(n)
    for i in range(window):
        logger.info("Compute matrix %d-th power", i+1)
        X_power = X_power.dot(X)
        S += X_power
    S *= vol / window / b
    D_rt_inv = sparse.diags(d_rt ** -1)
    M = D_rt_inv.dot(D_rt_inv.dot(S).T)
    m = T.matrix()
    f = aesara.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.todense().astype(aesara.config.floatX))
    return sparse.csr_matrix(Y)

def compute_S_paths(paths, N, window):
    data = dict()
    for path in paths:
        for i in range(len(path)):
            for j in range(i+1, i+window+1):
                if j >= len(path):
                    break
                u = path[i]
                v = path[j]
                data.setdefault(j-i, dict())
                data[j-i].setdefault((u,v), 0)
                data[j-i][(u,v)] += 1

    S = np.zeros((N,N))
    node_mapping = dict()
    idx = 0
    for d, mat_dat in data.items():
        curr = np.zeros((N,N))
        for (u,v) in mat_dat:
            if u not in node_mapping:
                node_mapping[u] = idx
                idx += 1
            if v not in node_mapping:
                node_mapping[v] = idx
                idx += 1
            curr[node_mapping[u], node_mapping[v]] += mat_dat[(u,v)]

        S += curr

    return S

def direct_compute_seqwalk_matrix(A, window, b, paths):
    import aesara
    from aesara import tensor as T
    # TODO: Their function uses the laplacian and I don't get why
    n = A.shape[0]
    vol = float(A.sum())
    # Simple option: seq_walk_mat_k(paths, i) that returns an NxN 
    # matrix where each entry is the total number of real paths that 
    # started at i and ended at j
    S = compute_S_paths(paths, n, window)
    M = S * (vol) / ( window * b )
    m = T.matrix()
    f = aesara.function([m], T.log(T.maximum(m, 1)))
    Y = f(M.astype(aesara.config.floatX))
    return sparse.csr_matrix(Y)


def netmf_small(args):
    logger.info("Running NetMF for a small window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix
    A = load_adjacency_matrix(args.input,
            variable_name=args.matfile_variable_name)
    # directly compute deepwalk matrix
    deepwalk_matrix = direct_compute_deepwalk_matrix(A,
            window=args.window, b=args.negative)

    # factorize deepwalk matrix with SVD
    deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
    logger.info("Save embedding to %s", args.output)
    np.save(args.output, deepwalk_embedding, allow_pickle=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
            help=".mat input file path")
    parser.add_argument('--matfile-variable-name', default='network',
            help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--output", type=str, required=True,
            help="embedding output file path")

    parser.add_argument("--rank", default=256, type=int,
            help="#eigenpairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
            help="dimension of embedding")
    parser.add_argument("--window", default=10,
            type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
            help="negative sampling")

    parser.add_argument('--large', dest="large", action="store_true",
            help="using netmf for large window size")
    parser.add_argument('--small', dest="large", action="store_false",
            help="using netmf for small window size")
    parser.set_defaults(large=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(message)s') # include timestamp

    if args.large:
        netmf_large(args)
    else:
        netmf_small(args)

