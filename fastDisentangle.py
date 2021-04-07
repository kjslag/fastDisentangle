import numpy as np

def fastDisentangle(chi1, chi2, A):
    """Calculate a unitary tensor with shape (chi1, chi2, chi1*chi2)
    that approximately disentangles the input tensor 'A'.
    'A' must have a shape (chi1*chi2, chi3, chi4) where chi1<=chi3 and chi2<=chi4.
    
    example: fastDisentangle(2, 3, randomComplex([6,5,7]))
    """
    A = np.asarray(A)
    n,chi3,chi4 = A.shape
    if not (n == chi1*chi2 and chi1 <= chi3 and chi2 <= chi4):
        raise ValueError("fastDisentangle: 'A' must have a shape (chi1*chi2, chi3, chi4) where chi1<=chi3 and chi2<=chi4.")
    r = randomComplex(n) # 1
    alpha3, _, alpha4 = np.linalg.svd(np.tensordot(r, A, 1), full_matrices=False) # 2
    alpha3, alpha4 = np.conj(alpha3[:,0]), np.conj(alpha4[0,:])
    V3  = np.mat(np.linalg.svd(np.tensordot(A, alpha4,   1  ), full_matrices=False)[2][:chi1]).H # 3
    V4  = np.mat(np.linalg.svd(np.tensordot(A, alpha3, (1,0)), full_matrices=False)[2][:chi2]).H # 4
    B   = np.einsum("kab,ai,bj -> kij", A, V3, V4, optimize=True) # 5
    transposeQ = chi1 > chi2
    Bdg = np.transpose(np.conj(B), (2,1,0) if transposeQ else (1,2,0))
    U   = np.reshape(orthogonalize(np.reshape(Bdg, (chi1*chi2, n))), Bdg.shape) # 6
    if transposeQ:
        U = np.swapaxes(U, 0, 1)
    return U

def randomComplex(ns):
    """Gaussian random array with dimensions ns"""
    if isinstance(ns, int):
        ns = [ns]
    return np.reshape(np.random.normal(scale=1/np.sqrt(2), size=ns+[2]).view(np.complex128), ns)

def orthogonalize(M):
    """Gram-Schmidt orthonormalization of the rows of M.
       Inserts random vectors in the case of linearly dependent rows."""
    M = np.array(M)
    assert M.shape[0] <= M.shape[1]
    for i in range(0, M.shape[0]):
        Mi = M[i]
        while True:
            for j in range(0, i):
                Mi = Mi - M[j] * (np.conj(M[j]) @ Mi)
            norm = np.linalg.norm(Mi)
            if norm == 0:
                # M[i] was a linear combination of M[:i-1]
                # try a random vector instead:
                Mi = randomComplex(Mi.shape[0])
            else:
                M[i] = Mi / np.linalg.norm(Mi)
                break
    return M

def entanglement(UA):
    """Compute the entanglement entropy of UA = np.tensordot(U, A, 1)"""
    UA = np.asarray(UA)
    chi1,chi2,chi3,chi4 = UA.shape
    lambdas  = np.linalg.svd(np.reshape(np.swapaxes(UA, 1, 2), (chi1*chi3, chi2*chi4)), compute_uv=False)
    ps  = lambdas*lambdas
    ps /= np.sum(ps)
    return max(0., -np.sum(ps * np.log(np.maximum(ps, np.finfo(ps.dtype).tiny))))
