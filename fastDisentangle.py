import math
import numpy as np

def fastDisentangle(chi1, chi2, A):
    """Calculate a unitary tensor with shape (chi1, chi2, chi1*chi2)
    that approximately disentangles the input tensor 'A'.
    'A' must have the shape (chi1*chi2, chi3, chi4) where either
    chi2 <= ceil(chi4 / ceil(chi1/chi3)) or chi1 <= ceil(chi3 / ceil(chi2/chi4)).
    If these ratios are integers, then chi1*chi2 <= chi3*chi4 is sufficient.
    chi1 <= chi3 and chi2 <= chi4 is also sufficient.
    
    example: fastDisentangle(2, 3, randomComplex([6,5,7]))
    """
    A = np.asarray(A)
    n,chi3,chi4 = A.shape
    if n != chi1*chi2:
        raise ValueError("fastDisentangle: The input array must have the shape (chi1*chi2, chi3, chi4).")
    
    # implementing Appendix B in https://arxiv.org/pdf/2104.08283
    if chi1 > chi3:
        chi4to3 = math.ceil(chi1 / chi3)
        chi4new = math.ceil(chi4 / chi4to3)
        if not chi2 <= chi4new:
            raise ValueError("""
                fastDisentangle: The input array must have the shape (chi1*chi2, chi3, chi4) where either
                chi2 <= ceil(chi4 / ceil(chi1/chi3)) or
                chi1 <= ceil(chi3 / ceil(chi2/chi4)).""")
        _,_,V1  = np.linalg.svd(np.reshape(A, (n*chi3, chi4))) # 1
        V1      = V1.conj().T
        V       = np.pad(V1, ((0,0), (0, chi4to3*chi4new - chi4))) # 2
        V       = np.reshape(V, (chi4, chi4to3, chi4new))
        Anew    = np.reshape(np.tensordot(A, V, 1), (n, chi3*chi4to3, chi4new)) # 3
        return fastDisentangle(chi1, chi2, Anew)
    
    if chi2 > chi4:
        return np.transpose(fastDisentangle(chi2, chi1, np.transpose(A, (0,2,1))), (1,0,2))
    
    # implementing Algorithm 1 in https://arxiv.org/pdf/2104.08283
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
                M[i] = Mi / norm
                break
    return M

def entanglement(UA):
    """Compute the entanglement entropy of UA = np.tensordot(U, A, 1)"""
    # defined in equation (10) of https://arxiv.org/pdf/2104.08283
    UA = np.asarray(UA)
    chi1,chi2,chi3,chi4 = UA.shape
    lambdas  = np.linalg.svd(np.reshape(np.swapaxes(UA, 1, 2), (chi1*chi3, chi2*chi4)), compute_uv=False)
    ps  = lambdas*lambdas
    ps /= np.sum(ps)
    return max(0., -np.dot(ps, np.log(np.maximum(ps, np.finfo(ps.dtype).tiny))))
