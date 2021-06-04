import math
import numpy as np

def fastDisentangle(chi1, chi2, A, transposeQ=None):
    """Calculate a unitary tensor with shape (chi1, chi2, chi1*chi2)
    that approximately disentangles the input tensor 'A'.
    'A' must have the shape (chi1*chi2, chi3, chi4) where either
    chi2 <= ceil(chi4 / ceil(chi1/chi3)) or chi1 <= ceil(chi3 / ceil(chi2/chi4)).
    If these ratios are integers, then chi1*chi2 <= chi3*chi4 is sufficient.
    chi1 <= chi3 and chi2 <= chi4 is also sufficient.
    
    example: fastDisentangle(2, 3, randomComplex([6,5,7]))
    """
    A = np.asarray(A)
    rand = randomComplex if np.iscomplexobj(A) else randomReal
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
        return fastDisentangle(chi1, chi2, Anew, False)
    
    if chi2 > chi4:
        return np.transpose(fastDisentangle(chi2, chi1, np.transpose(A, (0,2,1))), (1,0,2))
    
    # implementing Algorithm 1 in https://arxiv.org/pdf/2104.08283
    r = rand(n) # 1
    alpha3, _, alpha4 = np.linalg.svd(np.tensordot(r, A, 1), full_matrices=False) # 2
    alpha3, alpha4 = np.conj(alpha3[:,0]), np.conj(alpha4[0,:])
    V3  = np.mat(np.linalg.svd(np.tensordot(A, alpha4,   1  ), full_matrices=False)[2][:chi1]).H # 3
    V4  = np.mat(np.linalg.svd(np.tensordot(A, alpha3, (1,0)), full_matrices=False)[2][:chi2]).H # 4
    B   = np.einsum("kab,ai,bj -> kij", A, V3, V4, optimize=True) # 5
    if transposeQ is None:
        transposeQ = chi1 > chi2
    Bdg = np.transpose(np.conj(B), (2,1,0) if transposeQ else (1,2,0))
    U   = np.reshape(orthogonalize(np.reshape(Bdg, (chi1*chi2, n))), Bdg.shape) # 6
    if transposeQ:
        U = np.swapaxes(U, 0, 1)
    return U

def randomReal(*ns):
    """Gaussian random array with dimensions ns"""
    return np.random.normal(size=ns)

def randomComplex(*ns):
    """Gaussian random array with dimensions ns"""
    return np.reshape(np.random.normal(scale=1/np.sqrt(2), size=(*ns,2)).view(np.complex128), ns)

def orthogonalize(M):
    """Gram-Schmidt orthonormalization of the rows of M.
       Inserts random vectors in the case of linearly dependent rows."""
    M = np.array(M)
    rand = randomComplex if np.iscomplexobj(M) else randomReal
    epsMin = np.sqrt(np.finfo(M.dtype).eps) # once eps<epsMin, we add random vectors if needed
    eps = 0.5*np.sqrt(epsMin) # only accept new orthogonal vectors if their relative norm is at last eps after orthogonalization
    m,n = M.shape
    assert m <= n
    norms = np.linalg.norm(M, axis=1)
    maxNorm = np.max(norms)
    orthogQ = np.zeros(m, 'bool')
    allOrthog = False
    while not allOrthog:
        allOrthog = True
        eps1 = max(eps, epsMin)
        for i in range(m):
            if not orthogQ[i]:
                if norms[i] > eps1 * maxNorm:
                    Mi = M[i]
                    for j in range(m):
                        if orthogQ[j]:
                            Mi = Mi - M[j] * (np.conj(M[j]) @ Mi)
                    norm = np.linalg.norm(Mi)
                    if norm > eps1 * norms[i]:
                        M[i] = Mi / norm
                        orthogQ[i] = True
                        continue
                # M[i] was a linear combination of the other vectors
                if eps < epsMin:
                    M[i]  = rand(n)
                    M[i] *= maxNorm/np.linalg.norm(M[i])
                    norms[i] = maxNorm
                allOrthog = False
        eps = eps*eps
    #assert(np.linalg.norm(M * np.mat(M).H - np.eye(m)) < np.sqrt(epsMin))
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

# verification code

def checkAnsatz(chi1, chi2, chi3a, chi4b, chi3c, chi4c, eps=0, complexQ=True):
    """Check that the ansatz in equation (1) of https://arxiv . org/pdf/2104.08283 results in the minimal entanglement entropy."""
    rand = randomComplex if complexQ else randomReal
    normalize = lambda x: x / np.linalg.norm(x)
    M1,M2,M3 = rand(chi1,chi3a), rand(chi2,chi4b), rand(chi3c,chi4c)
    A = np.reshape(np.transpose(np.tensordot(np.tensordot(M1, M2, 0), M3, 0), (0,2,1,4,3,5)),
                   (chi1*chi2, chi3a*chi3c, chi4b*chi4c))
    A = normalize(A) + eps * normalize(rand(*A.shape))
    U = fastDisentangle(chi1,chi2,A)
    return entanglement(np.tensordot(U,A,1)) - entanglement(np.reshape(M3,(1,1,chi3c,chi4c)))

def checkAnsatzRepeated(maxChi=5):
    """repeatedly check the ansatz"""
    c = 0
    while True:
        c += 1
        if c%1000 == 0:
            print(c)
        chis = np.random.randint(1,maxChi,6)
        chi1,chi2,chi3a,chi4b,chi3c,chi4c = chis
        chi3 = chi3a*chi3c
        chi4 = chi4b*chi4c
        eps  = 10**np.random.uniform(-20,-6)
        complexQ = np.random.choice([True, False])
        #if chi2 <= math.ceil(chi4 / math.ceil(chi1/chi3)) or chi1 <= math.ceil(chi3 / math.ceil(chi2/chi4)):
        if (chi1 <= chi3 and chi2 <= chi4) or \
           ((chi3c==1 or chi4c==1) and (chi2 <= math.ceil(chi4 / math.ceil(chi1/chi3)) or
                                        chi1 <= math.ceil(chi3 / math.ceil(chi2/chi4)))):
            args = (*chis, eps, complexQ)
            S = checkAnsatz(*args)
            if S > np.sqrt(max(eps, np.finfo(S).eps)):
                print(args, " -> ", S)
                return
