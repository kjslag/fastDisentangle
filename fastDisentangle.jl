# NOTE BLAS.set_num_threads(1) may increase performance significantly

using LinearAlgebra

"""
Calculate a unitary tensor with size (chi1, chi2, chi1*chi2)
that approximately disentangles the input tensor 'A'.
'A' must have the size (chi1*chi2, chi3, chi4) where either
chi2 <= ceil(chi4 / ceil(chi1/chi3)) or chi1 <= ceil(chi3 / ceil(chi2/chi4)).
If these ratios are integers, then chi1*chi2 <= chi3*chi4 is sufficient.
chi1 <= chi3 and chi2 <= chi4 is also sufficient.

example: fastDisentangle(2, 3, randn(6,5,7))
"""
function fastDisentangle(chi1::Int, chi2::Int, A::AbstractArray{T,3},
                         transposeQ::Union{Bool,Nothing}=nothing) where T
    n,chi3,chi4 = size(A)
    if n != chi1*chi2
        throw(ArgumentError("fastDisentangle: The input array must have the shape (chi1*chi2, chi3, chi4)"))
    end
    
    # implementing Appendix B in https://arxiv.org/pdf/2104.08283
    if chi1 > chi3
        chi4to3 = divUp(chi1, chi3)
        chi4new = divUp(chi4, chi4to3)
        if chi2 > chi4new
            throw(ArgumentError("""
                fastDisentangle: The input array must have the shape (chi1*chi2, chi3, chi4) where either
                chi2 <= ceil(chi4 / ceil(chi1/chi3)) or
                chi1 <= ceil(chi3 / ceil(chi2/chi4))."""))
        end
        V1      = svd(flatten(A, 2), full=true).V # 1
        V       = permutedims(reshape([V1 zeros(chi4, chi4to3*chi4new - chi4)], (chi4, chi4new, chi4to3)), [1,3,2]) # 2
        Anew    = reshape(tensordot(A, V), (n, chi3*chi4to3, chi4new)) # 3
        return fastDisentangle(chi1, chi2, Anew, false)
    end
    
    if chi2 > chi4
        return permutedims(fastDisentangle(chi2, chi1, permutedims(A, [1,3,2])), [2,1,3])
    end
    
    # implementing Algorithm 1 in https://arxiv.org/pdf/2104.08283
    r = randn(T, n) # 1
    svd_A = svd(tensordot(r, A)) # 2
    alpha3, alpha4 = conj(svd_A.U[:,1]), conj(svd_A.Vt[1,:])
    V3  = svd(tensordot(A, alpha4)).Vt[1:chi1,:]' # 3
    V4  = svd(tensordot(permutedims(A,[1,3,2]), alpha3)).Vt[1:chi2,:]' # 4
    B   = permutedims(tensordot(permutedims(tensordot(A, V4), [1,3,2]), V3), [1,3,2]) # 5 TODO optimize order
    @assert size(B) == (n, chi1, chi2)
    if transposeQ == nothing
        transposeQ = chi1 > chi2
    end
    Bt  = transposeQ ? permutedims(B, [1,3,2]) : B
    Ut  = reshape(orthogonalize!(flatten(conj(Bt), 1)), size(Bt)) # 6
    permutedims(Ut, transposeQ ? [3,2,1] : [2,3,1])
end

"""Compute the entanglement entropy of UA = tensordot(U, A)"""
# defined in equation (10) of https://arxiv.org/pdf/2104.08283
function entanglement(UA::AbstractArray{T,4}) where T
    lambdas  = svdvals(flatten(permutedims(UA, [1,3,2,4]), 2))
    ps   = lambdas .* lambdas
    ps ./= sum(ps)
    max(0., -dot(ps, log.(max.(ps, floatmin(real(T))))))
end

divUp(x,y) = div(x,y,RoundUp)

"""Convert an array to a matrix (or vector if n==0 or ndims(A)) by merging the first n and last ndims(A)-n indices."""
function flatten(A::AbstractArray{T,N}, n) where {T,N}
    n==0 || n==ndims(A) ? vec(A) : reshape(A, (prod(size(A)[1:n]), prod(size(A)[n+1:end])))
end

"""
Contract the last n indices of A with the first n indices of B.
This is similar to Python's numpy.tensordot, except n defaults to 1.
"""
function tensordot(A::AbstractArray{T,M}, B::AbstractArray{T,N}, n::Int=1) where {T,M,N}
    if n == ndims(A) # needed since 'vector * matrix' returns an error
        return dropdims(tensordot(reshape(A, (1, size(A)...)), B, n); dims=1)
    elseif n == 0
        return tensordot(reshape(A, (size(A)..., 1)), reshape(B, (1, size(B)...)), 1)
    end
    reshape(flatten(A, ndims(A)-n) * flatten(B, n), (size(A)[1:ndims(A)-n]..., size(B)[n+1:end]...))
end

"""
Gram-Schmidt orthonormalization of the rows of M.
Inserts random vectors in the case of linearly dependent rows.
"""
function orthogonalize!(M::Matrix{T}) where T
    epsMin = sqrt(eps(real(T))) # once eps0<epsMin, we add random vectors if needed
    eps0 = 0.5sqrt(epsMin) # only accept new orthogonal vectors if their relative norm is at last eps0 after orthogonalization
    n,m = size(M)
    @assert n >= m
    norms = [norm(M[:,i]) for i in 1:m]
    maxNorm = maximum(norms)
    orthogQ = zeros(Bool, m)
    allOrthog = false
    while !allOrthog
        allOrthog = true
        eps1 = max(eps0, epsMin)
        for i in 1:m
            if !orthogQ[i]
                if norms[i] > eps1 * maxNorm
                    Mi = M[:,i]
                    for j in 1:m
                        if orthogQ[j]
                            Mi = Mi - M[:,j] * dot(M[:,j], Mi)
                        end
                    end
                    normMi = norm(Mi)
                    if normMi > eps1 * norms[i]
                        M[:,i] = Mi / normMi
                        orthogQ[i] = true
                        continue
                    end
                end
                # M[:,i] was a linear combination of the other vectors
                if eps0 < epsMin
                    M[:,i]   = maxNorm * normalize(randn(T, n))
                    norms[i] = maxNorm
                end
                allOrthog = false
            end
        end
        eps0 = eps0*eps0
    end
    if norm(M' * M - I) > eps(real(T)) ^ 0.75
        return orthogonalize!(M)
    end
    M
end

# verification code

"""Check that the ansatz in equation (1) of https://arxiv.org/pdf/2104.08283 results in the minimal entanglement entropy."""
function checkAnsatz(chi1::Int, chi2::Int, chi3a::Int, chi4b::Int, chi3c::Int, chi4c::Int,
                     eps0::T=ComplexF64(0)) where T
    M1,M2,M3 = randn(T,chi1,chi3a), randn(T,chi2,chi4b), randn(T,chi3c,chi4c)
    A = reshape(permutedims(tensordot(tensordot(M1, M2, 0), M3, 0), (1,3,2,5,4,6)),
                (chi1*chi2, chi3a*chi3c, chi4b*chi4c))
    A = A + eps0 * norm(A) * normalize(randn(T,size(A)...))
    U = fastDisentangle(chi1,chi2,A)
    return entanglement(tensordot(U,A)) - entanglement(reshape(M3,(1,1,chi3c,chi4c)))
end

"""repeatedly check the ansatz"""
function checkAnsatzRepeated(maxChi::Int=9)
    c = 0
    BLAS.set_num_threads(1) # this seems to increase performance significantly
    while true
        c += 1
        if c%1000 == 0
            println(c)
        end
        chis = rand(1:maxChi, 6)
        chi1,chi2,chi3a,chi4b,chi3c,chi4c = chis
        chi3 = chi3a*chi3c
        chi4 = chi4b*chi4c
        eps0 = 10^(-6 - 14*rand(Float64))
        complexQ = rand(Bool)
        if (chi1 <= chi3 && chi2 <= chi4) ||
           ((chi3c==1 || chi4c==1) && (chi2 <= divUp(chi4, divUp(chi1,chi3)) ||
                                       chi1 <= divUp(chi3, divUp(chi2,chi4))))
            args = (chis..., complexQ ? ComplexF64(eps0) : eps0)
            S = checkAnsatz(args...)
            if S > 0.1sqrt(max(eps0, eps(Float64)))
                println(args, " -> ", S)
                return
            end
        end
    end
end
