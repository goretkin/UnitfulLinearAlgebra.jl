using LinearAlgebra: BlasInt, checknonsingular, LU

function my_generic_lufact!(A::StridedMatrix{T}, ::Val{Pivot} = Val(true);
                         check::Bool = true) where {T,Pivot}
    m, n = size(A)
    minmn = min(m,n)
    info = 0
    ipiv = Vector{BlasInt}(undef, minmn)
    @inbounds begin
        for k = 1:minmn
            # find index max
            kp = k
            if Pivot
                amax = abs(zero(T))
                for i = k:m
                    absi = abs(A[i,k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            ipiv[k] = kp
            if !iszero(A[kp,k])
                if k != kp
                    # Interchange
                    for i = 1:n
                        tmp = A[k,i]
                        A[k,i] = A[kp,i]
                        A[kp,i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k,k])
                for i = k+1:m
                    A[i,k] *= Akkinv
                end
            elseif info == 0
                info = k
            end
            # Update the rest
            for j = k+1:n
                for i = k+1:m
                    A[i,j] -= A[i,k]*A[k,j]
                end
            end
        end
    end
    check && checknonsingular(info)
    return LU{T,typeof(A)}(A, ipiv, convert(BlasInt, info))
end