using LinearAlgebra: BlasInt, checknonsingular, LU

function my_generic_lufact!(A::StridedMatrix{ET}, ::Val{Pivot} = Val(true);
                         check::Bool = true) where {ET,Pivot}
    NET = Float64
    m, n = size(A)
    minmn = min(m,n)
    info = 0
    ipiv = Vector{BlasInt}(undef, minmn)
    @inbounds begin
        for k = 1:minmn
            @show k
            #println("A"); display(A)
            # find index max
            kp = k
            if Pivot
                amax = abs(zero(NET))
                for i = k:m
                    absi = abs(ustrip(A[i,k]))
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
                    println("A"); display(A)
                    for i = 1:n
                        println("loop swap row")
                        @show i
                        #println("A"); display(A)
                        tmp = A[k,i]
                        A[k,i] = A[kp,i]
                        A[kp,i] = tmp
                    end
                    println("A"); display(A)
                end
                # Scale first column.
                # if this were out-of-place LU, make A[k,k] unity
                Akkinv = inv(A[k,k])
                for i = k+1:m
                    println("loop scale row")
                    @show i
                    #println("A"); display(A)
                    A[i,k] *= Akkinv
                end
            elseif info == 0
                info = k
                @show info
            end
            # Update the rest
            # if this were out-of-place LU, zero out rest of row k.
            for j = k+1:n
                for i = k+1:m
                    println("loop subtract")
                    #@show i, j, k
                    #println("A"); display(A)
                    #@show A[k,j]
                    #@show A[i,k]
                    #@show A[i,j]
                    println("A[$i,$j] -= A[$i,$k]*A[$k,$j]")
                end
            end
        end
    end
    check && checknonsingular(info)
    return LU{ET,typeof(A)}(A, ipiv, convert(BlasInt, info))
end