using Unitful: unit, FreeUnits
using LinearAlgebra: LinearAlgebra, dot
# TODO consider just going this instead of defining dot product.
# then generic dot product fallback should work.
# figure out what mathematical object this would imply
# (structure is different from addition. no zero, no negative (no identity, no inverse))
# Base.:+(a::FreeUnits, b::FreeUnits) = unit(true * a + true * b)

Base.adjoint(x::FreeUnits) = x
Base.transpose(x::FreeUnits) = x

# TODO: alternatively, define `iterate` for `FreeUnits`, just like it is for `Number` This would be consistent with Julia's design.
LinearAlgebra.dot(a::FreeUnits, b::FreeUnits) = a * b
