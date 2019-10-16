module UnitfulLinearAlgebra

using Unitful: Unitful, @u_str, FreeUnits, unit
using LinearAlgebra: dot, AdjOrTrans
import LinearAlgebra

include("unitful.jl")
include("linear_algebra.jl")
include("tags_along_axis.jl")
include("tags_outer_product.jl")
include("tagged_tensor.jl")

# define outer product
function Base.:*(u::TagsAlongAxis, v::AdjOrTrans{T, <:TagsAlongAxis}) where {T}
  return TagsOuterProduct(Tuple{u, adj_or_trans_parent(v)})
end

end # module
