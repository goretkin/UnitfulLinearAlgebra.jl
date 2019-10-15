module UnitfulLinearAlgebra

using Unitful: Unitful, @u_str, FreeUnits, unit
using LinearAlgebra: dot, AdjOrTrans

include("unitful.jl")

# TODO Tag can maybe be any Unitful.Units (ContextUnits, FixedUnits, FreeUnits)
struct TagsAlongAxis{T} <: AbstractVector{FreeUnits}
end

# TODO restrict T to be tuple?
Base.firstindex(taa::TagsAlongAxis{T}) where {T} = Base.firstindex(T)
Base.lastindex(taa::TagsAlongAxis{T}) where {T} = Base.lastindex(T)
Base.size(taa::TagsAlongAxis{T}) where {T} = (length(T), )
Base.getindex(taa::TagsAlongAxis{T}, i) where {T} = T[i]


import LinearAlgebra

# generic fallback won't work unless we define `iterate` on `typeof(u"m")`.
# u"m" is an object which supports only `:*`. `u"m" + u"cm"` has no meaning.
# to see if two `TagsAlongAxis` are compatible for contraction, give them units and check that.
function LinearAlgebra.dot(v1::TagsAlongAxis{UNITS1}, v2::TagsAlongAxis{UNITS2}) where {UNITS1, UNITS2}
  @assert length(UNITS1) == length(UNITS2)
  u1_one = map(u->true * u, UNITS1)
  u2_one = map(u->true * u, UNITS2)

  s = u1_one[1] * u2_one[1]
  for i = 2:length(UNITS1)
    s += u1_one[i] * u2_one[i]
  end
  return unit(s)
end


function Base.:*(a::FreeUnits, v::TagsAlongAxis{UNITS}) where {UNITS}
  return TagsAlongAxis{map(u -> a * u, UNITS)}()
end


function Base.:*(v::TagsAlongAxis{UNITS}, a::FreeUnits) where {UNITS}
  # field multiplication should be commutative, but maintain multipliation order anyway.
  return TagsAlongAxis{map(u -> u * a, UNITS)}()
end

function Base.:/(v::TagsAlongAxis{UNITS}, a::FreeUnits) where {UNITS}
  return TagsAlongAxis{map(u -> u / a, UNITS)}()
end


#---
# TODO alternative: Base.length(::Type{<:Tuple{Vararg{<:Any,N}}}) where {N} = N (thanks Yingbo Ma)
Base.length(t::Type{T}) where {T<:Tuple} = length(T.parameters)
#---

struct TagsOuterProduct{T, N} <: AbstractArray{FreeUnits, N}
end

function TagsOuterProduct(tag)
  N = length(tag)
  return TagsOuterProduct{tag, N}()
end

Base.firstindex(::TagsOuterProduct{TAG}) where {TAG} = 1
Base.lastindex(top::TagsOuterProduct{TAG}) where {TAG} = length(top)
Base.size(::TagsOuterProduct{TAG}) where {TAG} = map(length, Tuple_to_tuple(TAG))
# TODO overload show on TagsOuterProduct to make sure `NoDims` prints something instead of nothing

function Base.getindex(top::TagsOuterProduct{Tuple{TAA}}, ind) where {TAA}
  return getindex(TAA, ind)
end


# TODO define recursively, dispatch on Tuple{TAA1, Vararg}
function Base.getindex(top::TagsOuterProduct{Tuple{TAA1, TAA2}}, ind1, ind2) where {TAA1, TAA2}
  return getindex(TAA1, ind1) * getindex(TAA2, ind2)
end

adj_or_trans_parent(v::LinearAlgebra.Adjoint) = adjoint(v)
adj_or_trans_parent(v::LinearAlgebra.Transpose) = transpose(v)

# define outer product
function Base.:*(u::TagsAlongAxis, v::AdjOrTrans{T, <:TagsAlongAxis}) where {T}
  return TagsOuterProduct(Tuple{u, adj_or_trans_parent(v)})
end

# TODO would `normalize` be a pun on the generic function?
function canonicalize(m::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
  a = TAA1[1]
  return (TAA1 * inv(a))  * (TAA2 * a)'
end

# matrix-vector product
function Base.:*(m::TagsOuterProduct{Tuple{mTAA1, mTAA2}}, v::TagsOuterProduct{Tuple{vTAA}}) where {mTAA1, mTAA2, vTAA}
  taa = mTAA1 * dot(mTAA2, vTAA)
  return TagsOuterProduct(Tuple{taa})
end

# matrix-matrix product
function Base.:*(m::TagsOuterProduct{Tuple{m1TAA1, m1TAA2}}, v::TagsOuterProduct{Tuple{m2TAA1, m2TAA2}}) where {m1TAA1, m1TAA2, m2TAA1, m2TAA2}
  # parentheses necessary to avoid infinite recursion.
  return m1TAA1 * (m1TAA2' * m2TAA1) * m2TAA2'
end

# matrix-scalar product
function Base.:*(m::TagsOuterProduct{Tuple{mTAA1, mTAA2}}, s::FreeUnits) where {mTAA1, mTAA2}
  return mTAA1 * (mTAA2 * s)'
end

# scalar-matrix product
# preserves `mTAA1[1] == 1`
function Base.:*(s::FreeUnits, m::TagsOuterProduct{Tuple{mTAA1, mTAA2}}) where {mTAA1, mTAA2}
  return mTAA1 * (mTAA2 * s)'
end

# integer power. Generic fallbacks use `power_by_squaring`, which assumes type closure under `*`, which is a property that Unitful matrices do not have.
# TODO factorize so that this can be a O(1) operation
function Base.:^(m::TagsOuterProduct{Tuple{TAA1, TAA2}}, pow::Integer) where {TAA1, TAA2}
  @assert pow > 0 # TODO generalize for pow <= 0
  acc = m
  for i = 2:pow
    acc = acc * m
  end
  return acc
end

"""
Definition 3.5 from Hart Multidimensional Analysis
"""
function is_endomorphic(m::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
  return canonicalize(m) == canonicalize(m^2)
end

# TODO restrict TT to TT<:TagsOuterProduct
struct TaggedTensor{T, N, A<:AbstractArray, TT} <: AbstractArray{T, N}
  x::A
end

# TODO put in inner constructor
function TTensor(tag, x)
  # TODO assert tag and x have the same size
  @assert eltype(tag) == Unitful.FreeUnits # TODO generalize
  T = Unitful.Quantity{eltype(x),D,U} where U where D # TODO specialize U
  TaggedTensor{T, ndims(x), typeof(x), tag}(x)
end

get_tag(::TaggedTensor{T, N, A, TAG}) where {T, N, A, TAG} = TAG
get_Tuple(::TagsOuterProduct{TUPLE}) where {TUPLE} = TUPLE

Tuple_to_tuple(::Type{TUPLE}) where {TUPLE<:Tuple} = tuple(TUPLE.parameters...)

Base.size(t::TaggedTensor) = size(get_tag(t))
Base.getindex(t::TaggedTensor, ind...) = t.x[ind...] * get_tag(t)[ind...]
#get_freeunits(quantity::Unitful.Quantity{T, Dim, Unit}) = Unit


function Base.:*(t1::TaggedTensor, t2::TaggedTensor)
  TAG1 = get_tag(t1)
  TAG2 = get_tag(t2)
  TAG = TAG1 * TAG2
  t = t1.x * t2.x

  return TTensor(TAG, t)
end

end # module
