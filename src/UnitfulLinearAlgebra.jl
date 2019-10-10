module UnitfulLinearAlgebra

using Unitful: @u_str, FreeUnits, unit
using LinearAlgebra: dot

# We want TagsAlongAxis to act like an AbstactVector
# TagsAlongAxis{T} <: AbstractVector{Unitful.FreeUnits} means that *instances* of TagsAlongAxis act like an AbstractVector. Not what we want.
struct TagsAlongAxis{T}
end

# TODO also define `lastindex`
# TODO restrict T to be tuple?
Base.getindex(taa::Type{TagsAlongAxis{T}}, i) where {T} = T[i]
Base.iterate(taa::Type{TagsAlongAxis{T}}, args...) where {T} = iterate(T, args...)
Base.length(taa::Type{TagsAlongAxis{T}}) where {T} = length(T)

import LinearAlgebra

# generic fallback won't work unless we define `iterate` on `typeof(u"m")`.
# u"m" is an object which supports only `:*`. `u"m" + u"cm"` has no meaning.
# to see if two `TagsAlongAxis` are compatible for contraction, give them units and check that.
function LinearAlgebra.dot(v1::Type{TagsAlongAxis{UNITS1}}, v2::Type{TagsAlongAxis{UNITS2}}) where {UNITS1, UNITS2}
  @assert length(UNITS1) == length(UNITS2)
  u1_one = map(u->true * u, UNITS1)
  u2_one = map(u->true * u, UNITS2)

  s = u1_one[1] * u2_one[1]
  for i = 2:length(UNITS1)
    s += u1_one[i] * u2_one[i]
  end
  return unit(s)
end

# TODO consider using (Holy-) Traits
Multipliable = Union{Number, FreeUnits}

function Base.:*(a::Multipliable, v::Type{TagsAlongAxis{UNITS}}) where {UNITS}
  return TagsAlongAxis{map(u -> a * u, UNITS)}
end


function Base.:*(v::Type{TagsAlongAxis{UNITS}}, a::Multipliable) where {UNITS}
  # field multiplication should be commutative, but maintain multipliation order anyway.
  return TagsAlongAxis{map(u -> u * a, UNITS)}
end

struct TagsOuterProduct{T}
end

#TODO types vs singletons. Can dispatch on types.
function Base.getindex(top::Type{TagsOuterProduct{Tuple{TAA}}}, ind) where {TAA}
  return getindex(TAA, ind)
end


# TODO define recursively, dispatch on Tuple{TAA1, VarArgs}
function Base.getindex(top::Type{TagsOuterProduct{Tuple{TAA1, TAA2}}}, ind1, ind2) where {TAA1, TAA2}
  return getindex(TAA1, ind1) * getindex(TAA2, ind2)
end


# TODO restrict TT to TT<:TagsOuterProduct
struct TaggedTensor{T, N, A<:AbstractArray, TT} <: AbstractArray{T, N}
  x::A
end

# TODO put in inner constructor
function TTensor(tag, x)
  TaggedTensor{eltype(x), ndims(x), typeof(x), tag}(x)
end

get_tag(::TaggedTensor{T, N, A, TAG}) where{T, N, A, TAG} = TAG

Base.size(t::TaggedTensor) = size(t.x)
Base.getindex(t::TaggedTensor, ind...) = t.x[ind...] * get_tag(t)[ind...]
#get_freeunits(quantity::Unitful.Quantity{T, Dim, Unit}) = Unit

# matrix-vector product
function Base.:*(m::Type{TagsOuterProduct{Tuple{mTAA1, mTAA2}}}, v::Type{TagsOuterProduct{Tuple{vTAA}}}) where {mTAA1, mTAA2, vTAA}
  taa = mTAA1 * dot(mTAA2, vTAA)
  return TagsOuterProduct{Tuple{taa}}
end

# matrix-matrix product
function Base.:*(m::Type{TagsOuterProduct{Tuple{m1TAA1, m1TAA2}}}, v::Type{TagsOuterProduct{Tuple{m2TAA1, m2TAA2}}}) where {m1TAA1, m1TAA2, m2TAA1, m2TAA2}
  return m1TAA1 * dot(m1TAA2, m2TAA1) * m2TAA2
end

function Base.:*(t1::TaggedTensor, t2::TaggedTensor)
  TAG1 = get_tag(t1)
  TAG2 = get_tag(t2)
  TAG = TAG1 * TAG2
  t = t1.x * t2.x

  return TTensor(TAG, t)
end

end # module
