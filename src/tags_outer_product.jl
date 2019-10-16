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

"""
oneleft(x) * x = x
"""
function oneleft(m::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
end

"""
x * oneright(x) = x
"""
function oneright(m::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
end