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
function power(m::TagsOuterProduct{Tuple{TAA1, TAA2}, 2}, p::Real) where {TAA1, TAA2}
  @assert p >= 0 # TODO generalize for pow < 0
  acc = one(m)
  for i = 1:p
    acc = acc * m
  end
  return acc
end

Base.:^(m::TagsOuterProduct{Tuple{TAA1, TAA2}, 2}, p::Real) where {TAA1, TAA2} = power(m, p)

# avoid method ambiguity with `^(A::AbstractArray{T,2} where T, p::Integer) in LinearAlgebra`
Base.:^(m::TagsOuterProduct{Tuple{TAA1, TAA2}, 2}, p::Integer) where {TAA1, TAA2} = power(m, p)

# make sure literal_pow also doesn't use fallback.
Base.literal_pow(::typeof(^), x::TagsOuterProduct{Tuple{TAA1, TAA2}, 2}, ::Val{p}) where {TAA1, TAA2, p} = x^p
# avoid method ambiguity
Base.literal_pow(::typeof(^), x::TagsOuterProduct{Tuple{TAA1, TAA2}, 2}, ::Val{-1}) where {TAA1, TAA2, p} = power(x, -1)  # TODO avoid naming

"""
Definition 3.5 from Hart Multidimensional Analysis
"""
function is_endomorphic(m::TagsOuterProduct)
  return canonicalize(m) == canonicalize(m^2)
end

"""
Factor m = c * e where e is endomorphic, if possible
return (taa, c, is_endomorphic(e)) where m = c * e
"""
# TODO: this kind of looks like an eigen decomposition: TAA * c * dual(TAA)'
function factor_endomorphic(m::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
  if canonicalize(m) != m
    return factor_endomorphic(canonicalize(m))
  end

  d = dual(TAA2) ./ TAA1
  maybe_factor = first(d)
  if all(maybe_factor .== d)
    @assert dual(TAA2) == TAA1 * maybe_factor
    (TAA1 * dual(TAA1)', inv(maybe_factor), true)
  else
    (m, Unitful.FreeUnits{(),Unitful.NoDims,nothing}(), false)
  end
end


"""
oneleft(x) * x = x
"""
function oneleft(::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
  return TAA1 *  dual(TAA1)'
end

"""
x * oneright(x) = x
"""
function oneright(::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
  return dual(TAA2) * TAA2'
end

struct NoCommutativeIdentityElement{E} <: Exception
  element::E
end

Base.showerror(io::IO, e::NoCommutativeIdentityElement) = print(io, e.element, " does not have an identity element. Try `oneright` or `oneleft`")

"""
Only defined for c * e where `is_endomorphic(e)` is true and c is a scalar.
"""
function Base.one(m::TagsOuterProduct)
  if canonicalize(oneright(m)) == canonicalize(oneleft(m))
    return canonicalize(oneright(m))
  end
  throw(NoCommutativeIdentityElement(m))
end


# inv(m) * m ≠ m * inv(m) in general
function Base.inv(::TagsOuterProduct{Tuple{TAA1, TAA2}}) where {TAA1, TAA2}
  dual(TAA2) * dual(TAA1)'
end