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
get_wrapped(m::TaggedTensor) = m.x
get_tag(q::Number) = Unitful.unit(q)
get_wrapped(q::Number) = Unitful.ustrip(q)
# TODO ???
get_tag(q::FreeUnits) = q
get_wrapped(q::FreeUnits) = true

get_Tuple(::TagsOuterProduct{TUPLE}) where {TUPLE} = TUPLE

Tuple_to_tuple(::Type{TUPLE}) where {TUPLE<:Tuple} = tuple(TUPLE.parameters...)

Base.size(t::TaggedTensor) = size(get_tag(t))
Base.getindex(t::TaggedTensor, ind...) = t.x[ind...] * get_tag(t)[ind...]
Base.setindex!(t::TaggedTensor, val, ind...) = setindex!(t.x, val / get_tag(t)[ind...], ind...)
#get_freeunits(quantity::Unitful.Quantity{T, Dim, Unit}) = Unit


# TODO rewrite as macro, unless constant propagation on `operation` and splatting is free.
function lower_operation(operation, ts...)
  tags = map(get_tag, ts)
  tag = operation(tags...)
  arrs = map(get_wrapped, ts)
  t = operation(arrs...)
  return TTensor(tag, t)
end


Multipliable = Union{Number, FreeUnits}
Base.:*(a::TaggedTensor, b::TaggedTensor) = lower_operation(*, a, b)
Base.:*(a::TaggedTensor, b::Multipliable) = lower_operation(*, a, b)
Base.:*(a::Multipliable, b::TaggedTensor) = lower_operation(*, a, b)
# for the sake of avoiding method ambiguity errors. not sure how to use Multipliable otherwise
Base.:*(a::TaggedTensor, b::Number) = lower_operation(*, a, b)
Base.:*(a::Number, b::TaggedTensor) = lower_operation(*, a, b)

const TaggedTensor2 = TaggedTensor{T, 2} where T

Base.one(t::TaggedTensor) = lower_operation(one, t)

constant_like(top, k) = TTensor(top, fill(k, size(top)))
Base.zero(t::TaggedTensor) = TTensor(get_tag(t), zero(get_wrapped(t)))

# TODO rewrite as macro, unless constant propagation on `operation` and splatting is free.
function lower_operation_endomorphic(operation, t)
  tag = canonicalize(get_tag(t))
  @assert is_endomorphic(tag)
  return TTensor(tag, operation(get_wrapped(t)))
end

Base.exp(t::TaggedTensor) = lower_operation_endomorphic(exp, t)
Base.sqrt(t::TaggedTensor) = lower_operation_endomorphic(exp, t)

# TODO maybe extend `lower_operation_endomorphic` to obviate this
power(t::TaggedTensor, p) = TTensor(get_tag(t)^p, get_wrapped(t)^p)
# method ambiguities
Base.:^(t::TaggedTensor2, p::Integer) = power(t, p)
Base.:^(t::TaggedTensor2, p::Real) = power(t, p)

Base.literal_pow(::typeof(^), t::TaggedTensor2, ::Val{p}) where p = power(t, p)

#canonicalize(t::TaggedTensor) = TTensor(canonicalize(get_tag(t)), t)

# This method definition in Base messes things up!
#isapprox(x::AbstractArray{S}, y::AbstractArray{T};
#    kwargs...) where {S <: AbstractQuantity,T <: AbstractQuantity} = false



