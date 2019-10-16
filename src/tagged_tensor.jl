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
Base.setindex!(t::TaggedTensor, val, ind...) = setindex!(t.x, val / get_tag(t)[ind...], ind...)
#get_freeunits(quantity::Unitful.Quantity{T, Dim, Unit}) = Unit


function Base.:*(t1::TaggedTensor, t2::TaggedTensor)
  TAG1 = get_tag(t1)
  TAG2 = get_tag(t2)
  TAG = TAG1 * TAG2
  t = t1.x * t2.x

  return TTensor(TAG, t)
end