# TODO Tag can maybe be any Unitful.Units (ContextUnits, FixedUnits, FreeUnits)
struct TagsAlongAxis{T} <: AbstractVector{FreeUnits}
end

# TODO restrict T to be tuple?
Base.firstindex(taa::TagsAlongAxis{T}) where {T} = Base.firstindex(T)
Base.lastindex(taa::TagsAlongAxis{T}) where {T} = Base.lastindex(T)
Base.size(taa::TagsAlongAxis{T}) where {T} = (length(T), )
Base.getindex(taa::TagsAlongAxis{T}, i) where {T} = T[i]



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

"""
Unitful.dimension(dot(dual(v), v)) == Unitful.NoDims
"""
function dual(v::TagsAlongAxis{UNITS}) where {UNITS}
  return TagsAlongAxis{map(inv, UNITS)}()
end