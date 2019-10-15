using Test
using StaticArrays
using Unitful
import LinearAlgebra

import UnitfulLinearAlgebra
using UnitfulLinearAlgebra: TagsAlongAxis, TagsOuterProduct, TaggedTensor, TTensor, get_tag, canonicalize

@testset "all" begin

UX = (u"m", u"m/s")
UXdot = UX ./ u"s"

UXinv = map(inv, UX)

TX = map(typeof, UX)
TXdot = map(typeof, UXdot)

taa_UX = TagsAlongAxis{UX}()
taa_UXinv = TagsAlongAxis{UXinv}()
taa_UXdot = TagsAlongAxis{UXdot}()

@testset "TagsAlongAxis" begin
  @test size(taa_UX) == (2, )
  @test taa_UX[1] == u"m"
  @test taa_UX[2] == u"m/s"

  @test 2 * (u"m" * taa_UXdot) == (2 * u"m") * taa_UXdot
  @test typeof(u"m" * taa_UXdot) <: TagsAlongAxis
end

top1 = TagsOuterProduct(Tuple{taa_UX, })
top2 = TagsOuterProduct(Tuple{taa_UXdot, taa_UXinv})

@testset "TagsOuterProduct" begin
  @test top1[1] == u"m"
  @test top2[1,1] == u"s^-1"
end

@testset "outer product construction" begin
  top2_ = taa_UXdot * taa_UXinv'
  @test top2_ === top2
end

t1 = TTensor(top1,
  @SArray rand(2,)
)

t2 = TTensor(top2,
  @SArray rand(2,2)
)

t2_mutable = TTensor(top2,
  rand(2,2)
)

@testset "construct TaggedTensor" begin
  t = t2 * t1
  @test t.x == t2.x * t1.x
  @test get_tag(t) == TagsOuterProduct(Tuple{taa_UXdot})
end

@testset "mutate" begin
  @test_throws Unitful.DimensionError t2_mutable[1,1] = 1.0
  t2_mutable[1,1] = 1 * u"s^-1"
  @test t2_mutable[1,1] == 1 * u"s^-1"
  t2_mutable[1,1] = 120 * u"minute^-1"
  @test t2_mutable[1,1] == 2 * u"s^-1"
end

@testset "adjoint" begin
  @test (t1' * t2')' == (t2 * t1)
end

@testset "continuous-time LTI system" begin
  @test UnitfulLinearAlgebra.is_endomorphic(top2) == false
  @test UnitfulLinearAlgebra.is_endomorphic(top2 * u"s") == true
  # TODO decide if we want this behavior
  @test UnitfulLinearAlgebra.is_endomorphic(t2 * 1u"hr") == true
end

SHO_A = TTensor(top2,
  @SArray [
    +0.0 +1.0;
    -1.0 +0.0
  ]
)

top3 = TagsOuterProduct(Tuple{taa_UX, taa_UXinv})

Identity_A = TTensor(top3,
  @SArray [
    1 0;
    0 1
  ]
)

@testset "linear algebra" begin
  @test one(t2) * t2 == t2
  @test t2 * one(t2) == t2
  @test t2 + zero(t2) == t2

  @test canonicalize(top3) === canonicalize(top2 * u"s")
  @test LinearAlgebra.det(SHO_A) == +1.0*u"s^-2"
  @test exp(SHO_A * 2π*u"s") ≈ Identity_A
end

end