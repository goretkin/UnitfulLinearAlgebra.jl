using Test
using StaticArrays
using Unitful

import UnitfulLinearAlgebra
using UnitfulLinearAlgebra: TagsAlongAxis, TagsOuterProduct, TaggedTensor, TTensor, get_tag


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

@testset "construct TaggedTensor" begin
  t = t2 * t1
  @test t.x == t2.x * t1.x
  @test get_tag(t) == TagsOuterProduct(Tuple{taa_UXdot})
end

@testset "adjoint" begin
  @test (t1' * t2')' == (t2 * t1)
end

@testset "continuous-time LTI system"
  @test UnitfulLinearAlgebra.is_endomorphic(get_tag(t2)) == false
  @test UnitfulLinearAlgebra.is_endomorphic(t2 * 1u"s") == true
  # TODO decide if we want this behavior
  @test UnitfulLinearAlgebra.is_endomorphic(t2 * 1u"hr") == true
end