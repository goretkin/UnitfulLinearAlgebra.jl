using Test
using StaticArrays
using Unitful

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
  @test taa_UX[1] == u"m"
  @test taa_UX[2] == u"m/s"
end

top1 = TagsOuterProduct{Tuple{taa_UX, }}()
top2 = TagsOuterProduct{Tuple{taa_UXdot, taa_UXinv}}()

@testset "TagsOuterProduct" begin
  @test top1[1] == u"m"
  @test top2[1,1] == u"s^-1"
end

@testset "construct TaggedTensor" begin
  t1 = TTensor(top1,
    @SArray rand(2,)
  )

  t2 = TTensor(top2,
    @SArray rand(2,2)
  )

  t = t2 * t1
  @test t.x == t2.x * t1.x
  @test get_tag(t) == TagsOuterProduct{Tuple{taa_UXdot}}()
end
