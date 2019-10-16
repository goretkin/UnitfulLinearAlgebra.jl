# Things that might be included in Base.LinearAlgebra

adj_or_trans_parent(v::LinearAlgebra.Adjoint) = adjoint(v)
adj_or_trans_parent(v::LinearAlgebra.Transpose) = transpose(v)