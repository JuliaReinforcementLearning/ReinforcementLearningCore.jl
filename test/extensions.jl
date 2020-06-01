@testset "Zygote" begin
grads = IdDict()
grads[:x] = [-3.0 0.0 0.0; 4.0 0.0 0.0]
gs = Zygote.Grads(grads, Zygote.Params([:x]))
@test isapprox(clip_by_global_norm(gs, 4.0f0)[:x], [-2.4 0.0 0.0; 3.2 0.0 0.0])

gs.grads[:x] = [1.0 0.0 0.0; 1.0 0.0 0.0]
@test isapprox(clip_by_global_norm(gs, 4.0f0)[:x], [1.0 0.0 0.0; 1.0 0.0 0.0])

gs.grads[:x] = [.0 0.0 0.0; .0 0.0 0.0]
@test isapprox(clip_by_global_norm(gs, 4.0f0)[:x], [.0 0.0 0.0; .0 0.0 0.0])
end