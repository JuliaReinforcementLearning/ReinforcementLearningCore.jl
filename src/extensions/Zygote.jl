export clip_by_global_norm, global_norm

using Zygote

Zygote.@adjoint argmax(xs; dims = :) = argmax(xs; dims = dims), _ -> nothing

global_norm(gs::Zygote.Grads) = sqrt(sum(mapreduce(x->x^2, +, gs[p]) for p in gs.params.params))

function clip_by_global_norm(gs::Zygote.Grads, clip_norm::Float32)
    gn = global_norm(gs)
    if clip_norm <= gn
        for p in gs.params.params
            gs[p] .*= clip_norm / max(clip_norm, gn)
        end
    end
    gs
end