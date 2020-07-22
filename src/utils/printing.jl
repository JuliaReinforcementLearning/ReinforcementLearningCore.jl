export StructTree

using AbstractTrees

const AT = AbstractTrees

struct StructTree{X}
    x::X
end

AT.children(t::StructTree{X}) where X = Tuple(f => StructTree(getfield(t.x, f)) for f in fieldnames(X))
AT.children(t::StructTree{<:AbstractArray}) = ()
AT.children(t::Pair{Symbol, <:StructTree}) = children(last(t))
AT.printnode(io::IO, t::StructTree) = summary(io, t.x)

AT.printnode(io::IO, t::StructTree{<:Number}) = print(io, t.x)

function AT.printnode(io::IO, t::StructTree{String})
    s = t.x
    i = findfirst('\n', s)
    if isnothing(i)
        if length(s) > 79
            print(io, "\"s[1:79]...\"")
        else
            print(io, "\"$s\"")
        end
    else
        if i > 79
            print(io, "\"s[1:79]...\"")
        else
            print(io, "\"$(s[1:i])...\"")
        end
    end
end

function AT.printnode(io::IO, t::Pair{Symbol, <:StructTree})
    print(io, first(t), " => ")
    AT.printnode(io, last(t))
end
