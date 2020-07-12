
"""
    MultiGPInput(x::AbstractVector, out_dim::Integer)

A data type to accomodate modelling multi-dimensional output data.

"""
struct MultiGPInput{T<:AbstractVector}
    x::T
    out_dim::Integer
end

Base.length(inp::MultiGPInput) = inp.out_dim * length(inp.x)

Base.size(inp::MultiGPInput, d) = d::Integer == 1 ? inp.out_dim * size(inp.x, 1) : 1 
Base.size(inp::MultiGPInput) = (inp.out_dim * size(inp.x, 1),)

Base.lastindex(inp::MultiGPInput) = length(inp)
Base.firstindex(inp::MultiGPInput) = 1

function Base.getindex(inp::MultiGPInput, ind::Integer)
    if ind > 0
        out_dim = ind ÷ length(inp.x) + 1
        ind = ind % length(inp.x)
        if ind==0 ind = length(inp.x); out_dim-=1 end
        return (inp.x[ind], out_dim)
    else
        return BoundsError(string("Trying to access at ", ind))
    end
end

Base.iterate(inp::MultiGPInput) = (inp[1], 1)
Base.iterate(inp::MultiGPInput, state) = (state<length(inp)) ? (inp[state + 1], state + 1) : nothing

"""
    MultiFiniteGP{Tf<:AbstractGP, Tx<:MultiGPInput, TΣy} 

The finite-dimentional projection of the AbstractGP `f` at `x`. Here, we model a 
multi-output GP.  

"""
struct MultiFiniteGP{Tf<:AbstractGP, Tx<:MultiGPInput, TΣy} 
    f::Tf
    x::Tx
    Σy::TΣy
end

MultiFiniteGP(f::AbstractGP, x::MultiGPInput) = MultiFiniteGP(f, x, 1e-18)

MultiFiniteGP(f::AbstractGP, x::AbstractVector; out_dim) = MultiFiniteGP(f, MultiGPInput(x, out_dim), 1e-18)

(f::AbstractGP)(x::MultiGPInput) = MultiFiniteGP(f, x)
