@recipe function f(gp::AbstractGP, x::AbstractArray)
    Base.depwarn(
        "`plot(gp::AbstractGP, x::AbstractArray)` is deprecated, " *
        "use `plot(x, gp)` instead.",
        :apply_recipe,
    )
    return x, gp
end
@recipe function f(gp::AbstractGP, xmin::Real, xmax::Real)
    Base.depwarn(
        "`plot(gp::AbstractGP, xmin::Real, xmax::Real)` is deprecated, use " *
        "`plot(range(xmin, xmax; length=1_000), gp)` instead.",
        :apply_recipe,
    )
    return range(xmin, xmax; length=1_000), gp
end

@recipe function f(z::AbstractVector, gp::AbstractGP, x::AbstractArray)
    Base.depwarn(
        "`plot(z::AbstractVector, gp::AbstractGP, x::AbstractArray)` is deprecated, " *
        "use `plot(z, gp(x))` instead.",
        :apply_recipe,
    )
    return z, gp(x)
end
@recipe function f(z::AbstractVector, gp::AbstractGP, xmin::Real, xmax::Real)
    Base.depwarn(
        "`plot(z::AbstractVector, gp::AbstractGP, xmin::Real, xmax::Real)` is deprecated, " *
        "use `plot(z, gp(range(xmin, xmax; length=1_000))` instead.",
        :apply_recipe,
    )
    return z, gp(range(xmin, xmax; length=1_000))
end

@deprecate cov_diag(f::AbstractGP, x::AbstractVector) var(f, x)
@deprecate cov_diag(gp::FiniteGP) var(gp)
@deprecate mean_and_cov_diag(f::AbstractGP, x::AbstractVector) mean_and_var(f, x)
@deprecate mean_and_cov_diag(gp::FiniteGP) mean_and_var(gp)
