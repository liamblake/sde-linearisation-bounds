using LinearAlgebra

using DifferentialEquations
using Parameters

include("models.jl")

"""
	star_grid(x::AbstractVector, δx::Float64)::Array

Construct a star grid around a point x, for calculating a finite difference 
approximation of a spatial derivative. The number of points in the grid is 
determined from the dimension of x; if the length of x is n, then 2n points are
calculated.
"""
function star_grid(x::AbstractVector, δx::Float64)::Array{Float64}
    n = length(x)

    # Matrix of coordinate shifts
    # ⌈1  0  0  0  ⋯ 0⌉
    # |-1 0  0  0  ⋯ 0|
    # |0  1  0  0  ⋯ 0|
    # |0  -1 0  0  ⋯ 0|
    # |⋮     ⋱       ⋮|
    # |0  ⋯     0  ⋯ 1|
    # ⌊0  ⋯     0  ⋯ 0⌋
    A = zeros(2 * n, n)
    A[1:(2*n+2):(2*n^2)] .= 1
    A[2:(2*n+2):(2*n^2)] .= -1

    return repeat(x', 2 * n) + δx * A
end

"""
    ∇F(star_values, δx)

Approximate the flow map gradient with a centered finite-difference approximation, given a star grid of values.
"""
function ∇F(star_values::AbstractArray, n::UInt8, δx::Float64)
    return 1 / (2 * δx) * (star_values[1:2:(2*n), :] - star_values[2:2:(2*n), :])'
end

"""
∇F_eov!(dest::AbstractVector, ∇u::Function, d::UInt8, t₀::Real, T::Real, dt::Real)

Calculate the flow map gradient by directly solving the equation of variations,
given the corresponding trajectory. The equation of variations is
	∂∇F/∂t = ∇u(F(t), t)∇F.

The gradient of the flow map is taken with respect to the initial condition at time
t₀. A vector of matrices, corresponding to the flow map gradient at time steps at dt,
is placed in the preallocated dest vector.

"""
function ∇F_eov!(dest::AbstractVector, ∇u::Function, d::UInt8, t₀::Real, T::Real, dt::Real)
    # Inplace definition of the ODE 
    function rate!(dx::Matrix{Float64}, x::Matrix{Float64}, _, t)
        dx[:, :] = ∇u(t) * x
        return nothing
    end

    u₀ = zeros(d, d)
    u₀[diagind(u₀)] .= 1.0

    prob = ODEProblem(rate!, u₀, (t₀, T))
    dest[:] = solve(prob, saveat = dt).u

end

"""
	Σ_calculation

Calculate the deviation covariance matrix Σ with an in-place specification of the velocity field.
"""
function Σ_calculation(
    model::Model,
    x₀::AbstractVector,
    t₀::Real,
    T::Real,
    dt::Real,
)::Symmetric{Float64}
    @unpack d, velocity!, ∇u = model

    ts = t₀:dt:T
    # Generate the required flow map data
    # First, advect the initial condition forward to obtain the final position
    prob = ODEProblem(velocity!, x₀, (t₀, T))
    det_sol = solve(prob, Euler(), dt = dt)

    ∇u_F = t -> ∇u(det_sol(t), t)

    # Calculate the flow map gradients by solving the equation of variations directly
    ∇Fs = Vector{Matrix{Float64}}(undef, length(ts))
    ∇F_eov!(∇Fs, ∇u_F, d, t₀, T, dt)
    # ∇F_sol = ∇F_eov(∇u_F, d, t₀, T)
    # ∇Fs = ∇F_sol.(ts)

    # Form the star grid around the final position
    # star = star_grid(w, dx)

    # # Advect these points backwards to the initial time
    # prob = ODEProblem(velocity!, star[1, :], (T, t₀))
    # ensemble =
    #     EnsembleProblem(prob, prob_func=(prob, i, _) -> remake(prob, u0=star[i, :]))
    # sol =
    #     solve(ensemble, Euler(), EnsembleThreads(), dt=dt, trajectories=2 * d)

    # # Permute the dimensions of the ensemble solution so star_values is indexed
    # # as (timestep, gridpoint, coordinate).
    # star_values = Array{Float64}(undef, nₜ + 1, 2 * d, d)
    # permutedims!(star_values, Array(sol), [2, 3, 1])

    # # Approximate the flow map gradient at each time step 
    # ∇Fs = ∇F.(eachslice(star_values, dims=1), d, dx)

    # TODO: Work with non-identity σ
    Ks = [last(∇Fs)] .* inv.(∇Fs)
    integrand = Ks .* transpose.(Ks)

    # Approximate an integral given discrete evaluations, using the composite Simpson's rule
    # Assume the data is equidistant with respect to the variable being integrate.
    feven = @view integrand[2:2:end]
    fodd = @view integrand[1:2:end]

    Σ = dt / 3 * (integrand[1] + 2 * sum(feven) + 4 * sum(fodd) + last(integrand))

    # Tell Julia that the matrix is symmetric, to optimise operations like calculating eigenvalues. 
    return Symmetric(Σ)
end

"""
	bdg_constant(p::Real)::Real

Calculate the upper bound constant in the Burkholder-Davis-Gundy inequality.
"""
function bdg_constant(p::Real)::Real
    if p ≤ 0
        throw(DomainError())
    elseif p < 1
        return (16 / p)^p
    else
        return 2^(2 * p^2) * p^(2 * p^2 + p) * (2 * p - 1)^(p - 2 * p^2)
    end
end


function Dᵣ(r, n, t, K_u, K_σ)
    return 6^(r - 1) *
           K_u^r *
           K_σ^r *
           t^(2 * r) *
           n^(3 * r / 2) *
           bdg_constant(r) *
           (
               n^(r / 2) * K_σ^r * exp(2^(2 * r - 1) * K_u^(2 * r) * t^(2 * r)) +
               bdg_constant(r / 2) * exp(2^(r - 1) * K_u^r * t^r)
           ) *
           exp(3^(r - 1) * t^r * K_u^r)
end


function log_Dᵣ(r, n, t, K_u, K_σ)
    return (r - 1) * log(6) +
           5 * r / 2 * log(n) +
           r * log(K_u) +
           r * log(K_σ) +
           r * log(t) +
           log(bdg_constant(r)) +
           (2^(2 * r - 1) * K_u^(2 * r) * t^(2 * r)) +
           log(t^r * n^r + bdg_constant(r / 2)) +
           (3^(r - 1) * t^r * K_u^r)
end
