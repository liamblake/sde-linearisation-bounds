using LinearAlgebra

using DifferentialEquations

include("models.jl")

"""
	star_grid(x::AbstractVector, δx::Float64)::Array

Construct a star grid around a point x, for calculating a finite difference 
approximation of a spatial derivative. The number of points in the grid is 
determined from the dimension of x; if the length of x is n, then 2n points are
calculated.
"""
function star_grid(x::AbstractVector, δx::Float64)::Array{Float64, }
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
	Σ_calculation

Calculate the deviation covariance matrix Σ with an in-place specification of the velocity field.
"""
function Σ_calculation(model::Model, dt::Float64, dx::Float64)::Symmetric
    nₜ = Int64(fld(model.T - model.t₀, dt))
    # Generate the required flow map data
    # First, advect the initial condition forward to obtain the final position
    prob = ODEProblem(model.velocity!, model.x₀, (model.t₀, model.T))
    det_sol = solve(prob, Euler(), dt=dt)
	w = last(det_sol)::Vector{Float64}
    # Fs = reverse(det_sol.u)

    # Form the star grid around the final position
    star = star_grid(w, dx)

    # Advect these points backwards to the initial time
    prob = ODEProblem(model.velocity!, star[1, :], (model.T, model.t₀))
    ensemble =
        EnsembleProblem(prob, prob_func=(prob, i, _) -> remake(prob, u0=star[i, :]))
    sol =
        solve(ensemble, Euler(), EnsembleThreads(), dt=dt, trajectories=2 * model.d)

    # Rearrange the ensemble solution 
    star_values = [
        reduce(
            hcat,
            DifferentialEquations.EnsembleAnalysis.componentwise_vectors_timestep(
                sol,
                i,
            ),
        ) for i = 1:(nₜ+1)
    ]

    # Approximate the flow map gradient at each time step 
    ∇Fs = ∇F.(star_values, model.d, dx)

    # TODO: Work with non-identity σ
    Ks = inv.(∇Fs)
    integrand = Ks .* transpose.(Ks)

    # Approximate an integral given discrete evaluations, using the composite Simpson's rule
    # Assume the data is equidistant with respect to the variable being integrate.
    feven = @view integrand[2:2:end]
    fodd = @view integrand[1:2:end]

    Σ = dt / 3 * (integrand[1] + 2 * sum(feven) + 4 * sum(fodd) + last(integrand))

    # Tell Julia that the matrix is symmetric, to optimise operations like calculating eigenvalues. 
    return Symmetric(Σ)
end
