using OrdinaryDiffEq

include("models.jl")

"""
	star_grid(x, δx)

Construct a star grid around a point x, for calculating a finite difference
approximation of a spatial derivative. The number of points in the grid is
determined from the dimension of x; if the length of x is n, then 2n points are
calculated.
"""
function star_grid(x, δx)
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
    A[1:(2 * n + 2):(2 * n^2)] .= 1
    A[2:(2 * n + 2):(2 * n^2)] .= -1

    return repeat(x', 2 * n) + δx * A
end

"""
    ∇F(star_values, n, δx)

Approximate the flow map gradient with a centered finite-difference approximation, given a star grid of values.
"""
function ∇F(star_values, n, δx)
    return 1 / (2 * δx) * (star_values[1:2:(2 * n), :] - star_values[2:2:(2 * n), :])'
end

"""
    gaussian_computation(
        model,
        x₀,
        t₀,
        T,
        dt,
        dx,
        ode_solver;
        Σ₀ = zeros(model.d, model.d),
        ode_solver_kwargs...,
    )

Jointly compute the state variable and limiting covariance matrix for a given SDE model, initial
condition x₀, and time span (t₀, T).

"""
function gaussian_computation(
    model,
    x₀,
    t₀,
    T,
    dt,
    dx,
    ode_solver;
    Σ₀ = zeros(model.d, model.d),
    ode_solver_kwargs...,
)
    @unpack d, velocity, ∇u, σ = model

    ts = t₀:dt:T
    if last(ts) < T
        ts = range(t₀; stop = T, length = length(ts) + 1)
    end

    # Generate the required flow map data
    # First, advect the initial condition forward to obtain the final position
    prob = ODEProblem(velocity, x₀, (t₀, T))
    det_sol = solve(prob, ode_solver; dt = dt, dtmax = dt, saveat = ts)
    ws = Array(det_sol)

    # Form the star grid around the initial position
    star = star_grid(x₀, dx)

    # Advect these points forwards to the final time
    prob = ODEProblem(velocity, star[1, :], (t₀, T))
    ensemble = EnsembleProblem(prob; prob_func = (prob, i, _) -> remake(prob; u0 = star[i, :]))
    sol = solve(
        ensemble,
        ode_solver,
        EnsembleThreads();
        dt = dt,
        saveat = ts,
        trajectories = 2 * d,
        ode_solver_kwargs...,
    )

    # Permute the dimensions of the ensemble solution so star_values is indexed
    # as (timestep, gridpoint, coordinate).
    star_values = Array{Float64}(undef, length(sol[1]), 2 * d, d)
    permutedims!(star_values, Array(sol), [2, 3, 1])

    # Approximate the flow map gradient at each time step
    ∇Fs = ∇F.(eachslice(star_values; dims = 1), d, dx)

    # Evaluate σ along the trajectory
    σs = σ.(det_sol.(ts), nothing, ts)

    # Evaluate the integrand at each time point.
    # See Theorem 2.2 for the integral form of Σ.
    Ks = inv.(∇Fs) .* σs
    integrand = Ks .* transpose.(Ks)

    # Approximate the time integral with the trapezoidal rule.
    Σs = Vector{Matrix}(undef, length(ts))
    Σs[1] = Σ₀
    prev = Σ₀
    for i in 2:length(ts)
        # The integral itself
        prev += 0.5 * dt * (integrand[i] + integrand[i + 1])
        # Pre- and post-multiply by the final time flow map gradient gradient
        Σs[i + 1] = ∇Fs[i + 1] * prev * transpose(∇Fs[i + 1])
    end

    return ws, Σs
end