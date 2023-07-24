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
_solve_state_cov_forward!(
    state::AbstractVector,
    cov::AbstractMatrix,
    model::SDEModel,
    x₀::AbstractVector,
    Σ₀::AbstractMatrix,
    t₀::Float64,
    T::Float64,
)

Solve the joint system for the state and covariance, starting from an initial state x₀ and
covariance Σ₀ at time t₀, until stochastic sensitivity reaches a threshold ν or a time T is reached.
The joint equation is solved via the hybrid Taylor-Heun and Gauss-Legendre method proposed by
Mazzoni.

Arguments:
    state: A vector storing the state, for in-place computations and to store the final state.
    cov: A matrix storing the covariance matrix, for in-place computations and to store the final
         covariance matrix.
    model: Details of the SDE system, as a SDEModel instance.
    x₀: The initial state.
    Σ₀: The initial covariance matrix.
    t₀: The initial time.
    T: The final time, at which the state and covariance are returned if the threshold is not
       reached.
    dt: The time-step to use in the solver.

The resulting state and covariance are stored in the destination variables state and covm.

"""
function gaussian_computation!(
    state,
    covm,
    model::SDEModel,
    x₀::AbstractVector,
    Σ₀::AbstractMatrix,
    ts::AbstractVector,
)
    if !(size(state) == size(covm) == (length(ts),))
        throw(
            DimensionMismatch(
                "Destination vectors state and cov should have size $(size(ts)), but got $(size(state)) and $(size(covm)) respectively.",
            ),
        )
    end

    # Initialise
    state[1] = x₀
    covm[1] = Σ₀

    # Pre-allocate temporary variables to reuse
    ut = Vector{Float64}(undef, model.n)
    ∇ut = Matrix{Float64}(undef, model.n, model.n)
    Kτ = Matrix{Float64}(undef, model.n, model.n)

    # Iterate the Mazzoni hybrid scheme, until the S² threshold OR the final time T is reached.
    for (i, t) in enumerate(ts[2:end])
        dt = t - ts[i]

        # Update the state variable
        model.u!(ut, state[i], t)
        model.∇u!(∇ut, state[i], t)
        state[i + 1] = state[i] + dt * inv(I - ∇ut * dt / 2) * ut

        # Interpolate the state at time t + dt/2
        ut .= 0.5 .* (state[i] .+ state[i + 1] .- ∇ut * ut .* dt^2 / 4)

        # Compute the covariance
        # Compute ∇u(w_τ, t + dt/2) and store in ∇ut
        model.∇u!(∇ut, ut, t + dt / 2)
        Kτ .= inv(I - ∇ut * dt / 2)

        # Compute Mτ and store in ∇ut
        mul!(∇ut, Kτ, (I + ∇ut * dt / 2))
        covm[i + 1] = ∇ut * covm[i] * transpose(∇ut)

        # Compute [σσᵀ](w_τ, t + dt/2) and store in ∇ut
        model.σσᵀ!(∇ut, ut, t + dt / 2)
        covm[i + 1] = covm[i + 1] + Kτ * ∇ut * transpose(Kτ) .* dt

        # Correct any violations of symmetry in Σt.
        # These should only arise from floating point errors and therefore be small over a single
        # step. However, these errors can accumulate over many steps.
        covm[i + 1] = @. 0.5 * (covm[i + 1] + covm[i + 1]')
    end
end

# """
#     gaussian_computation(
#         model,
#         x₀,
#         t₀,
#         T,
#         dt,
#         dx,
#         ode_solver;
#         Σ₀ = zeros(model.d, model.d),
#         ode_solver_kwargs...,
#     )

# Jointly compute the state variable and limiting covariance matrix for a given SDE model, initial
# condition x₀, and time span (t₀, T).

# """
# function gaussian_computation(
#     model,
#     x₀,
#     t₀,
#     T,
#     dt,
#     dx,
#     ode_solver;
#     Σ₀ = zeros(model.d, model.d),
#     ode_solver_kwargs...,
# )
#     d = model.d

#     ts = t₀:dt:T

#     # Generate the required flow map data
#     # First, advect the initial condition forward to obtain the final position
#     prob = ODEProblem(model.velocity, x₀, (t₀, T))
#     det_sol = solve(prob, ode_solver; dt = dt, saveat = ts, ode_solver_kwargs...)
#     ws = det_sol.u

#     # Form the star grid around the initial position
#     star = star_grid(x₀, dx)

#     # Advect these points forwards to the final time
#     prob = ODEProblem(model.velocity, star[1, :], (t₀, T))
#     ensemble = EnsembleProblem(prob; prob_func = (prob, i, _) -> remake(prob; u0 = star[i, :]))
#     sol = solve(
#         ensemble,
#         ode_solver,
#         EnsembleThreads();
#         dt = dt,
#         saveat = ts,
#         trajectories = 2 * d,
#         ode_solver_kwargs...,
#     )

#     # Permute the dimensions of the ensemble solution so star_values is indexed
#     # as (timestep, gridpoint, coordinate).
#     star_values = Array{Float64}(undef, length(sol[1]), 2 * d, d)
#     permutedims!(star_values, Array(sol), [2, 3, 1])

#     # Approximate the flow map gradient at each time step
#     ∇Fs = ∇F.(eachslice(star_values; dims = 1), d, dx)

#     # Evaluate σ along the trajectory
#     σs = model.σ.(det_sol.(ts), nothing, ts)

#     # Evaluate the integrand at each time point.
#     # See Theorem 2.2 for the integral form of Σ.
#     Ks = inv.(∇Fs) .* σs
#     integrand = Ks .* transpose.(Ks)

#     # Approximate the time integral with the composite 1/3 Simpson's rule
#     Σs = Vector{Matrix}(undef, length(ts))
#     Σs[1] = Σ₀
#     prev = dt / 3 * integrand[1]
#     for i in 2:length(ts)
#         # The integral itself
#         prev += dt / 3 * (integrand[i] + (1 + 2 * ((i - 1) % 2)) * integrand[i - 1])

#         # Pre- and post-multiply by the final time flow map gradient gradient
#         Σs[i] = ∇Fs[i] * (prev + Σ₀) * transpose(∇Fs[i])
#     end

#     return ws, Σs
# end