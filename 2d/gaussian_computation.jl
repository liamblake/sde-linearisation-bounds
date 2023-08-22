"""
    gaussian_computation!(
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
    n,
    u!,
    ∇u!,
    σσᵀ!,
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
    ut = Vector{Float64}(undef, n)
    ∇ut = Matrix{Float64}(undef, n, n)
    Kτ = Matrix{Float64}(undef, n, n)

    # Iterate the Mazzoni hybrid scheme, until the S² threshold OR the final time T is reached.
    for (i, t) in enumerate(ts[2:end])
        dt = t - ts[i]

        # Update the state variable
        u!(ut, state[i], t)
        ∇u!(∇ut, state[i], t)
        state[i + 1] = state[i] + dt * inv(I - ∇ut * dt / 2) * ut

        # Interpolate the state at time t + dt/2
        ut .= 0.5 .* (state[i] .+ state[i + 1] .- ∇ut * ut .* dt^2 / 4)

        # Compute the covariance
        # Compute ∇u(w_τ, t + dt/2) and store in ∇ut
        ∇u!(∇ut, ut, t + dt / 2)
        Kτ .= inv(I - ∇ut * dt / 2)

        # Compute Mτ and store in ∇ut
        mul!(∇ut, Kτ, (I + ∇ut * dt / 2))
        covm[i + 1] = ∇ut * covm[i] * transpose(∇ut)

        # Compute [σσᵀ](w_τ, t + dt/2) and store in ∇ut
        σσᵀ!(∇ut, ut, t + dt / 2)
        covm[i + 1] = covm[i + 1] + Kτ * ∇ut * transpose(Kτ) .* dt

        # Correct any violations of symmetry in Σt.
        # These should only arise from floating point errors and therefore be small over a single
        # step. However, these errors can accumulate over many steps.
        covm[i + 1] = @. 0.5 * (covm[i + 1] + covm[i + 1]')
    end
end
