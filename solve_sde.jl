using DifferentialEquations
using LaTeXStrings
using Parameters
using ProgressMeter

"""
Generate N realisations of an SDE, filling a matrix of the final position in-place.
"""
function sde_realisations!(dest, vel!, σ!, N, d_y, d_W, y₀, t₀, T, dt)
    sde_prob = SDEProblem(vel!, σ!, y₀, (t₀, T), noise_rate_prototype=zeros(d_y, d_W))

    # This function will be called on the output of each realisation when solving the
    # EnsembleProblem. Here, we take the final position of the solution and place it
    # directly into the dest array.
    function output_func(sol, i)
        dest[:, i] = last(sol)
        (nothing, false)
    end
    ens = EnsembleProblem(sde_prob, output_func=output_func)
    solve(ens, EM(), EnsembleThreads(), trajectories=N, dt=dt, save_everystep=false)

    nothing
end

"""

Generate realisations of the SDE solution and the limiting solution.
"""
function generate_data!(y_dest, z_dest, gauss_z_dest, gauss_y_dest, model::Model, space_time::SpaceTime, N::Int64, εs::AbstractVector; dt=nothing, quiet=false)
    @unpack name, d, velocity!, ∇u, Kᵤ = model
    @unpack x₀, t₀, T = space_time
    nε = length(εs)

    # Ensure destination is of the the expected size
    @assert size(y_dest) == size(z_dest) == size(gauss_z_dest) == size(gauss_y_dest) == (nε, d, N)

    # Use dt = min(εs) / 10 if not specified
    if dt === nothing
        dt = minimum(εs) / 10.0
    end

    # For storing simulations - pre-allocate once and reuse
    joint_rels = Array{Float64}(undef, (2 * d, N))

    # Calculate the deterministic trajectory. This is needed to form the limiting velocity field
    !quiet && println("Solving for deterministic trajectory...")
    det_prob = ODEProblem(velocity!, x₀, (t₀, T))
    det_sol = solve(det_prob, Euler(), dt=dt)
    w = last(det_sol.u)

    # Set up as a joint system so the same noise realisation is used.
    function joint_system!(dx, x, _, t)
        velocity!(dx, x, NaN, t)
        dx[(d+1):(2*d)] = ∇u(det_sol(t), t) * x[(d+1):(2*d)]
        nothing
    end

    !quiet && println("Generating realisations for values of ε...")
    @showprogress for (i, ε) in enumerate(εs)
        # !quiet && println("\t ε = $(ε) \t ($(i)/$(nε))")

        data_path = "data/$(name)_$(ε).jld"

        # Diffusion matrix for the joint system
        function σ!(dW, _, _, _)
            dW .= 0.0
            # TODO: Do not use a for loop here
            for j = 1:d
                dW[j, j] = ε
                dW[d+j, j] = 1.0
            end

            nothing
        end

        # Simulate from the y equation and the limiting equation simultaneously
        sde_realisations!(
            joint_rels,
            joint_system!,
            σ!,
            N,
            2 * d,
            d,
            vcat(x₀, zeros(d)),
            t₀,
            T,
            dt,
        )

        # Store the realisations appropriately
        y_dest[i, :, :] .= joint_rels[1:d, :]
        gauss_z_dest[i, :, :] .= joint_rels[(d+1):(2*d), :]
        z_dest[i, :, :] .= (y_dest[i, :, :] .- w) / ε
        gauss_y_dest[i, :, :] .= ε * z_dest[i, :, :] .+ w

    end

    nothing
end