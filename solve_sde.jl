using DifferentialEquations
using LaTeXStrings
using Parameters
using ProgressMeter

"""
Generate N realisations of an SDE, filling a matrix of the final position in-place.
Will use multithreading if available: Threads.nthreads()
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
function generate_data!(y_dest, z_dest, gauss_z_dest, gauss_y_dest, model::Model, space_time::SpaceTime, N::Int64, εs::AbstractVector, dts::AbstractVector; quiet=false)
    @unpack name, d, velocity!, ∇u, σ! = model
    @unpack x₀, t₀, T = space_time
    nε = length(εs)

    # Ensure destination is of the the expected size
    @assert size(y_dest) == size(z_dest) == size(gauss_z_dest) == size(gauss_y_dest) == (nε, d, N)

    # For storing simulations - pre-allocate once and reuse
    joint_rels = Array{Float64}(undef, (3 * d, N))

    # Calculate the deterministic trajectory. This is needed to generate directly from the SDE satisfied by z, and to form the limiting velocity field
    !quiet && println("Solving for deterministic trajectory...")
    det_prob = ODEProblem(velocity!, x₀, (t₀, T))
    det_sol = solve(det_prob, Euler(), dt=minimum(dts))
    w = last(det_sol.u)


    !quiet && println("Generating realisations for values of ε...")
    @showprogress for (i, ε) in enumerate(εs)
        dt = dts[i]
        # Set up as a joint system so the same noise realisation is used.
        # TODO: There are probably some clever tricks to avoid new allocations each time - maybe carry
        # F and u_F as state variables?
        function joint_system!(dx, x, _, t)
            F = det_sol(t)

            # y^ε equation
            velocity!(@view(dx[1:d]), x[1:d], nothing, t)

            # z^ε equation
            u_F = Vector{Float64}(undef, d)
            velocity!(u_F, F, nothing, t)
            velocity!(@view(dx[(d+1):(2*d)]), ε * x[(d+1):(2*d)] + F, nothing, t)
            dx[(d+1):(2*d)] -= u_F
            dx[(d+1):(2*d)] *= 1 / ε

            # Limiting equation
            dx[(2*d+1):(3*d)] = ∇u(det_sol(t), t) * x[(2*d+1):(3*d)]

            nothing
        end

        # Diffusion matrix for the joint system
        function joint_σ!(dW, x, _, t)
            dW .= 0.0
            # TODO: Do not use a for loop here
            for j = 1:d
                # y^ε
                dW[j, j] = ε
                # z^ε
                dW[d+j, j] = 1.0
                # z
                dW[d+2*j, j] = 1.0
            end

            nothing
        end

        # Simulate from the y equation, the z equation and the limiting equation simultaneously.
        # This ensures the same realisation of the Wiener process Wₜ is used for each.
        sde_realisations!(
            joint_rels,
            joint_system!,
            joint_σ!,
            N,
            3 * d,
            d,
            vcat(x₀, zeros(2 * d)),
            t₀,
            T,
            dt,
        )

        # Store the realisations appropriately
        y_dest[i, :, :] .= joint_rels[1:d, :]
        z_dest[i, :, :] .= joint_rels[(d+1):(2*d), :] #(y_dest[i, :, :] .- w) ./ ε
        gauss_z_dest[i, :, :] .= joint_rels[(2*d+1):(3*d), :]
        gauss_y_dest[i, :, :] .= ε * gauss_z_dest[i, :, :] .+ w

    end

    nothing
end