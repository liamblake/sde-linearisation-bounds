using DifferentialEquations
using LaTeXStrings
using Parameters
using ProgressMeter
using StaticArrays

"""
Generate N realisations of an SDE, filling a matrix of the final position in-place.
Will use multithreading if available: Threads.nthreads()
"""
function sde_realisations!(dest, vel!, σ, N, d_y, d_W, ε, y₀, t₀, T, dt, sde_solver)
    nrp = @SArray(zeros(d_y, d_W))
    sde_prob = SDEProblem(vel!, σ, y₀, (t₀, T), ε; noise_rate_prototype = nrp)

    # This function will be called on the output of each realisation when solving the
    # EnsembleProblem. Here, we take the final position of the solution and place it
    # directly into the dest array.
    function output_func(sol, i)
        dest[:, i] = last(sol)
        (nothing, false)
    end
    ens = EnsembleProblem(sde_prob; output_func = output_func)
    solve(
        ens,
        sde_solver,
        EnsembleThreads();
        trajectories = N,
        dt = dt,
        dtmax = dt,
        save_everystep = false,
    )

    nothing
end

"""

Generate realisations of the SDE solution and the limiting solution.
"""
function generate_ε_data!(
    y_dest,
    z_dest,
    gauss_z_dest,
    model,
    space_time,
    N,
    εs,
    dt,
    ode_solver = Euler(),
    sde_solver = EM(),
)
    @unpack name, d, velocity, ∇u = model
    @unpack x₀, t₀, T = space_time
    nε = length(εs)

    # Ensure destination is of the the expected size
    @assert size(y_dest) == size(z_dest) == size(gauss_z_dest) == (nε, d, N)

    # For storing simulations - pre-allocate once and reuse
    joint_rels = Array{Float64}(undef, (2 * d, N))

    # Calculate the deterministic trajectory. This is needed to form the limiting velocity field
    println("Solving for deterministic trajectory...")
    det_prob = ODEProblem(velocity, x₀, (t₀, T))
    det_sol = solve(det_prob, ode_solver; dt = dt, dtmax = dt)
    w = last(det_sol.u)

    # Set up as a joint system so the same noise realisation is used.
    function joint_system(x, _, t)
        SVector{2 * d}([velocity(x, NaN, t); (∇u(det_sol(t), t) * x[(d + 1):(2 * d)])])
    end

    # Diffusion matrix for the joint system.
    function σ(_, ε, _)
        dW = zeros(2 * d, d)
        # TODO: Do not use a for loop here
        for j = 1:d
            dW[j, j] = ε
            dW[d + j, j] = 1.0
        end
        return SMatrix{2 * d,d}(dW)
    end
    println("Generating realisations for values of ε...")

    @showprogress for (i, ε) in enumerate(εs)
        # Simulate from the y equation and the limiting equation simultaneously
        sde_realisations!(
            joint_rels,
            joint_system,
            σ,
            N,
            2 * d,
            d,
            ε,
            SVector{2 * d}([x₀; zeros(d)]),
            t₀,
            T,
            dt,
            sde_solver,
        )

        # Store the realisations appropriately
        y_dest[i, :, :] .= joint_rels[1:d, :]
        gauss_z_dest[i, :, :] .= joint_rels[(d + 1):(2 * d), :]
        z_dest[i, :, :] .= (y_dest[i, :, :] .- w) ./ ε
    end

    nothing
end

function generate_time_data!(dest, model, ε, x₀, t₀, ts, N, dt)
    @unpack name, d, velocity, σ = model

    function εσ(x, _, t)
        return ε * σ(x, nothing, t)
    end

    # Ensure destination is of the the expected size
    @assert size(dest) == (length(ts), d, N)

    println("Generating realisations through time...")
    @showprogress for (i, t) in enumerate(ts)
        sde_realisations!(@view(dest[i, :, :]), velocity, εσ, N, d, d, ε, x₀, t₀, t, dt, sde_solver)
    end
    nothing
end
