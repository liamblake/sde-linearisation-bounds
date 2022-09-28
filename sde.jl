using DifferentialEquations

"""
Generate N realisations of an SDE, returning a matrix of the final position.
"""
function sde_realisations(dest, vel!, σ!, N, d_y, d_W, y₀, t₀, T, dt)
    sde_prob = SDEProblem(vel!, σ!, y₀, (t₀, T), noise_rate_prototype = zeros(d_y, d_W))
    ens = EnsembleProblem(sde_prob)
    sol = solve(
        ens,
        EM(),
        EnsembleThreads(),
        trajectories = N,
        dt = dt,
        save_everystep = false,
    )

    # Only need the final position
    dest .= reduce(hcat, DifferentialEquations.EnsembleAnalysis.get_timepoint(sol, T))
    nothing
end
