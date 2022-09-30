using DifferentialEquations

"""
Generate N realisations of an SDE, returning a matrix of the final position.
"""
function sde_realisations(dest, vel!, σ!, N, d_y, d_W, y₀, t₀, T, dt)
    sde_prob = SDEProblem(vel!, σ!, y₀, (t₀, T), noise_rate_prototype = zeros(d_y, d_W))

    function output_func(sol, i)
        dest[:, i] = last(sol)
        (nothing, false)
    end
    ens = EnsembleProblem(sde_prob, output_func = output_func)
    solve(ens, EM(), EnsembleThreads(), trajectories = N, dt = dt, save_everystep = false)

    nothing
end
