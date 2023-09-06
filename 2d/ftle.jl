using ProgressMeter
using DifferentialEquations

include("gaussian_computation.jl")
include("../pyplot_setup.jl")

"""
    compute_∇Fs!(dest, x₀s, d, u!, ∇u!, t, T; solver = Tsit5(), solver_kwargs...)

Given a dynamical system ẋ = u(x,t) and a set of initial conditions, compute the flow map gradient
∇Fₜᵀ(x₀) for each initial condition, by solving the equation of variations directly.
"""
function compute_∇Fs!(dest, x₀s, d, u!, ∇u!, t, T; solver = Tsit5(), solver_kwargs...)
    tmp_mat = Array{Float64}(undef, d, d)
    init_mat = diagm(ones(d))[:]
    function flow_map!(s, x, _, t)
        # Flow map
        u!(@view(s[1:d]), x[1:d], t)

        # Flow map gradient
        ∇u!(tmp_mat, x[1:d], t)
        s[(d + 1):end] .= (tmp_mat * reshape(x[(d + 1):end], d, d))[:]

        nothing
    end

    prob = ODEProblem(flow_map!, zeros(d + d^2), (t, T))
    ens =
        EnsembleProblem(prob; prob_func = (prob, i, _) -> remake(prob; u0 = vcat(x₀s[i], init_mat)))
    sol = solve(
        ens,
        solver,
        EnsembleThreads();
        trajectories = length(x₀s),
        save_everystep = false,
        save_start = false,
        solver_kwargs...,
    )

    dest .= reshape(Array(sol)[(d + 1):end, :, :], d, d, length(x₀s))
end
function ftle_S2_comparison(name, xgrid, ygrid, t0, T, u!, ∇u!, σσᵀ!; solver_kwargs...)
    mesh = [[x, y] for x in xgrid, y in ygrid][:]

    ∇Fs = Array{Float64}(undef, 2, 2, length(mesh))
    compute_∇Fs!(∇Fs, mesh, 2, u!, ∇u!, t0, T; solver_kwargs...)

    ftle_comp = J -> eigen(Symmetric(J' * J)).values[2]
    ftle_grid = reshape(map(ftle_comp, eachslice(∇Fs; dims = 3)), length(xgrid), length(ygrid))

    # Plot as a heatmap
    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    p = ax.pcolormesh(xgrid, ygrid, ftle_grid'; norm = "log", cmap = :pink, rasterized = true)
    colorbar(p; ax = ax, location = "top", aspect = 40)

    fig.savefig("output/$name/ftle.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)

    # Compute the stochastic sensitivity field with a zero initial condition
    ts = range(t0; stop = T, length = 300)
    wtmp = Vector{Vector}(undef, length(ts))
    Σ_tmp = Vector{Matrix}(undef, length(ts))
    Σs_zero = Array{Float64}(undef, 2, 2, length(mesh))
    Σs_I = Array{Float64}(undef, 2, 2, length(mesh))

    pbar = Progress(length(mesh), 1, "Computing S² values...", 50)
    for (i, x₀) in enumerate(mesh)
        gaussian_computation!(wtmp, Σ_tmp, 2, u!, ∇u!, σσᵀ!, x₀, zeros(2, 2), ts)
        Σs_zero[:, :, i] = Σ_tmp[end]

        gaussian_computation!(wtmp, Σ_tmp, 2, u!, ∇u!, σσᵀ!, x₀, diagm(ones(2)), ts)
        Σs_I[:, :, i] = Σ_tmp[end]

        next!(pbar)
    end
    finish!(pbar)

    S2_comp = J -> eigen(J).values[2]

    S2_zero_grid = reshape(map(S2_comp, eachslice(Σs_zero; dims = 3)), length(xgrid), length(ygrid))
    S2_I_grid = reshape(map(S2_comp, eachslice(Σs_I; dims = 3)), length(xgrid), length(ygrid))

    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    p = ax.pcolormesh(xgrid, ygrid, S2_zero_grid'; norm = "log", cmap = :pink, rasterized = true)
    colorbar(p; ax = ax, location = "top", aspect = 40)

    fig.savefig("output/$name/S2_zero.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)

    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    p = ax.pcolormesh(xgrid, ygrid, S2_I_grid'; norm = "log", cmap = :pink, rasterized = true)
    colorbar(p; ax = ax, location = "top", aspect = 40)

    fig.savefig("output/$name/S2_I.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)
end