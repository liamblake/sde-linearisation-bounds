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

"""
    S2_computation(name, xgrid, ygrid, t0, T, dt, u!, ∇u!, σσᵀ!, R; solver_kwargs...)

Compute and plot the stochastic sensitivity (S²) field for the 2-dimensional model with velocity
field `u!`, gradient `∇u!`, and diffusion matrix `σσᵀ!`, on the grid of initial
conditions specified by xgrid and ygrid and over the time interval (t0, T) with integration timestep
dt. The S² field is plotted, along with the robust sets for the given threshold R.
"""
function S2_computation(name, xgrid, ygrid, t0, T, dt, u!, ∇u!, σσᵀ!, R; solver_kwargs...)
    mesh = [[x, y] for x in xgrid, y in ygrid][:]

    # Compute the stochastic sensitivity field with a zero initial condition
    ts = range(t0; stop = T, step = dt)
    wtmp = Vector{Vector}(undef, length(ts))
    Σ_tmp = Vector{Matrix}(undef, length(ts))
    Σs = Array{Float64}(undef, 2, 2, length(mesh))

    pbar = Progress(length(mesh), 1, "Computing covariance matrices...", 50)
    for (i, x₀) in enumerate(mesh)
        # S2 with zero initial condition
        gaussian_computation!(wtmp, Σ_tmp, 2, u!, ∇u!, σσᵀ!, x₀, zeros(2, 2), ts)
        Σs[:, :, i] = Σ_tmp[end]

        next!(pbar)
    end
    finish!(pbar)

    S2_comp = J -> eigen(J).values[2]

    S2_grid = reshape(map(S2_comp, eachslice(Σs; dims = 3)), length(xgrid), length(ygrid))

    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    p = ax.pcolormesh(xgrid, ygrid, S2_grid'; norm = "log", cmap = :pink, rasterized = true)
    colorbar(p; ax = ax, location = "top", aspect = 40, label = L"S^2")

    fig.savefig("output/$name/S2_zero.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)

    # Extract robust sets from the S² field
    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    p = ax.pcolormesh(xgrid, ygrid, (S2_grid .< R)'; cmap = "twocolor_blue", rasterized = true)

    # Trying to get an invisible colourbar - overlay a transparent copy of the field, but with a
    # completely transparent colormap. A dirty hack, according to my Copilot overlord.
    p = ax.pcolormesh(
        xgrid,
        ygrid,
        (S2_grid .< R)';
        cmap = ColorMap("nothing", [RGBA(0.0, 0.0, 0.0, 0.0), RGBA(0.0, 0.0, 0.0, 0.0)]),
        rasterized = true,
    )

    cb = colorbar(p; ax = ax, location = "top", aspect = 40, label = L"S^2")

    # Hide label, ticks, and outline of colorbar
    cb.ax.xaxis.label.set_color(:white)
    cb.ax.tick_params(; axis = "x", colors = :white)
    cb.outline.set_visible(false)

    fig.savefig("output/$name/S2_robust.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)
end