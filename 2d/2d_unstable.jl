using PyPlot

# Define the 2D model
function u!(s, x, _)
    s[1] = -4 * x[1] + x[1]^2
    s[2] = 3 * x[2] - x[2]^3
end

function ∇u!(s, x, _)
    s[1, 1] = -4 + 2 * x[1]
    s[1, 2] = 0.0
    s[2, 1] = 0.0
    s[2, 2] = 3 - 3 * x[2]^2
end

function σ!(s, x, _)
    s[1, 1] = 1.0
    s[1, 2] = 0.0
    s[2, 1] = x[2] - 1
    s[2, 2] = 3 * sin(2 * π * x[1]) * exp(0.8(x[1]))
end

function σσᵀ!(s, x, _)
    s[1, 1] = 1.0
    s[2, 2] = x[2] - 1
    s[1, 2] = x[2] - 1
    s[2, 2] = (x[2] - 1)^2 + 9 * sin(2 * π * x[1])^2 * exp(1.6 * x[1])
end

"""

Dumb algorithm for finding the maximising ridges when it is expected to be solely horizontal.
"""
function greedy_ridge_finder(xgrid, field)
    nx, ny = size(field)

    ridge_idx = Vector{Int64}(undef, nx)
    ridge_idx[1] = argmax(field[1, :])

    for j in 2:nx
        # i = ridge_idx[j - 1]
        ridge_idx[j] = argmax(field[j, :]) # i - 1 + argmax(field[j, (i - 1):(i + 1)])
    end

    return ridge_idx
end

### FTLE versus S²
begin
    # include("ftle.jl")

    # Declare a grid of initial conditions
    xgrid = collect(range(0; stop = 2, length = 400))
    ygrid = collect(range(-0.2; stop = 0.2, length = 400))
    # ftle_S2_comparison("unstable", xgrid, ygrid, 0.0, 1.0, u!, ∇u!, σσᵀ!, 10.0)

    include("ftle.jl")
    include("../pyplot_setup.jl")
    name = "unstable"
    t0 = 0.0
    T = 1.0
    R = 10.0
    solver_kwargs = Dict()

    mesh = [[x, y] for x in xgrid, y in ygrid][:]

    ∇Fs = Array{Float64}(undef, 2, 2, length(mesh))
    compute_∇Fs!(∇Fs, mesh, 2, u!, ∇u!, t0, T; solver_kwargs...)

    ftle_comp = J -> eigen(Symmetric(J' * J)).values[2]
    ftle_grid = reshape(map(ftle_comp, eachslice(∇Fs; dims = 3)), length(xgrid), length(ygrid))

    # Plot as a heatmap
    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    # PyPlot.matplotlib.colormaps.get_cmap("bwr").set_bad(RGB(0.0, 0.0, 0.0))
    ftle_ridge = greedy_ridge_finder(xgrid, ftle_grid)
    # ftle_grid[:, ftle_ridge] .= NaN

    p = ax.pcolormesh(xgrid, ygrid, ftle_grid'; norm = "log", cmap = :bwr, rasterized = true)
    colorbar(p; ax = ax, location = "top", aspect = 40)

    # ftle_ridge = greedy_ridge_finder(xgrid, ftle_grid)
    # # Turn into a binary matrix
    # ftle_ridge_grid = zeros(length(xgrid), length(ygrid))
    # ftle_ridge_grid[:, ftle_ridge] .= 1.0
    # ax.pcolor(xgrid, ygrid, ftle_ridge_grid'; cmap = "twocolor", rasterized = true)

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

    S2_ridge = greedy_ridge_finder(xgrid, S2_zero_grid)
    # S2_zero_grid[:, S2_ridge] .= NaN
    p = ax.pcolormesh(xgrid, ygrid, S2_zero_grid'; norm = "log", cmap = :bwr, rasterized = true)
    colorbar(p; ax = ax, location = "top", aspect = 40)

    fig.savefig("output/$name/S2_zero.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)

    # Extract robust sets from the S² field
    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    p = ax.pcolormesh(xgrid, ygrid, (S2_zero_grid .< R)'; cmap = "twocolor", rasterized = true)

    fig.savefig("output/$name/S2_robust.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)

    fig, ax = subplots()
    ax.set_xlabel(L"y_1")
    ax.set_ylabel(L"y_2")

    S2_ridge = greedy_ridge_finder(xgrid, S2_I_grid)
    # S2_I_grid[:, S2_ridge] .= NaN
    p = ax.pcolormesh(xgrid, ygrid, S2_I_grid'; norm = "log", cmap = :bwr, rasterized = true)
    colorbar(p; ax = ax, location = "top", aspect = 40)

    fig.savefig("output/$name/S2_I.pdf"; dpi = 600, bbox_inches = "tight")
    close(fig)
end

