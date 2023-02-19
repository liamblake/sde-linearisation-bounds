using LinearAlgebra
using Printf
using Random

using CairoMakie
using Distributions
using Statistics

include("covariance.jl")

# The color palette to use throughout
PALETTE = :spring

# The alphabet
ALPH = 'a':'z'

"""
	pnorm(A; dims=1, p=2)

Calculate the p-norm along the given dimension of a multidimensional array.
"""
function pnorm(A; dims = 1, p = 2)
    f = a -> norm(a, p)
    return mapslices(f, A; dims = dims)
end

"""
save_figure(p, fname; show_print=true)

Helper function to save a Plots.plot object to the given file fname. If show_print is true, then
a line is printed indicating where the figure has been saved.
"""
function save_figure(p, fname; show_print = true)
    path = "output/$(fname)"
    if show_print
        println("Saving figure to $(path)")
    end

    # Trim any unneeded whitespace
    resize_to_layout!(p)

    save(path, p; pt_per_unit = 1)
end

"""
    normalised_hexbin!(ax, x, y; kwargs...)

Plot a hexbin with normalised cell counts.
"""
function normalised_hexbin!(ax, x, y; kwargs...)
    total = length(x)
    @assert length(y) == total

    hb = hexbin!(ax, x, y; scale = c -> c / total, kwargs...)
end

"""
	bivariate_gaussian_std_dev(μ, Σ; nσ = 1, plt = plot(), ...)

Plot the n standard-deviation regions of a bivariate Gaussian distribution
with mean μ and covariance matrix Σ. The number of regions plotted is specified
by nσ.
"""
function bivariate_std_dev!(ax, μ, Σ; nσ = 1, colour = :black, label = nothing, kwargs...)
    # Calculate the first two principal axes of the covariance matrix
    # These correspond to the major and minor axes of the ellipse
    evals, evecs = eigen(Σ)

    # Angle of rotation - use the principal axis
    θ = atan(evecs[2, 1], evecs[1, 1])

    # Magnitude of major and minor axes
    a, b = sqrt.(evals[1:2])

    ts = 0:0.001:(2π)

    # Plot each contour
    for n = 1:nσ
        # Parametric equations for the resulting ellipse
        # TODO: Should be a way to calculate this by operating directly on the eigenvectors
        # i.e. x = cos(θ), y = sin(θ)
        x = t -> n * (a * cos(t) * cos(θ) - b * sin(t) * sin(θ)) + μ[1]
        y = t -> n * (a * cos(t) * sin(θ) + b * sin(t) * cos(θ)) + μ[2]

        lines!(ax, x.(ts), y.(ts); color = colour, label = (n == 1) ? label : nothing, kwargs...)
    end

    # Also plot the mean
    scatter!(ax, [μ[1]], [μ[2]]; markersize = 9, color = colour, markerstrokecolor = colour)
end

"""
	lobf(x, y; intercept = false)

Given some 2-dimensional data, calculate a line of best fit, with a least-squares estimate.
An intercept is included by default.
"""
function lobf(x, y)
    n = length(x)
    X = hcat(ones(n), x)

    # Calculate the least-squares estimate.
    coefs = inv(X' * X) * X' * y

    # Fit the line to each datapoint, for plotting
    return X * coefs, coefs
end

"""

Plot a series of values y against another x, both in the base-10 logarithmic scale.
The plot is saved to save_dir.
"""
function plot_log_with_lobf!(g, x, y; annotation = s -> "", kwargs...)
    logx = log10.(x)
    logy = log10.(y)

    # Add a line of best fit
    fit, coefs = lobf(logx, logy)
    slope = round(coefs[2]; digits = 2)
    ax = Axis(g; kwargs...)
    lines!(ax, logx, fit; color = :red)

    # Plot the points about the LOBF
    scatter!(ax, logx, logy; color = :black)

    # Determine the position of the text. Assuming the line is decreasing, take the second smallest
    # x-value and the second largest y-value. Seems to work well.
    t_pos_x = logx[partialsortperm(logx, 1:2)[2]]
    t_pos_y = logy[partialsortperm(logy, 1:2; rev = true)[2]]

    text!(t_pos_x, t_pos_y; text = annotation(slope), color = :red)
    hidedecorations!(ax; label = false, ticklabels = false, ticks = false)

    return ax
end

"""

"""
function theorem_validation(
    y_rels,
    z_rels,
    gauss_z_rels,
    model,
    space_time,
    εs,
    dt,
    rs,
    ptheme,
    rlabels;
    ode_solver = Euler(),
    legend_idx = 1,
    plot_attrs = Dict(),
    hist_idxs = 1:length(εs),
    hist_ncols = 2,
)
    @unpack x₀, t₀, T = space_time
    name = "$(model.name)_$(x₀)_[$(t₀),$(T)]"

    nε = length(εs)
    nr = length(rs)
    N = size(y_rels)[3]

    # Preallocate storage of results
    # y_abs_diff = Array{Float64}(undef, nr, nε)
    z_abs_diff = Array{Float64}(undef, nr, nε)
    z_means = Array{Float64}(undef, nε, model.d)
    y_means = Array{Float64}(undef, nε, model.d)
    sample_S2s = Vector{Float64}(undef, nε)

    # Calculate the deviation covariance from the integral expression
    w, Σ = Σ_calculation(model, x₀, t₀, T, dt, 0.001, ode_solver)
    # Theoretical stochastic sensitivity - the maximum eigenvalue of Σ
    S2 = opnorm(Matrix(Σ))

    # Big histogram plot
    f_hist = Figure()
    n_histed = 1

    all_hists = for (i, ε) in enumerate(εs)
        # Diagnostics - mean should be zero
        y_means[i, :] = mean(y_rels[i, :, :] .- w; dims = 2)
        z_means[i, :] = mean(z_rels[i, :, :]; dims = 2)

        # Calculate the normed distance between the scaled deviation and the solution,
        # in order to estimate 𝔼[|z_ε - z|ʳ]
        z_diffs = pnorm(z_rels[i, :, :] .- gauss_z_rels[i, :, :]; dims = 1)

        # Calculate the sample covariance matrices
        s_mean_y = mean(y_rels[i, :, :]; dims = 2)
        S_y = cov(y_rels[i, :, :]; dims = 2)
        s_mean_z = mean(z_rels[i, :, :]; dims = 2)
        S_z = cov(z_rels[i, :, :]; dims = 2)
        # Calculate empirical stochastic sensitivity
        sample_S2s[i] = opnorm(S_z)

        # println("ε = $(ε): $(s_mean_z)")
        for (j, r) in enumerate(rs)
            z_abs_diff[j, i] = mean(z_diffs .^ r)
        end

        if i in hist_idxs
            grid = f_hist[ceil(Int64, n_histed / hist_ncols), mod1(n_histed, 2)] = GridLayout()

            # Plot a histogram of the realisations, with covariance bounds
            ax = Axis(grid[2, 1]; xlabel = L"y_1", ylabel = L"y_2")
            hb = normalised_hexbin!(
                ax,
                y_rels[i, 1, :],
                y_rels[i, 2, :];
                bins = 100,
                colormap = cgrad(PALETTE; rev = true),
            )
            bivariate_std_dev!(
                ax,
                w,
                ε^2 * Σ;
                nσ = 3,
                colour = :black,
                linestyle = :solid,
                label = "Theory",
            )
            bivariate_std_dev!(
                ax,
                s_mean_y,
                S_y;
                nσ = 3,
                colour = :blue,
                linestyle = :dash,
                label = "Sample",
            )

            # Add the colourbar above the plot
            Colorbar(grid[1, 1], hb; vertical = false)

            # Add the label - assuming εs are powers of 10
            eps_pow_str = @sprintf("%.2f", log10(ε))
            Label(
                grid[3, 1],
                L"(%$(ALPH[n_histed])) $\epsilon = 10^{%$(eps_pow_str)}$";
                fontsize = ptheme.fontsize.val * 1.25,
            )

            # Add a legend if required
            if i == legend_idx
                axislegend(ax; position = :rt)
            end

            n_histed += 1
        end
    end

    save_figure(f_hist, "$(name)/y_histogram.pdf")

    f_errs = Figure()
    for (j, r) in enumerate(rs)
        grid = f_errs[ceil(Int64, j / 2), mod1(j, 2)] = GridLayout()
        ax = plot_log_with_lobf!(
            grid[1, 1],
            εs,
            @view(z_abs_diff[j, :]);
            annotation = slope -> L"\Gamma_z^{(%$r)}(\epsilon) \sim \epsilon^{%$slope}",
            xlabel = L"\log_{10}{\,\epsilon}",
            ylabel = L"\log_{10}{\,\Gamma_z^{(%$r)}(\epsilon)}",
        )

        Label(grid[2, 1], rlabels[j]; fontsize = ptheme.fontsize.val * 1.25)
    end

    save_figure(f_errs, "$(name)/z_diff.pdf")
end

"""

"""
function Σ_through_time!(
    fig,
    rels,
    model,
    ε,
    x₀,
    t₀,
    ts,
    dt,
    g_label1,
    g_label2,
    fig_col;
    hist_idxs = 1:length(ts),
    ode_solver = Euler(),
)
    T = maximum(ts)
    name = "$(model.name)_$(x₀)_[$(t₀),$(T)]"

    # Solve for the full deterministic trajectory
    det_prob = ODEProblem(model.velocity, x₀, (t₀, T))
    det_sol = solve(det_prob, ode_solver; dt = dt, dtmax = dt)

    S²_vals = Vector{Float64}(undef, length(ts))
    sample_S²_vals = Vector{Float64}(undef, length(ts))
    ws = Vector{Vector{Float64}}(undef, length(ts))
    Σs = Vector{Matrix{Float64}}(undef, length(ts))

    # Setup the trajectory through time plot
    g_traj = fig[1, fig_col] = GridLayout()
    traj_ax = Axis(g_traj[1, 1]; xlabel = L"x_1", ylabel = L"x_2")

    for (i, t) in enumerate(ts)
        w, Σ = Σ_calculation(model, x₀, t₀, t, dt, 0.001, ode_solver)
        ws[i] = w
        Σs[i] = Σ
        S²_vals[i] = opnorm(Σ)
        sample_S²_vals[i] = opnorm(cov((rels[i, :, :] .- w) ./ ε; dims = 2))

        if i in hist_idxs
            hexbin!(
                traj_ax,
                rels[i, 1, :],
                rels[i, 2, :];
                bins = 100,
                colormap = cgrad(PALETTE; rev = true),
            )
        end
    end

    # Plot the deterministic trajectory (above the histograms, below the SD bounds)
    sol_through_time_x = det_sol.(t₀:dt:T, idxs = 1)
    sol_through_time_y = det_sol.(t₀:dt:T, idxs = 2)
    lines!(traj_ax, sol_through_time_x, sol_through_time_y; color = :gray, linewidth = 0.5)

    # Plot the covariance visualisations
    for i in hist_idxs
        bivariate_std_dev!(traj_ax, ws[i], ε^2 * Σs[i]; nσ = 2, linecolor = :black, linewidth = 0.5)
    end

    Label(g_traj[2, 1], g_label1; fontsize = ptheme.fontsize.val * 1.25)

    # g2 = fig[2, fig_col] = GridLayout()
    # ax = Axis(g2[1, 1]; xlabel = L"t", ylabel = "Covariance operator norm")

    # scatter!(g2[1, 1], ts, sample_S²_vals; color = :black, label = L"Limiting ($S^2(x,t)$)")
    # scatter!(
    #     ax,
    #     ts,
    #     S²_vals;
    #     color = :transparent,
    #     strokewidth = 1.5,
    #     strokecolor = :red,
    #     label = "Sample",
    # )
    # hidedecorations!(ax; label = false, ticklabels = false, ticks = false)
    # axislegend(ax; halign = :left, valign = :top)

    # Label(g2[2, 1], g_label2; fontsize = ptheme.fontsize.val * 1.25)
end

"""
S²_grid_sets(
    model,
    x₀_grid,
    t₀,
    T,
    threshold,
    dt,
    dx,
    fname_ext;
    ode_solver = Euler(),
    plot_attrs...,
)

Computes ...

Arguments:


"""
function S²_grid_sets(model, xs, ys, t₀, T, threshold, dt, dx, fname_ext; ode_solver = Euler())
    # Compute the S² value for each initial condition, as the operator norm of Σ
    S²_grid = map(collect(Base.product(xs, ys))) do pair
        opnorm(Σ_calculation(model, [pair[1], pair[2]], t₀, T, dt, dx, ode_solver)[2])
    end

    f = Figure()
    ax = Axis(f[2, 1]; xlabel = L"x_1", ylabel = L"x_2")
    hm = heatmap!(ax, xs, ys, log.(S²_grid); colormap = :pink, fxaa = false)
    Colorbar(f[1, 1], hm; vertical = false)

    # Extract robust sets and plot
    R = S²_grid .< threshold
    ax2 = Axis(f[2, 2]; xlabel = L"x_1", ylabel = L"x_2")
    heatmap!(ax2, xs, ys, R; colormap = cgrad([:white, :lightskyblue]), fxaa = false)

    save_figure(f, "s2_field$(fname_ext).pdf")
end