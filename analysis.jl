using LinearAlgebra
using Printf
using Random

using CairoMakie
using Distributions
using Statistics

include("covariance.jl")

# The color palette to use throughout
PALETTE = :spring

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
    save(path, p)
end

"""
	bivariate_gaussian_std_dev(μ, Σ; nσ = 1, plt = plot(), ...)

Plot the n standard-deviation regions of a bivariate Gaussian distribution
with mean μ and covariance matrix Σ. The number of regions plotted is specified
by nσ.
"""
function bivariate_std_dev!(ax, μ, Σ; nσ = 1, colour = :black, label = "", kwargs...)
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

        lines!(ax, x.(ts), y.(ts); color = colour, label = (n == 1) ? label : "", kwargs...)
    end

    # Also plot the mean
    scatter!(ax, [μ[1]], [μ[2]]; markersize = 12, color = colour, label = "")
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
function plot_log_with_lobf(x, y, save_dir; annotation = s -> "", kwargs...)
    logx = log10.(x)
    logy = log10.(y)

    p, ax, _ = scatter(logx, logy; color = :black, kwargs...)

    # Add a line of best fit
    fit, coefs = lobf(logx, logy)
    slope = round(coefs[2]; digits = 2)
    lines!(ax, logx, fit; color = :red, text = annotation(slope))

    save_figure(p, save_dir)
end

"""

"""
function theorem_validation(
    y_rels,
    z_rels,
    gauss_z_rels,
    # gauss_y_rels,
    model,
    space_time,
    εs,
    dts,
    rs;
    ode_solver = Euler(),
    legend_idx = 1,
    plot_attrs = Dict(),
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

    # Only plot histograms if working in 2D
    plot_histograms = (model.d == 2)
    S2 = 1.0
    Σ = zeros(model.d, model.d)

    for (i, ε) in enumerate(εs)
        # Calculate the deviation covariance from the integral expression
        w, Σ = Σ_calculation(model, x₀, t₀, T - dts[i], dts[i], 0.001, ode_solver)
        # Theoretical stochastic sensitivity - the maximum eigenvalue of Σ
        S2 = opnorm(Matrix(Σ))

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

        if plot_histograms
            # Plot a histogram of the realisations, with covariance bounds
            p, ax, _ = hexbin(
                y_rels[i, 1, :],
                y_rels[i, 2, :];
                bins = 100,
                colormap = cgrad(PALETTE; rev = true),
                xlabel = L"y_1",
                ylabel = L"y_2",
            )
            bivariate_std_dev!(
                ax,
                w,
                ε^2 * Σ;
                nσ = 3,
                plt = p,
                colour = :black,
                linestyle = :solid,
                label = "Theory",
            )
            bivariate_std_dev!(
                ax,
                s_mean_y,
                S_y;
                nσ = 3,
                plt = p,
                colour = :red,
                linestyle = :dash,
                label = "Sample",
            )

            save_figure(p, "$(name)/y_histogram_$(@sprintf("%.7f", ε)).pdf")

            # p = histogram2d(
            #     y_rels[i, 1, :],
            #     y_rels[i, 2, :];
            #     bins = 100,
            #     xlabel = L"y_1",
            #     ylabel = L"y_2",
            #     legend = (i == legend_idx),
            #     c = cgrad(PALETTE; rev = true),
            #     label = "",
            #     grid = false,
            #     plot_attrs...,
            # )
            # p = bivariate_std_dev(
            #     w,
            #     ε^2 * Σ;
            #     nσ = 3,
            #     plt = p,
            #     colour = :black,
            #     linestyle = :solid,
            #     label = "Theory",
            # )
            # p = bivariate_std_dev(
            #     s_mean_y,
            #     S_y;
            #     nσ = 3,
            #     plt = p,
            #     colour = :red,
            #     linestyle = :dash,
            #     label = "Empirical",
            # )
            # save_figure(p, "$(name)/y_histogram_$(@sprintf('%.7f', ε)).pdf")

        end
    end

    for (j, r) in enumerate(rs)
        plot_log_with_lobf(
            εs,
            @view(z_abs_diff[j, :]),
            "$(name)/z_diff_$(r).pdf";
            annotation = slope -> L"\Gamma_z^{(%$r)}(\varepsilon) \sim \varepsilon^{%$slope}",
            axis = (;
                xlabel = L"\log_{10}{\,\varepsilon}",
                ylabel = L"\log_{10}{\,\Gamma_z^{(%$r)}(\varepsilon)}",
            ),
        )
    end

    # Plot the difference in stochastic sensitivity
    abs_S2_diff = abs.(sample_S2s .- S2)
    plot_log_with_lobf(
        εs,
        abs_S2_diff,
        "$(name)/s2_diff.pdf";
        annotation = slope -> L"\Gamma_{S^2}(\varepsilon) \sim \varepsilon^{%$slope}",
        axis = (;
            xlabel = L"\log_{10}{\,\varepsilon}",
            ylabel = L"\log_{10}{\,\Gamma_{S^2}(\varepsilon)}",
        ),
    )

    # plot_log_with_lobf(
    #     εs,
    #     pnorm(y_means; dims = 2),
    #     "$(name)/y_norm_mean.pdf";
    #     annotation = slope -> L"\Gamma_{E}(\varepsilon) \sim \varepsilon^{%$slope}",
    #     axis = (;
    #         xlabel = L"\log_{10}{\,\varepsilon}",
    #         ylabel = L"\log_{10}{\,\Gamma_{E}(\varepsilon)}",
    #     ),
    # )

    # plot_log_with_lobf(
    #     εs,
    #     pnorm(z_means; dims = 2),
    #     "$(name)/z_norm_mean.pdf";
    #     annotation = slope -> L"\Gamma_{E}(\varepsilon) \sim \varepsilon^{%$slope}",
    #     axis = (;
    #         xlabel = L"\log_{10}{\,\varepsilon}",
    #         ylabel = L"\log_{10}{\,\Gamma_{E}(\varepsilon)}",
    #     ),
    # )
end

function Σ_through_time(
    rels,
    model,
    ε,
    x₀,
    t₀,
    ts,
    dt,
    σ_label;
    hist_idxs = 1:length(ts),
    ode_solver = Euler(),
    plot_attrs...,
)
    T = maximum(ts)
    name = "$(model.name)_$(x₀)_[$(t₀),$(T)]"

    # Solve for the full deterministic trajectory
    det_prob = ODEProblem(model.velocity, x₀, (t₀, T))
    det_sol = solve(det_prob, ode_solver; dt = dt, dtmax = dt)

    traj_plot = plot()

    S²_vals = Vector{Float64}(undef, length(ts))
    sample_S²_vals = Vector{Float64}(undef, length(ts))
    ws = Vector{Vector{Float64}}(undef, length(ts))
    Σs = Vector{Matrix{Float64}}(undef, length(ts))
    for (i, t) in enumerate(ts)
        w, Σ = Σ_calculation(model, x₀, t₀, t, dt, 0.001, ode_solver)
        ws[i] = w
        Σs[i] = Σ
        S²_vals[i] = opnorm(Σ)
        sample_S²_vals[i] = opnorm(cov((rels[i, :, :] .- w) ./ ε; dims = 2))

        if i in hist_idxs
            histogram2d!(
                traj_plot,
                rels[i, 1, :],
                rels[i, 2, :];
                bins = 100,
                c = cgrad(PALETTE; rev = true),
            )
        end
    end

    # Plot the deterministic trajectory (above the histograms, below the SD bounds)
    plot!(
        traj_plot,
        det_sol;
        idxs = (1, 2),
        linecolor = :gray,
        linewidth = 0.5,
        grid = false,
        legend = false,
        xlabel = L"x_1",
        ylabel = L"x_2",
        plot_attrs...,
    )

    # Plot the covariance visualisations
    for i in hist_idxs
        bivariate_std_dev(
            ws[i],
            ε^2 * Σs[i];
            nσ = 2,
            plt = traj_plot,
            linecolor = :black,
            linewidth = 0.5,
        )
    end

    save_figure(traj_plot, "$(name)/time_traj_$(σ_label).pdf")

    p = scatter(
        ts,
        sample_S²_vals;
        markercolor = :black,
        label = L"Limiting ($S^2(x,t)$)",
        grid = false,
        xlabel = L"t",
        ylabel = "Covariance operator norm",
        legend_position = :topleft,
    )
    scatter!(p, ts, S²_vals; markercolor = :transparent, markerstrokecolor = :red, label = "Sample")
    save_figure(p, "$(name)/time_S2_$(σ_label).pdf")
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
function S²_grid_sets(
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
    # Compute the S² value for each initial condition, as the operator norm of Σ
    S²_grid = map(x₀_grid) do x
        opnorm(Σ_calculation(model, x, t₀, T, dt, dx, ode_solver)[2])
    end

    p = heatmap(xs, ys, log.(S²_grid)'; plot_attrs...)
    savefig(p, "output/s2_field$(fname_ext).pdf")

    # Extract robust sets and plot
    R = S²_grid .< threshold
    p = heatmap(xs, ys, R'; c = cgrad([:white, :lightskyblue]), cbar = false, plot_attrs...)

    savefig(p, "output/s2_robust$(fname_ext).pdf")
end