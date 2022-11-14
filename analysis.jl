using LinearAlgebra

using Plots
using Statistics

include("covariance.jl")

# The color palette to use throughout
PALETTE = :spring

"""
	pnorm(A::AbstractArray, dims; p::Real = 2)

Calculate the p-norm along the given dimension of a multidimensional array.
"""
function pnorm(A::AbstractArray; dims=1, p::Real=2)
    f = a -> norm(a, p)
    return mapslices(f, A, dims=dims)
end

"""
save_figure(p::Plots.Plot, fname::String; show_print::Bool=true)

Helper function to save a Plots.plot object to the given file fname. If show_print is true, then
a line is printed indicating where the figure has been saved.
"""
function save_figure(p::Plots.Plot, fname::String; show_print::Bool=true)
    path = "output/$(fname)"
    if show_print
        println("Saving figure to $(path)")
    end
    savefig(p, path)
end


"""
	bivariate_gaussian_std_dev(μ, Σ; nσ = 1, plt = plot(), ...)

Plot the n standard-deviation regions of a bivariate Gaussian distribution
with mean μ and covariance matrix Σ. The number of regions plotted is specified
by nσ.
"""
function bivariate_std_dev(μ, Σ; nσ=1, plt=plot(), colour=:black, args...)
    # Calculate the first two principal axes of the covariance matrix
    # These correspond to the major and minor axes of the ellipse
    evals, evecs = eigen(Σ)

    # Angle of rotation - use the principal axis
    θ = atan(evecs[2, 1], evecs[1, 1])

    # Magnitude of major and minor axes
    a, b = sqrt.(evals[1:2])

    # Plot each contour
    for n = 1:nσ
        # Parametric equations for the resulting ellipse
        # TODO: Should be a way to calculate this by operating directly on the eigenvectors
        # i.e. x = cos(θ), y = sin(θ)
        x = t -> n * (a * cos(t) * cos(θ) - b * sin(t) * sin(θ)) + μ[1]
        y = t -> n * (a * cos(t) * sin(θ) + b * sin(t) * cos(θ)) + μ[2]

        plot!(x, y, 0, 2π, linecolor=colour; args...)
    end

    # Also plot the mean
    scatter!([μ[1]], [μ[2]], markersize=3, markercolor=colour, label="")

    return plt

end

"""
	lobf(x::AbstractVector, y::AbstractVector; intercept::Bool = false)

Given some 2-dimensional data, calculate a line of best fit, with a least-squares estimate.
An intercept is included by default.
"""
function lobf(x::AbstractVector, y::AbstractVector; intercept::Bool=true)
    n = length(x)

    if intercept
        X = hcat(ones(n), x)
    else
        X = reshape(x, (:, 1))
    end

    # Calculate the least-squares estimate.
    coefs = inv(X' * X) * X' * y

    # Fit the line to each datapoint, for plotting
    return X * coefs, coefs
end



"""
	add_lobf_to_plot!(
		p::Plots.Plot,
		x::AbstractVector,
		y::AbstractVector;
		intercept::Bool = true,
		annotation::Function = nothing,
	)

Add a line of best fit to a given plot, with an optional annotation.
"""
function add_lobf_to_plot!(
    p::Plots.Plot,
    x::AbstractVector,
    y::AbstractVector;
    intercept::Bool=true,
    annotation::Function=nothing
)
    fit, coefs = lobf(x, y; intercept=intercept)
    slope = round(coefs[2], digits=2)
    plot!(p, x, fit, linecolor=:red, linestyle=:dash)

    # Add an annotation with the given label
    if annotation !== nothing
        annotate!(p, [((0.25, 0.75), Plots.text(annotation(slope)))])
    end
end

function full_analysis(y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, εs, rs, dt)
    @unpack x₀, t₀, T = space_time
    name = "$(model.name)_$(x₀)_[$(t₀),$(T)]"

    # Calculate the deviation covariance from the integral expression
    w, Σ = Σ_calculation(model, x₀, t₀, T, dt)

    # Theoretical stochastic sensitivity - the maximum eigenvalue of Σ
    S2 = eigmax(Matrix(Σ), permute=false, scale=false)

    nε = length(εs)
    nr = length(rs)

    # Preallocate storage of results
    y_abs_diff = Array{Float64}(undef, (nr, nε))
    z_abs_diff = Array{Float64}(undef, (nr, nε))
    z_mean_diff = Vector{Float64}(undef, nε)
    sample_S2s = Vector{Float64}(undef, nε)

    # Only plot histograms if working in 2D
    plot_histograms = (model.d == 2)

    for (i, ε) in enumerate(εs)
        # Diagnostics - mean should be zero
        z_mean_diff[i] = mean(pnorm(z_rels[i, :, :], dims=1))

        # Calculate the normed distance between each pair of realisations
        # The overall mean provides an estimate of 𝔼[|y_ε - w - ε * z|ʳ]
        y_diffs = pnorm(y_rels[i, :, :] .- gauss_y_rels[i, :, :], dims=1)

        # Calculate the normed distance between the scaled deviation and the solution,
        # in order to estimate 𝔼[|z_ε - z|ʳ]
        z_diffs = pnorm(z_rels[i, :, :] .- gauss_z_rels[i, :, :], dims=1)

        # Calculate the sample covariance matrix
        s_mean_y = mean(y_rels[i, :, :], dims=2)
        S_y = 1 / (N - 1) .* (y_rels[i, :, :] .- s_mean_y) * (y_rels[i, :, :] .- s_mean_y)'
        s_mean_z = mean(z_rels[i, :, :], dims=2)
        S_z = 1 / (N - 1) .* (z_rels[i, :, :] .- s_mean_z) * (z_rels[i, :, :] .- s_mean_z)'
        # Calculate empirical stochastic sensitivity
        sample_S2s[i] = eigmax(S_z, permute=false, scale=false)

        for (j, r) in enumerate(rs)
            y_abs_diff[j, i] = mean(y_diffs .^ r)
            z_abs_diff[j, i] = mean(z_diffs .^ r)
        end

        if plot_histograms
            # Plot a histogram of the realisations for the smallest value of ε
            p = histogram2d(
                y_rels[i, 1, :],
                y_rels[i, 2, :],
                bins=100,
                xlabel=L"y_1",
                ylabel=L"y_2",
                legend=false,
                cbar=true,
                c=cgrad(PALETTE, rev=true),
                label="",
            )
            p = bivariate_std_dev(
                w,
                ε^2 * Σ,
                nσ=2,
                plt=p,
                colour=:black,
                linestyle=:solid,
                label="Theory",
            )
            p = bivariate_std_dev(
                s_mean_y,
                S_y,
                nσ=2,
                plt=p,
                colour=:red,
                linestyle=:dash,
                label="Empirical",
            )
            save_figure(p, "$(name)/y_histogram_$(ε).pdf", show_print=false)

            # The scaled deviations z_ε
            p = histogram2d(
                z_rels[i, 1, :],
                z_rels[i, 2, :],
                bins=100,
                xlabel=L"z_1",
                ylabel=L"z_2",
                legend=false,
                cbar=true,
                c=cgrad(PALETTE, rev=true),
                label="",
            )
            p = bivariate_std_dev(
                [0, 0],
                Σ,
                nσ=2,
                plt=p,
                colour=:black,
                linestyle=:solid,
                label="Theory",
            )
            p = bivariate_std_dev(
                mean(z_rels, dims=2),
                S_z,
                nσ=2,
                plt=p,
                colour=:red,
                linestyle=:dash,
                label="Empirical",
            )
            save_figure(p, "$(name)/z_histogram_$(ε).pdf", show_print=false)

        end

    end

    for (j, r) in enumerate(rs)
        vals = log10.(@view y_abs_diff[j, :])
        p = scatter(
            log10.(εs),
            vals,
            xlabel=L"\log{\,\varepsilon}",
            ylabel=L"\log{\,\Gamma_y^{(%$r)}(\varepsilon)}",
            legend=false,
        )
        save_figure(p, "$(name)/y_diff_$(r).pdf")

        add_lobf_to_plot!(
            p,
            log10.(εs),
            vals,
            annotation=slope ->
                L"\Gamma_y^{(%$r)}(\varepsilon) \sim \varepsilon^{%$slope}",
        )
        save_figure(p, "$(name)/y_diff_$(r)_lobf.pdf")

        vals = log10.(@view z_abs_diff[j, :])
        p = scatter(
            log10.(εs),
            vals,
            xlabel=L"\log{\,\varepsilon}",
            ylabel=L"\log{\,\Gamma_z^{(%$r)}(\varepsilon)}",
            legend=false,
        )
        save_figure(p, "$(name)/z_diff_$(r).pdf")

        add_lobf_to_plot!(
            p,
            log10.(εs),
            vals,
            annotation=slope ->
                L"\Gamma_z^{(%$r)}(\varepsilon) \sim \varepsilon^{%$slope}",
        )
        save_figure(p, "$(name)/z_diff_$(r)_lobf.pdf")

    end

    # Plot the difference in stochastic sensitivity
    abs_S2_diff = abs.(sample_S2s .- S2)
    p = scatter(
        log10.(εs),
        abs_S2_diff,
        legend=false,
        xlabel=L"\log{\,\varepsilon}",
        ylabel=L"\Gamma_{S^2}(\varepsilon)",
    )
    save_figure(p, "$(name)/s2_diff.pdf")

    p = scatter(
        log10.(εs),
        log10.(abs_S2_diff),
        legend=false,
        xlabel=L"\log{\,\varepsilon}",
        ylabel=L"\log{\,\Gamma_{S^2}(\varepsilon)}",
    )
    save_figure(p, "$(name)/s2_diff_log.pdf")

    add_lobf_to_plot!(
        p,
        log10.(εs),
        log10.(abs_S2_diff),
        annotation=slope -> L"\Gamma_{S^2}(\varepsilon) \sim \varepsilon^{%$slope}",
    )
    save_figure(p, "$(name)/s2_diff_log_lobf.pdf")

    # Plot the difference between the realisations of the y SDE and w. Should be going to zero
    # as epsilon gets smaller. Use this to check whether the timestep size is not small enough.
    # Mainly for diagnostics.
    p = scatter(log10.(εs), z_mean_diff, legend=false)
    save_figure(p, "$(name)/diagnostics_z_mean.pdf")

end