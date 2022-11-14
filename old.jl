using LinearAlgebra
using Statistics

using JLD
using LaTeXStrings
using Parameters
using Plots

include("covariance.jl")
include("models.jl")
include("sde.jl")

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
		p,
		x::AbstractVector,
		y::AbstractVector;
		intercept::Bool = true,
		annotation::Function = nothing,
	)

Add a line of best fit to a given plot, with an optional annotation.
"""
function add_lobf_to_plot!(
    p,
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

"""
Helper function to save figures
"""
function save_figure(p, fname::String; show_print::Bool=true)
    path = "output/$(fname)"
    if show_print
        println("Saving figure to $(path)")
    end
    savefig(p, path)
end


"""

"""
function full_analysis(model_name::String, εs, rs)
    @unpack name, d, velocity!, ∇u, Kᵤ = model

    model_name = name
    !quiet && println("Validation for $(name) model...")

    # Only attempt to plot histograms if the model dimension is 2
    plot_histograms = (d == 2)

    rs = [1, 2, 3, 4]
    εs = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    log_εs = log10.(εs)
    nε = length(εs)
    nr = length(rs)

    # The universal step size. This needs to be small enough to overcome numerical issues for
    # small values of ε. A good rule of thumb is dt = min(ε) / 10.
    dt = 1e-5

    # Pre-allocate storage - these are reused across initial conditions
    w_abs_diff = Vector{Float64}(undef, nε)
    y_abs_diff = Array{Float64}(undef, (nr, nε))
    z_abs_diff = Array{Float64}(undef, (nr, nε))
    z_mean_diff = Vector{Float64}(undef, nε)
    sample_S2s = Vector{Float64}(undef, nε)

    # Save ALL realisations of the limiting equation. The more data the merrier.
    all_limit_samples = zeros(d, N * nε)

    # For storing simulations - pre-allocate once and reuse
    joint_rels = Array{Float64}(undef, (2 * d, N))

    for x₀ in x₀s
        name = "$(model_name)_$(x₀)"

        # Calculate the deterministic trajectory. This is needed to form the limiting velocity field
        !quiet && println("Solving for deterministic trajectory...")
        det_prob = ODEProblem(velocity!, x₀, (t₀, T))
        det_sol = solve(det_prob, Euler(), dt=dt)
        w = last(det_sol.u)

        # Set up as a joint system so the same noise realisation is used.
        function joint_system!(dx, x, _, t)
            velocity!(dx, x, NaN, t)
            dx[(d+1):(2*d)] = ∇u(det_sol(t), t) * x[(d+1):(2*d)]
            nothing
        end

        # Calculate the deviation covariance from the integral expression
        Σ = Σ_calculation(model, x₀, t₀, T, dt)
        # The maximum eigenvalue - the theoretical value for stochastic sensitivity
        S2 = eigmax(Matrix(Σ), permute=false, scale=false)

        !quiet && println("Generating realisations for values of ε...")
        for (i, ε) in enumerate(εs)
            !quiet && println("\t ε = $(ε) \td ($(i)/$(nε))")

            # See the joint system description in the script docstring
            function σ!(dW, _, _, _)
                dW .= 0.0
                hls

                # TODO: Do not use a for loop here
                for j = 1:d
                    dW[j, j] = ε
                    dW[d+j, j] = 1.0
                end

                nothing
            end

            # Simulate from the y equation and the limiting equation simultaneously
            # If attempt_reload is true and a data file exists, load the data. Otherwise,
            # generate new data and save.
            data_path = "data/$(name)_$(ε).jld"
            if attempt_reload && isfile(data_path)
                # Load previously simulated data
                joint_rels .= load(data_path)["data"]
            else
                sde_realisations(
                    joint_rels,
                    joint_system!,
                    σ!,
                    N,
                    2 * d,
                    d,
                    vcat(x₀, zeros(d)),
                    t₀,
                    T,
                    dt,
                )
                if save_on_generation
                    save(data_path, "data", joint_rels)
                end
            end
            y_rels = @view joint_rels[1:d, :]
            limit_rels = @view joint_rels[(d+1):(2*d), :]
            all_limit_samples[:, ((i-1)*N+1):(i*N)] = limit_rels

            # Calculate the corresponding z_ε realisations
            z_rels = 1 / ε .* (y_rels .- w)

            # Mainly for diagnostics - calculate the distance between each realisation and w.
            w_abs_diff[i] = mean(pnorm(y_rels .- w, dims=1))

            # Also diagnostics
            z_mean_diff[i] = mean(pnorm(z_rels, dims=1))

            # Calculate the normed distance between each pair of realisations
            # The overall mean provides an estimate of 𝔼[|y_ε - w - ε * z|ʳ]
            y_diffs = pnorm(y_rels .- w .- ε * limit_rels, dims=1)

            # Calculate the normed distance between the scaled deviation and the solution,
            # in order to estimate 𝔼[|z_ε - z|ʳ]
            z_diffs = pnorm(z_rels .- limit_rels, dims=1)

            # Calculate the sample covariance matrix
            s_mean_y = mean(y_rels, dims=2)
            S_y = 1 / (N - 1) .* (y_rels .- s_mean_y) * (y_rels .- s_mean_y)'
            s_mean_z = mean(z_rels, dims=2)
            S_z = 1 / (N - 1) .* (z_rels .- s_mean_z) * (z_rels .- s_mean_z)'
            # Calculate empirical stochastic sensitivity
            sample_S2s[i] = eigmax(S_z, permute=false, scale=false)

            for (j, r) in enumerate(rs)
                y_abs_diff[j, i] = mean(y_diffs .^ r)
                z_abs_diff[j, i] = mean(z_diffs .^ r)
            end

            if plot_histograms
                # Plot a histogram of the realisations for the smallest value of ε
                p = histogram2d(
                    y_rels[1, :],
                    y_rels[2, :],
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
                save_plots && save_figure(p, "$(name)/y_histogram_$(ε).pdf", show_print=false)

                # The scaled deviations z_ε
                p = histogram2d(
                    z_rels[1, :],
                    z_rels[2, :],
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


        if save_plots
            # Plot the deterministic trajectory
            p = plot(
                det_sol,
                idxs=(1, 2),
                xlabel=L"x_1",
                ylabel=L"x_2",
                color=:black,
                legend=false,
            )
            save_figure(p, "$(name)/deterministic_trajectory.pdf")

            for (j, r) in enumerate(rs)
                vals = log10.(@view y_abs_diff[j, :])
                p = scatter(
                    log_εs,
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
                    log_εs,
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
            p = scatter(log10.(εs), w_abs_diff, legend=false)
            save_figure(p, "$(name)/diagnostics_y_w.pdf")

            p = scatter(log10.(εs), z_mean_diff, legend=false)
            save_figure(p, "$(name)/diagnostics_z_mean.pdf")

            if plot_histograms
                # Plot a histogram of all the realisations of the limiting SDE solution.
                # Overlay the first two standard deviation bounds from the sample covariance and Σ calculated
                # from the integral expression.
                S = 1 / (nε * N) * all_limit_samples * all_limit_samples'
                p = histogram2d(
                    all_limit_samples[1, :],
                    all_limit_samples[2, :],
                    bins=100,
                    xlabel=L"z_1",
                    ylabel=L"z_2",
                    legend=true,
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
                    mean(all_limit_samples, dims=2),
                    S,
                    nσ=2,
                    plt=p,
                    colour=:red,
                    linestyle=:dash,
                    label="Empirical",
                )
                save_figure(p, "$(name)/limiting_histogram.pdf")
                !quiet && println("Plotted $(N*nε) realisations of the limiting SDE")
            end
        end
    end


end
