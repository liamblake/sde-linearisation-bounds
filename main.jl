using Random
using Statistics

using DifferentialEquations
using JLD
using LaTeXStrings
using Parameters
using Plots
using ProgressMeter

include("covariance.jl")
include("models.jl")
include("utils.jl")

"""
Generate N realisations of an SDE, returning a matrix of the final position.
"""
function sde_realisations(dest, vel!, σ!, N, d_y, d_W, y₀, t₀, T, dt)
    sde_prob = SDEProblem(vel!, σ!, y₀, (t₀, T), noise_rate_prototype=zeros(d_y, d_W))
    ens = EnsembleProblem(sde_prob)
    sol = solve(
        ens,
        EM(),
        EnsembleThreads(),
        trajectories=N,
        dt=dt,
        save_everystep=false,
    )

    # Only need the final position
    dest .= reduce(hcat, DifferentialEquations.EnsembleAnalysis.get_timepoint(sol, T))
    nothing
end


function plot_with_lines(x::Vector, y::Vector, filename::String, slope::AbstractFloat; kwargs...)
    # Find the line of best fit
    fit, coefs = lobf(x, y)
    slope = round(coefs[2], digits=2)
    make_main_plot = _ -> scatter(x, y, annotations=((0.25, 0.75), Plots.text(L"\Gamma_y^{(%$r)} \sim ε^{%$slope}"),), kwargs...)

    # Plot with line of best fit 
    p = make_main_plot()
    plot!(x, fit, linecolor=:red)
    save_figure(p, "$(filename)_lobf.pdf")

    # Plot with a line of fixed slope
    p = make_main_plot()
    plot!(linecolor=:red)
    save_figure(p, "$(filename)_lobf.pdf")


end

"""

If plot_histograms is true (the default), then histograms are plotted and saved, only if the model 
has dimension 2. Plotting and saving histograms is very slow for large N.
"""
function convergence_validation(
    model::Model,
    N::Int64;
    plot_histograms::Bool=true,
    reload_data::Bool=false,
    nosave::Bool=true
)
    @unpack name, d, velocity!, ∇u, Kᵤ, x₀s, t₀, T = model

	model_name = name
    println("Validation for $(name) model...")

		for x₀ in x₀s
			name = "$(model_name)_$(x₀)"

    # The universal step size. This needs to be small enough to overcome numerical issues for small values of ε
    dt = 1e-6

    # Calculate the deterministic trajectory. This is needed to form the limiting velocity field 
    println("Solving for deterministic trajectory...")
    det_prob = ODEProblem(velocity!, x₀, (t₀, T))
    det_sol = solve(det_prob, Euler(), dt=dt)
    w = last(det_sol.u)

    # Only attempt to plot histograms if the model dimension is 2
    plot_histograms *= (d == 2)

    # Set up as a joint system so the same noise realisation is used.
    function joint_system!(dx, x, _, t)
        velocity!(dx, x, NaN, t)
        dx[(d+1):(2*d)] = ∇u(det_sol(t), t) * x[(d+1):(2*d)]
        nothing
    end

    # Plot the deterministic trajectory
    p = plot(det_sol, idxs=(1, 2), xlabel=L"x_1", ylabel=L"x_2", color=:black, legend=false)
    save_figure(p, "$(name)/deterministic_trajectory.pdf")

    # Calculate the deviation covariance from the integral expression
    Σ = Σ_calculation(model, x₀, dt)
	# The maximum eigenvalue - the theoretical value for stochastic sensitivity
	S2 = eigmax(Matrix(Σ), permute = false, scale = false)

    rs = [1, 2, 3, 4]
    εs = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    nε = length(εs)

    # Keeping track across values of ε
    w_abs_diff = Vector{Float64}(undef, length(εs))
    y_abs_diff = Array{Float64}(undef, (length(rs), length(εs)))
    z_abs_diff = Array{Float64}(undef, (length(rs), length(εs)))
    z_mean_diff = Vector{Float64}(undef, length(εs))
	sample_S2s = Vector{Float64}(undef, length(εs))

    # Save ALL realisations of the limiting equation. The more data the merrier.
    all_limit_samples = zeros(d, N * nε)

    data_path = ε -> "data/$(name)_$(ε).jld"

    # For storing simulations - pre-allocate once and reuse
    joint_rels = Array{Float64}(undef, (2 * d, N))

    println("Generating realisations for values of ε...")
    @showprogress for (i, ε) in enumerate(εs)
        # See the joint system description in the script docstring
        function σ!(dW, _, _, _)
            dW .= 0.0
            dW[diagind(dW)] .= ε

            # TODO: Do not use a for loop here
            for j in 1:d
                dW[d+j, j] = 1.0
            end

            nothing
        end

        # Simulate from the y equation and the limiting equation simultaneously
        if reload_data
            # Load previously simulated data 
            joint_rels .= load(data_path(ε))["data"]

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
            if !nosave
                save(data_path(ε), "data", joint_rels)
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
		sample_S2s[i] = eigmax(S_z, permute = false, scale = false)

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
                c=cgrad(:spring, rev=true),
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
                z_rels[1, :],
                z_rels[2, :],
                bins=100,
                xlabel=L"z_1",
                ylabel=L"z_2",
                legend=false,
                cbar=true,
                c=cgrad(:spring, rev=true),
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

    log_εs = log10.(εs)

    for (j, r) in enumerate(rs)
        # Calculate the theoretical upper bound on the expectation
		log_D = log_Dᵣ(Float64(r), d, T, Kᵤ, 1)
		# println("Bounding constant: $(log_D)")


        vals = log10.(@view y_abs_diff[j, :])
        fit, coefs = lobf(log10.(εs), vals)
        slope = round(coefs[2], digits=2)
        p = scatter(
            log_εs,
            vals,
            xlabel=L"\log{\,\varepsilon}",
            ylabel=L"\log{\,\Gamma_y^{(%$r)}}",
            legend=false,
            annotations=(
                (0.25, 0.75),
                Plots.text(L"\Gamma_y^{(%$r)} \sim ε^{%$slope}"),
            ),
        )
        plot!(log10.(εs), fit, linecolor=:red)
        save_figure(p, "$(name)/y_diff_$(r).pdf")

        vals = log10.(@view z_abs_diff[j, :])
        fit, coefs = lobf(log10.(εs), vals)
        slope = round(coefs[2], digits=2)
        p = scatter(
            log_εs,
            vals,
            xlabel=L"\log{\,\varepsilon}",
            ylabel=L"\log{\,\Gamma_z^{(%$r)}}",
            legend=false,
            annotations=(
                (0.25, 0.75),
                Plots.text(L"\Gamma_z^{(%$r)} \sim ε^{%$slope}"),
            ),
        )

        plot!(log10.(εs), fit, linecolor=:red)
        save_figure(p, "$(name)/z_diff_$(r).pdf")

    end

	# Plot the difference in stochastic sensitivity
	abs_S2_diff = abs.(sample_S2s .- S2) 
	p = scatter(log10.(εs), abs_S2_diff, legend = false, xlabel = L"\log{\,\varepsilon}", ylabel = L"S^2")
	save_figure(p, "$(name)/s2_diff.pdf")

	p = scatter(log10.(εs), log10.(abs_S2_diff), legend = false, xlabel = L"\log{\,\varepsilon}", ylabel = L"\log{\,S^2}")
	save_figure(p, "$(name)/s2_diff_log.pdf")

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
            c=cgrad(:spring, rev=true),
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
        println("Plotted $(N*nε) realisations of the limiting SDE")
    end
end


end


function validate_all(N::Int64; reload_data::Bool=false, nosave::Bool=false)
    Random.seed!(20220805)
    convergence_validation(ex_rossby(), N, plot_histograms=true, reload_data=reload_data, nosave=nosave)
    # convergence_validation(ex_lorenz(), N, plot_histograms=false, reload_data=reload_data, nosave=nosave)
end
