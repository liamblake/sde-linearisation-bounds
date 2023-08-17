using LinearAlgebra
using Random
using Statistics

using Distributions
using DifferentialEquations
using FileIO
using ProgressMeter
using PyPlot

# The underlying shell in which PyCall is running Python does not source .zshenv, and therefore the
# PATH is not updated to include the directory with latex.
# TODO: Actually fix the problem.
using PyCall
py"""
from os import environ
environ["PATH"] = f'/Users/a1742080/bin:{environ["PATH"]}'
# environ["PATH"] = f'/Users/liam/bin:{environ["PATH"]}'
"""

include("../pyplot_setup.jl")
save_dpi = 600

# The default PyPlot cycle of colours
pyplot_colors = rcParams["axes.prop_cycle"].by_key()["color"]

function generate_joint_sde_samples!(y_dest, l_dest, N, T, εs, ρs, u, ∇u, σ, F!, x₀, init_dist)
    # Set up joint system to solve
    w = Vector{Float64}(undef, 1)
    function u_joint!(du, x, _, t)
        F!(w, t)
        du[1] = u(x[1], t)
        du[2] = u(w[1], t) + ∇u(w[1], t) * (x[2] - w[1])
    end

    function σ_joint!(du, x, ε, t)
        F!(w, t)
        du[1] = ε * σ(x[1], t)
        du[2] = ε * σ(w[1], t)
        # du[1, 2] = 0.0
        # du[2, 2] = 0.0
    end

    W = WienerProcess(0.0, 0.0, 0.0)
    prob = SDEProblem(u_joint!, σ_joint!, [x₀, x₀], (0.0, T), 0.0; noise = W)
    # prob =
    #     SDEProblem(u_joint!, σ_joint!, [x₀, x₀], (0.0, T), 0.0; noise_rate_prototype = zeros(2, 2))

    # Generate samples
    p = Progress(length(ρs) * length(εs); desc = "Generating SDE samples...", showspeed = true)
    for (i, ε) in enumerate(εs)
        prob = remake(prob; p = ε)
        for (j, ρ) in enumerate(ρs)
            init = init_dist(ρ)
            ens = EnsembleProblem(prob; prob_func = function (p, _, _)
                xrel = rand(init)
                remake(p; u0 = [xrel, xrel])
            end)
            sol = solve(
                ens,
                SRIW1(),
                EnsembleThreads();
                trajectories = N,
                save_everystep = false,
                saveat = [T],
                abstol = min(ε^2, ρ^2),
            )

            rels = Array(sol)
            y_dest[:, j, i] = rels[1, 1, :]
            l_dest[:, j, i] = rels[2, 1, :]
            next!(p)
        end
    end
    finish!(p)
end

function bound_validation_1d(
    name,
    x₀,
    init_dist,
    T,
    u,
    ∇u,
    σ,
    F!,
    ∇F,
    εs,
    δs,
    N,
    n_ignore,
    line_j,
    hist_idxs;
    regenerate = false,
    msize = 12.5,
    lwidth = 1.0,
)

    # Solve the covariance ODE for the linearisation variance
    w = [0.0]
    function cov_u(v, _, t)
        F!(w, t)
        return 2 * ∇u(w[1], t) * v + σ(w[1], t)^2
    end
    prob = ODEProblem(cov_u, 0.0, (0.0, T))
    sol = solve(prob; saveat = [T])
    Σ = sol.u[end]

    # Distribution of the linearisation - computed exactly
    wT = [0.0]
    F!(wT, T)
    linear_dist = (δ, ε) -> Normal(wT[1], sqrt(∇F(T)^2 * δ^2 + ε^2 * Σ))

    # Pre-allocate storage of generated values
    y_rels = Array{Float64}(undef, N, length(δs), length(εs))
    l_rels = Array{Float64}(undef, N, length(δs), length(εs))

    if regenerate
        generate_joint_sde_samples!(y_rels, l_rels, N, T, εs, δs, u, ∇u, σ, F!, x₀, init_dist)

        # Save the realisatons
        save("data/$(name)_rels.jld2", "y_rels", y_rels, "l_rels", l_rels)

    else
        # Reload previously saved data
        dat = load("data/$(name)_rels.jld2")
        y_rels .= dat["y_rels"]
        l_rels .= dat["l_rels"]
    end

    # Compute first 4 moments of strong error
    rs = 1.0:4.0
    str_errs = Array{Float64}(undef, length(rs), length(δs), length(εs))
    for (i, r) in enumerate(rs)
        str_errs[i, :, :] = mean(@.(abs(y_rels - l_rels)^r); dims = 1)
    end

    rcParams["font.size"] = "14"

    # Save each plot for easy legend manipulation

    # Plot strong errors
    # Ignore the two largest values of ε - too large for the bound to be reliable.
    ε_plotting_range = εs[(n_ignore + 1):end]
    δ_plotting_range = δs[(n_ignore + 1):end]
    εs_fine = minimum(ε_plotting_range):(minimum(ε_plotting_range) / 10.0):maximum(ε_plotting_range)
    δs_fine = minimum(δ_plotting_range):(minimum(δ_plotting_range) / 10.0):maximum(δ_plotting_range)
    for (i, r) in enumerate(rs)
        # ε plots
        fig, ax = subplots()

        handlers = []
        k = 1
        for (j, δ) in enumerate(δs)
            # Only plot every second value of δ
            if line_j(j)
                # Fit a LoBF of the form Γ = a₁ + a₂ ε²ʳ
                X = hcat(ones(length(εs) - n_ignore), ε_plotting_range .^ (2 * r))

                # Calculate the least-squares estimate.
                coefs = inv(X' * X) * X' * str_errs[i, j, (n_ignore + 1):end]

                ax.scatter(
                    ε_plotting_range,
                    str_errs[i, j, (n_ignore + 1):end];
                    s = msize,
                    zorder = 2,
                )

                ax.plot(
                    εs_fine,
                    @.(coefs[1] + coefs[2] * εs_fine^(2 * r));
                    linewidth = lwidth,
                    zorder = 1,
                )

                # Legend entry
                push!(
                    handlers,
                    PyPlot.matplotlib.lines.Line2D(
                        [],
                        [];
                        color = pyplot_colors[k % length(pyplot_colors)],
                        marker = "o",
                        markersize = sqrt(msize),
                        linewidth = lwidth,
                        label = L"\Gamma_{%$(Int64(r))}\!\left(\varepsilon, 10^{%$(round(log10(δs[j]); digits = 2))}\right)",
                    ),
                )
                k += 1
            end
        end

        ax.set_xlabel(L"\varepsilon")
        ax.set_ylabel(L"\Gamma_{%$(Int64(r))}\!\left(\varepsilon, \rho\right)")

        ax.legend(; handles = handlers, loc = "center left", bbox_to_anchor = (1, 0.5))

        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.savefig(
            "output/$name/str_err_eps_r_$(r)_log.pdf";
            bbox_inches = "tight",
            dpi = save_dpi,
        )

        close(fig)

        # δ plots
        fig, ax = subplots()
        handlers = []
        k = 1
        for (j, ε) in enumerate(εs)
            # Only plot every second value of ε
            if line_j(j)
                # Fit a LoBF of the form Γ = a₁ + a₂ δ²ʳ
                X = hcat(ones(length(δs) - n_ignore), δ_plotting_range .^ (2 * r))

                # Calculate the least-squares estimate.
                coefs = inv(X' * X) * X' * str_errs[i, (n_ignore + 1):end, j]
                ax.plot(
                    δs_fine,
                    @.(coefs[1] + coefs[2] * δs_fine^(2 * r));
                    linewidth = lwidth,
                    zorder = 1,
                )

                ax.scatter(
                    δ_plotting_range,
                    str_errs[i, (n_ignore + 1):end, j];
                    s = msize,
                    zorder = 2,
                )

                # Legend entry
                push!(
                    handlers,
                    PyPlot.matplotlib.lines.Line2D(
                        [],
                        [];
                        color = pyplot_colors[k % length(pyplot_colors)],
                        marker = "o",
                        markersize = sqrt(msize),
                        linewidth = lwidth,
                        label = L"\Gamma_{%$(Int64(r))}\!\left(10^{%$(round(log10(εs[j]); digits = 2))}, \rho\right)",
                    ),
                )
                k += 1
            end
        end

        ax.set_xlabel(L"\rho")
        ax.set_ylabel(L"\Gamma_{%$(Int64(r))}\!\left(\varepsilon, \rho\right)")
        ax.legend(; handles = handlers, loc = "center left", bbox_to_anchor = (1, 0.5))

        ax.set_xscale("log")
        ax.set_yscale("log")

        fig.savefig(
            "output/$name/str_err_rho_r_$(r)_log.pdf";
            bbox_inches = "tight",
            dpi = save_dpi,
        )

        close(fig)
    end

    # Plot selected histograms
    rcParams["font.size"] = "10"
    fig, axs = subplots(4, 4)
    for (i, δ_idx) in enumerate(hist_idxs)
        # y-label
        axs[i, 1].yaxis.set_label_text(L"\rho = 10^{%$(Int64(floor(log10(δs[δ_idx]))))}")
        for (j, ε_idx) in enumerate(hist_idxs)
            axs[i, j].hist(
                y_rels[:, δ_idx, ε_idx];
                bins = 75,
                density = true,
                color = "gray",
                edgecolor = "gray",
            )

            # Plot the linearised solution
            xrange = minimum(y_rels[:, δ_idx, ε_idx]):0.001:maximum(y_rels[:, δ_idx, ε_idx])
            axs[i, j].plot(
                xrange,
                pdf.(linear_dist(δs[δ_idx], εs[ε_idx]), xrange),
                "r--";
                linewidth = 1.0,
            )

            if i == 1
                axs[i, j].set_title(L"\varepsilon = 10^{%$(round(log10(εs[ε_idx]); digits = 2))}")
            end

            # Hide ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        end
    end

    fig.savefig("output/$name/selected_hists.pdf"; bbox_inches = "tight", dpi = save_dpi)
    close(fig)
end