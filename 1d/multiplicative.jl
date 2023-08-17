using LinearAlgebra
using Random
using Statistics

using Distributions
using DifferentialEquations
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

include("../models.jl")
# include("../pyplot_setup.jl")

Random.seed!(3259245)

outdir = "output/multiplicative"

# Model definition
function u(x, _)
    return 0.5 * x
end

function ∇u(_, _)
    return 0.5
end

function σ(x, _)
    return x
end

# The initial condition - a Gaussian centered around 0.5 with variance scaling by δ
μ₀ = 0.5
init_dist = δ -> Normal(μ₀, sqrt(δ))

# Time span
T = 1.5
dt = 0.001

# Number of samples
N = 10000

# Flow map and linearisation details
F = t -> exp(-0.5 * t) * μ₀
∇F = t -> exp(-0.5 * t)

# Solve the covariance ODE for the linearisation variance
function cov_u(v, _, t)
    return 2 * ∇u(F(t), t) * v + σ(F(t), t)^2
end
prob = ODEProblem(cov_u, 0.0, (0.0, T))
sol = solve(prob; saveat = [T])
Σ = sol.u[end]

# Distribution of the linearisation - computed exactly
linear_dist = (δ, ε) -> Normal(F(T), sqrt(∇F(T)^2 * δ + ε^2 * Σ))

# Range of parameters
δs = [10^(-i) for i in 0.0:0.25:3.0]
εs = [10^(-i) for i in 0.0:0.25:3.0]

# Set up joint system to solve
function u_joint!(du, x, _, t)
    du[1] = u(x[1], t)
    du[2] = u(F(t), t) + ∇u(F(t), t) * (x[2] - F(t))
end

function σ_joint!(du, x, ε, t)
    du[1] = σ(x[1], t)
    du[2] = σ(F(t), t)
    rmul!(du, ε)
end

W = WienerProcess(0.0, 0.0, 0.0)
prob = SDEProblem(u_joint!, σ_joint!, [μ₀, μ₀], (0.0, T), 0.0; noise = W)

# Pre-allocate storage of generated values
y_rels = Array{Float64}(undef, N, length(δs), length(εs))
l_rels = Array{Float64}(undef, N, length(δs), length(εs))

# Generate samples
begin
    p = Progress(length(δs) * length(εs); desc = "Generating SDE samples...", showspeed = true)
    for (i, ε) in enumerate(εs)
        prob = remake(prob; p = ε)
        for (j, δ) in enumerate(δs)
            init = init_dist(δ)
            ens = EnsembleProblem(prob; prob_func = function (p, _, _)
                xrel = rand(init)
                remake(p; u0 = [xrel, xrel])
            end)
            sol = solve(
                ens,
                SRIW1(),
                EnsembleThreads();
                # adaptive = false,
                # dt = ε^(1.5),
                trajectories = N,
                saveat = [T],
            )

            rels = Array(sol)
            y_rels[:, j, i] = rels[1, 1, :]
            l_rels[:, j, i] = rels[2, 1, :]
            next!(p)
        end
    end
    finish!(p)
end

# Compute first 4 moments of strong error
rs = 1.0:4.0
str_errs = Array{Float64}(undef, length(rs), length(δs), length(εs))
for (i, r) in enumerate(rs)
    str_errs[i, :, :] = mean(@.(abs(y_rels - l_rels)^r); dims = 1)
end

# Plot strong errors
# Ignore the two largest values of ε - too large for the bound to be reliable.
n_ignore = 1
ε_plotting_range = εs[(n_ignore + 1):end]
δ_plotting_range = δs[(n_ignore + 1):end]
εs_fine = minimum(ε_plotting_range):0.001:maximum(ε_plotting_range)
δs_fine = minimum(δ_plotting_range):0.001:maximum(δ_plotting_range)
for (i, r) in enumerate(rs)
    # ε plots
    fig, ax = subplots()
    for (j, δ) in enumerate(δs)
        # Only plot every second value of δ
        if j % 2 == 0
            # Fit a LoBF of the form Γ = a₂ ε²ʳ + a₃ εʳ
            X = hcat(ε_plotting_range .^ (2 * r), ε_plotting_range .^ (r))

            # Calculate the least-squares estimate.
            coefs = inv(X' * X) * X' * str_errs[i, j, (n_ignore + 1):end]

            ax.scatter(
                ε_plotting_range,
                str_errs[i, j, (n_ignore + 1):end];
                alpha = 0.5,
                label = L"\Gamma_{%$(Int64(r))}\!\left(\varepsilon, 10^{%$(Int64(floor(log10(δ))))}\right) =  %$(round(coefs[1], digits = 3)) \varepsilon^{%$(Int64(2 * r))} + %$(round(coefs[2], digits = 3)) \varepsilon^{%$(Int64(r))}",
            )

            ax.plot(εs_fine, @.(coefs[1] * εs_fine^(2 * r) + coefs[2] * εs_fine^(r)))
        end
    end

    ax.set_xlabel(L"\varepsilon")
    ax.set_ylabel(L"\Gamma_{%$(Int64(r))}\!\left(\varepsilon, \rho\right)")
    ax.legend(; loc = "center left", bbox_to_anchor = (1.0, 0.5))

    fig.savefig("$outdir/str_err_eps_r_$(r)_raw.pdf"; bbox_inches = "tight", dpi = save_dpi)

    ax.set_xscale("log")
    fig.savefig("$outdir/str_err_eps_r_$(r).pdf"; bbox_inches = "tight", dpi = save_dpi)

    ax.set_yscale("log")
    fig.savefig("$outdir/str_err_eps_r_$(r)_log.pdf"; bbox_inches = "tight", dpi = save_dpi)

    close(fig)

    # δ plots
    fig, ax = subplots()
    for (j, ε) in enumerate(εs)
        # Only plot every second value of ε
        if j % 2 == 0
            # Fit a LoBF of the form Γ = a₁ + a₂ δʳ
            X = hcat(ones(length(δs) - n_ignore), δ_plotting_range .^ r)

            # Calculate the least-squares estimate.
            coefs = inv(X' * X) * X' * str_errs[i, (n_ignore + 1):end, j]

            ax.scatter(
                δ_plotting_range,
                str_errs[i, (n_ignore + 1):end, j];
                alpha = 0.5,
                label = L"\Gamma_{%$(Int64(r))}\!\left(10^{%$(Int64(floor(log10(ε))))}, \rho\right) = %$(round(coefs[1]; digits = 4)) + %$(round(coefs[2]; digits = 4)) \rho^{%$(Int64(r))}",
            )

            ax.plot(δs_fine, @.(coefs[1] + coefs[2] * δs_fine^r))
        end
    end

    ax.set_xlabel(L"\rho")
    ax.set_ylabel(L"\Gamma_{%$(Int64(r))}\!\left(\varepsilon, \rho\right)")
    ax.legend(; loc = "center left", bbox_to_anchor = (1, 0.5))

    fig.savefig("$outdir/str_err_rho_r_$(r)_raw.pdf"; bbox_inches = "tight", dpi = save_dpi)

    ax.set_xscale("log")
    fig.savefig("$outdir/str_err_rho_r_$(r).pdf"; bbox_inches = "tight", dpi = save_dpi)

    ax.set_yscale("log")
    fig.savefig("$outdir/str_err_rho_r_$(r)_log.pdf"; bbox_inches = "tight", dpi = save_dpi)

    close(fig)
end

# Plot selected histograms
rcParams["font.size"] = "10"
begin
    fig, axs = subplots(4, 4)
    plot_vals = [1, 3, 7, 11]
    for (i, δ_idx) in enumerate(plot_vals)
        # y-label
        axs[i, 1].yaxis.set_label_text(L"\rho = 10^{%$(Int64(floor(log10(δs[δ_idx]))))}")
        for (j, ε_idx) in enumerate(plot_vals)
            axs[i, j].hist(
                y_rels[:, δ_idx, ε_idx];
                bins = 75,
                density = true,
                color = "gray",
                edgecolor = "gray",
            )

            # # Plot a Gaussian with the sample mean and variance
            # axs[i, j].plot(
            #     xrange,
            #     pdf.(Normal(mean(y_rels[:, δ_idx, ε_idx]), std(y_rels[:, δ_idx, ε_idx])), xrange),
            #     "b--";
            #     linewidth = 1.0,
            # )

            # Plot the linearised solution
            xrange = minimum(y_rels[:, δ_idx, ε_idx]):0.001:maximum(y_rels[:, δ_idx, ε_idx])
            axs[i, j].plot(
                xrange,
                pdf.(linear_dist(δs[δ_idx], εs[ε_idx]), xrange),
                "r--";
                linewidth = 1.0,
            )

            if i == 1
                axs[i, j].set_title(L"\varepsilon = 10^{%$(Int64(floor(log10(εs[ε_idx]))))}")
            end

            # Hide ticks
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        end
    end

    fig.savefig("$outdir/selected_hists.pdf"; bbox_inches = "tight", dpi = save_dpi)
    close(fig)
end