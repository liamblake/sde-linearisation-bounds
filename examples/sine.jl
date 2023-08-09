using LinearAlgebra
using Random
using Statistics

using Distributions
using DifferentialEquations
using ProgressMeter
using PyPlot

include("../models.jl")
include("../pyplot_setup.jl")

Random.seed!(3259245)

outdir = "output/sine"

# Model definition
function u(x, _)
    return sin(x)
end

function ∇u(x, _)
    return cos(x)
end

function σ(_, _)
    return 1.0
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
F = t -> 2 * atan(exp(t) * tan(μ₀ / 2))
∇F = t -> exp(t) * sec(μ₀ / 2)^2 / (exp(2 * t)tan(μ₀ / 2)^2 + 1)

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
δs = [10^(-i) for i in 0.0:0.5:5.0]
εs = [10^(-i) for i in 0.0:0.5:5.0]

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
                dt = dt,
                trajectories = N,
                saveat = [T],
                abstol = ε^2,
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
for (i, r) in enumerate(rs)
    # ε plots
    fig, ax = subplots()
    for (j, δ) in enumerate(δs)
        ax.scatter(εs, str_errs[i, :, j]'; alpha = 0.5, label = L"\rho = %$(round(δ; digits = 2))")
    end
    ax.set_xscale("log")

    ax.set_xlabel(L"\varepsilon")
    ax.set_ylabel(L"\Gamma\!\left(\varepsilon, \rho\right)")
    # ax.legend()

    fig.savefig("$outdir/str_err_eps_r_$(r).pdf"; bbox_inches = "tight", dpi = save_dpi)

    ax.set_yscale("log")
    fig.savefig("$outdir/str_err_eps_r_$(r)_log.pdf"; bbox_inches = "tight", dpi = save_dpi)

    close(fig)

    # δ plots
    fig, ax = subplots()
    for (j, ε) in enumerate(εs)
        ax.scatter(
            δs,
            str_errs[i, j, :]';
            alpha = 0.5,
            label = L"\varepsilon = %$(round(ε; digits = 2))",
        )
    end

    ax.set_xscale("log")

    ax.set_xlabel(L"\rho")
    ax.set_ylabel(L"\Gamma\!\left(\varepsilon, \rho\right)")

    fig.savefig("$outdir/str_err_del_r_$(r).pdf"; bbox_inches = "tight", dpi = save_dpi)

    ax.set_yscale("log")
    fig.savefig("$outdir/str_err_del_r_$(r)_log.pdf"; bbox_inches = "tight", dpi = save_dpi)

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