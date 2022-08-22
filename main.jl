using Statistics

using DifferentialEquations
using JLD
using Plots
using ProgressMeter

include("covariance.jl")
include("models.jl")
include("utils.jl")

"""
Generate N realisations of an SDE, returning a matrix of the final position.
"""
function sde_realisations(vel, σ, N, n, y₀, t₀, T, dt)
    sde_prob = SDEProblem(vel, σ, y₀, (t₀, T), noise_rate_prototype=zeros(2 * n, n))
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
    return reduce(hcat, DifferentialEquations.EnsembleAnalysis.get_timepoint(sol, T))

end


function convergence_validation(
    model::Model,
    N::Int64;
    reload_data::Bool=false,
    nosave::Bool=true
)
    println("Validation for $(model.name) model...")

    # The universal step size. This needs to be small enough to overcome numerical issues for small values of ε
    dt = 1e-6

    # Calculate the deterministic trajectory. This is needed to form the limiting velocity field 
    println("Solving for deterministic trajectory...")
    det_prob = ODEProblem(model.velocity!, model.x₀, (model.t₀, model.T))

    # Only interested in the final position of the deterministic trajectory
    # Telling the solver this makes things faster
    det_sol = solve(det_prob, Euler(), dt=dt)
    w = last(det_sol.u)

    n = model.d

    # Set up as a joint system so the same noise realisation is used.
    function joint_system!(dx, x, _, t)
        model.velocity!(dx, x, NaN, t)
        dx[(n+1):(2*n)] = model.∇u(det_sol(t), t) * x[(n+1):(2*n)]
        nothing
    end

    # Plot the deterministic trajectory
    p = plot(det_sol, vars=(1, 2), color=:black, legend=false)
    save_figure(p, "$(model.name)/deterministic_trajectory.pdf")

    # Calculate the deviation covariance from the integral expression
    Σ = Σ_calculation(model, dt, 0.0001)

    rs = [1, 2, 3, 4]
    εs = [0.5, 0.1] #, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]

    nε = length(εs)

    # Keeping track across values of ε
    w_abs_diff = zeros(length(εs))
    y_abs_diff = zeros(length(rs), length(εs))
    z_abs_diff = zeros(length(rs), length(εs))
    z_mean_diff = zeros(length(εs))

    # Save ALL realisations of the limiting equation. The more data the merrier.
    all_limit_samples = zeros(n, N * nε)

    data_path = ε -> "../data/sde_realisations/$(model.name)_$(ε).jld"

    println("Generating realisations for values of ε...")
    @showprogress for (i, ε) in enumerate(εs)
        # See the joint system description in the script docstring
        function σ!(dW, _, _, _)
            dW[1, 1] = ε
            dW[2, 2] = ε
            dW[1, 2] = 0.0
            dW[2, 1] = 0.0

            dW[3, 1] = 1.0
            dW[4, 2] = 1.0
            dW[3, 2] = 0.0
            dW[4, 1] = 0.0

            nothing
        end

        # Simulate from the y equation and the limiting equation simultaneously
        if reload_data
            # Load previously simulated data 
            joint_rels = load(data_path(ε))["data"]

        else
            joint_rels = sde_realisations(
                joint_system!,
                σ!,
                N,
                n,
                vcat(model.x₀, [0.0, 0.0]),
                model.t₀,
                model.T,
                dt,
            )
            if !nosave
                save(data_path(ε), "data", joint_rels)
            end
        end
        y_rels = @view joint_rels[1:n, :]
        limit_rels = @view joint_rels[(n+1):(2*n), :]
        all_limit_samples[:, ((i-1)*N+1):(i*N)] = limit_rels

        # Calculate the corresponding z_ε realisations
        z_rels = 1 / ε * (y_rels .- w)

        # Mainly for diagnostics - calculate the distance between each realisation and w.
        w_abs_diff[i] = mean(pnorm(y_rels .- w, dims=1))

        # Also diagnostics
        z_mean_diff[i] = mean(pnorm(z_rels, dims=1))

        # Calculate the normed distance between each pair of realisations
        # The overall mean provides an estimate of 𝔼[|y_ε - w - ε * z|ʳ]
        y_diffs = pnorm(y_rels .- w .- ε * limit_rels, dims=1)

        # Calculate the normed distance between the scaled deviation and the solution,
        # in order to estimate 𝔼[|z_ε - z|ʳ]
        z_diffs = pnorm(z_rels - limit_rels, dims=1)

        for (j, r) in enumerate(rs)
            y_abs_diff[j, i] = mean(y_diffs .^ r)
            z_abs_diff[j, i] = mean(z_diffs .^ r)
        end

        if n == 2
            # Plot a histogram of the realisations for the smallest value of ε
            s_mean = mean(y_rels, dims=2)
            S = 1 / (N - 1) * (y_rels .- s_mean) * (y_rels .- s_mean)'
            p = histogram2d(
                y_rels[1, :],
                y_rels[2, :],
                bins=100,
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
                s_mean,
                S,
                nσ=2,
                plt=p,
                colour=:red,
                linestyle=:dash,
                label="Empirical",
            )
            save_figure(p, "$(model.name)/y_histogram_$(ε).pdf", show_print=false)

            # The scaled deviations z_ε
            S = 1 / N * z_rels * z_rels'
            p = histogram2d(
                z_rels[1, :],
                z_rels[2, :],
                bins=100,
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
                S,
                nσ=2,
                plt=p,
                colour=:red,
                linestyle=:dash,
                label="Empirical",
            )
            save_figure(p, "$(model.name)/z_histogram_$(ε).pdf", show_print=false)
        end

    end

    log_εs = log10.(εs)

    for (j, r) in enumerate(rs)
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
        save_figure(p, "$(model.name)/y_diff_$(r).pdf")

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
        save_figure(p, "$(model.name)/z_diff_$(r).pdf")

    end
    return

    # Plot the difference between the realisations of the y SDE and w. Should be going to zero 
    # as epsilon gets smaller. Use this to check whether the timestep size is not small enough.
    # Mainly for diagnostics.
    p = scatter(log10.(εs), w_abs_diff, legend=false)
    save_figure(p, "$(model.name)/diagnostics_y_w.pdf")

    p = scatter(log10.(εs), z_mean_diff, legend=false)
    save_figure(p, "$(model.name)/diagnostics_z_mean.pdf")

    if n == 2
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
        save_figure(p, "$(model.name)/limiting_histogram.pdf")
        println("Plotted $(N*nε) realisations of the limiting SDE")
    end


end

"""
	ex_rossby()

Two dimensional example: perturbed Rossby wave.
"""
function ex_rossby()::Model
    # Velocity field parameters
    A = 1.0
    c = 0.5
    K = 4.0
    l₁ = 2.0
    c₁ = π
    k₁ = 1.0
    ϵ = 0.3
    # The velocity field, with an in-place update.
    # Much faster this way: 
    # https://diffeq.sciml.ai/stable/tutorials/faster_ode_example/#Example-Accelerating-a-Non-Stiff-Equation:-The-Lorenz-Equation
    function rossby!(dx, x, _, t)
        dx[1] =
            c - A * sin(K * x[1]) * cos(x[2]) +
            ϵ * l₁ * sin(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
        dx[2] =
            A * K * cos(K * x[1]) * sin(x[2]) +
            ϵ * k₁ * cos(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
        nothing
    end


    # The velocity gradient matrix is known exactly
    ∇u =
        (x, t) -> [
            -A*K*cos(K * x[1])*cos(x[2])+ϵ*k₁*l₁*cos(k₁ * (x[1] - c₁ * t))*cos(l₁ * x[2]) A*sin(K * x[1])*sin(x[2])-ϵ*l₁^2*sin(k₁ * (x[1] - c₁ * t))*sin(l₁ * x[2])
            -A*K^2*sin(K * x[1])*sin(x[2])-ϵ*k₁^2*sin(k₁ * (x[1] - c₁ * t))*sin(l₁ * x[2]) A*K*cos(K * x[1])*cos(x[2])+ϵ*k₁*l₁*cos(k₁ * (x[1] - c₁ * t))*cos(l₁ * x[2])
        ]

    # Time and space parameters
    x₀ = [0.0, 1.0]
    t₀ = 0.0
    T = 1.0

    return Model("rossby", 2, rossby!, ∇u, x₀, t₀, T)
end

"""
	ex_lorenz()

40-dimensional example: Lorenz 96 system. Currently unused
"""
function ex_lorenz()::Model
    # Parameters
    d = 8
    F = 8

    # In-place velocity field
    function lorenz!(dx, x, _, _)

        # 3 edge cases explicitly
        @inbounds dx[1] = (x[2] - x[d-1]) * x[d] - x[1] + F
        @inbounds dx[2] = (x[3] - x[d]) * x[1] - x[2] + F
        @inbounds dx[d] = (x[1] - x[d-2]) * x[d-1] - x[d] + F
        # The general case.
        for n = 3:(d-1)
            @inbounds dx[n] = (x[n+1] - x[n-2]) * x[n-1] - x[n] + F
        end

        nothing
    end

    # TODO: Need to test this construction.
    # Some magic using the diagm function from the LinearAlgebra library.
    ∇u =
        (x, _) -> diagm(
            -d => [x[d-1]],
            -2 => circshift(x, -1)[1:(d-1)],
            -1 => circshift(x, -1)[2:d] - circshift(x[2:d], 1),
            0 => -ones(d),
            1 => circshift(x, 1)[1:(d-1)],
            d - 1 => -[x[d], x[1]],
            d => [x[2] - x[d-1]],
        )

    # Time and space parameters
    x₀ = zeros(d)
    x₀[1] += 0.01
    t₀ = 0.0
    T = 1.0

    return Model("lorenz", d, lorenz!, ∇u, x₀, t₀, T)

end


function validate_all(N::Int64; reload_data::Bool=false, nosave::Bool=true)
    Random.seed!(20220805)
    convergence_validation(ex_rossby(), N, reload_data=reload_data, nosave=nosave)
    # convergence_validation(ex_lorenz()..., N)
end
