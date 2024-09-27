using LinearAlgebra
using Statistics

using DifferentialEquations
using JLD2
using ProgressMeter

include("../pyplot_setup.jl")
save_dpi = 600

include("../computations/gaussian_computation.jl")

"""
	pnorm(A; dims=1, p=2)

Calculate the p-norm along the given dimension of a multidimensional array.
"""
function pnorm(A; dims = 1, p = 2)
    f = a -> norm(a, p)
    return mapslices(f, A; dims = dims)
end

"""
    generate_joint_sde_samples!(
        y_dest,
        l_dest,
        N,
        d_y,
        d_W,
        T,
        εs,
        u!,
        σ!,
        ∇u!,
        x₀;
        dt = minimum(εs),
    )

Arguments:
    - y_dest: Array to store the realisations of the original SDE.
    - l_dest: Array to store the realisations of the correpsonding linearised SDE.
"""
function generate_joint_sde_samples!(
    y_dest,
    l_dest,
    N,
    d_y,
    d_W,
    T,
    εs,
    u!,
    σ!,
    ∇u!,
    x₀;
    dt = minimum(εs),
)
    # First, generate an accurate deterministic solution
    function sciml_u!(s, x, _, t)
        u!(s, x, t)
    end
    det_prob = ODEProblem(sciml_u!, x₀, (0.0, T))
    det_sol = solve(det_prob, Euler(); dt = dt, saveat = dt)
    F = det_sol.u[end]

    # Set up joint SDE problem
    # State is
    # [ SDE solution           ]
    # [ Linearised solution    ]
    tmp_vec = zeros(d_y)
    tmp_mat = zeros(d_y, d_y)
    function u_joint!(s, x, _, t)
        # Evaluate terms along determinstic trajectory
        u!(tmp_vec, det_sol(t), t)
        ∇u!(tmp_mat, det_sol(t), t)

        # SDE drift
        u!(@view(s[1:d_y]), x[1:d_y], t)

        # Linearised drift
        u!(@view(s[(d_y + 1):(2 * d_y)]), det_sol(t), t)
        s[(d_y + 1):(2 * d_y)] += tmp_mat * (x[(d_y + 1):(2 * d_y)] - det_sol(t))
    end

    function σ_joint!(s, x, ε, t)
        s .= 0.0

        # SDE diffusion
        σ!(@view(s[1:d_y, :]), x[1:d_y], t)

        # Linearised diffusion
        σ!(@view(s[(d_y + 1):(2 * d_y), :]), det_sol(t), t)
        # println(s)
        # println(det_sol(t))
        # i > 1 && sqrt(-1)
        # i += 1
        rmul!(s, ε)
    end

    sde_prob = SDEProblem(
        u_joint!,
        σ_joint!,
        repeat(x₀, 2),
        (0.0, T),
        0.0;
        noise_rate_prototype = zeros(2 * d_y, d_W),
    )

    # Generate samples
    p = Progress(length(εs); desc = "Generating SDE samples...", showspeed = true)
    for (i, ε) in enumerate(εs)
        sde_prob = remake(sde_prob; p = ε)
        ens = EnsembleProblem(sde_prob)
        sol = solve(
            ens,
            EM(),
            EnsembleSerial();
            # EnsembleThreads();
            trajectories = N,
            save_everystep = false,
            saveat = [T],
            dt = dt,
        )

        rels = Array(sol)
        y_dest[:, :, i] = rels[1:(d_y), 1, :]
        l_dest[:, :, i] = rels[(d_y + 1):(2 * d_y), 1, :]
        next!(p)
    end
    finish!(p)

    return F
end

function bound_validation_2d(
    name,
    x₀,
    T,
    u!,
    ∇u!,
    σ!,
    σσᵀ!,
    d_y,
    d_W,
    εs,
    N;
    dt = minimum(εs),
    regenerate = false,
)
    # Pre-allocate storage of generated values
    y_rels = Array{Float64}(undef, d_y, N, length(εs))
    l_rels = Array{Float64}(undef, d_y, N, length(εs))
    F = Vector{Float64}(undef, d_y)

    if regenerate
        F .= generate_joint_sde_samples!(
            y_rels,
            l_rels,
            N,
            d_y,
            d_W,
            T,
            εs,
            u!,
            σ!,
            ∇u!,
            x₀;
            dt = dt,
        )

        # Save the realisatons
        save("data/$(name)_rels.jld2", "y_rels", y_rels, "l_rels", l_rels, "F", F)

    else
        # Reload previously saved data
        dat = load("data/$(name)_rels.jld2")
        y_rels .= dat["y_rels"]
        l_rels .= dat["l_rels"]
        F .= dat["F"]
    end

    # Compute the linearised mean and covariance. This can be computed independently of ε.
    ts = 0.0:dt:T
    ws = Vector{Vector{Float64}}(undef, length(ts))
    Σs = Vector{Matrix{Float64}}(undef, length(ts))
    gaussian_computation!(ws, Σs, d_y, u!, ∇u!, σσᵀ!, x₀, zeros(d_y, d_y), ts)

    Σ = Σs[end]

    # Plot histograms for each value of ε
    for (i, ε) in enumerate(εs)
        fig = figure()

        ax_hist = fig.add_subplot(1, 1, 1)
        xlabel(L"y_1")
        ylabel(L"y_2")

        # Limiting covariance
        vals, evecs = eigen(Σ)
        θ = atand(evecs[2, 2], evecs[1, 2])
        ell_w = sqrt(vals[2])
        ell_h = sqrt(vals[1])
        ax_hist.scatter(F[1], F[2]; s = 2.5, c = :black, zorder = 2)
        for l in [1, 2, 3]
            ax_hist.add_artist(
                PyPlot.matplotlib.patches.Ellipse(;
                    xy = F,
                    width = ε * 2 * l * ell_w,
                    height = ε * 2 * l * ell_h,
                    angle = θ,
                    edgecolor = :black,
                    facecolor = :none,
                    linewidth = 1.0,
                    zorder = 2,
                    label = if l == 1
                        "Theory"
                    else
                        ""
                    end,
                ),
            )
        end

        # Sample covariance
        S_m = mean(y_rels[:, :, i]; dims = 2)
        S_y = cov(y_rels[:, :, i]')

        vals, evecs = eigen(S_y)
        θ = atand(evecs[2, 2], evecs[1, 2])
        ell_w = sqrt(vals[2])
        ell_h = sqrt(vals[1])
        ax_hist.scatter(S_m[1], S_m[2]; s = 2.5, c = :red, zorder = 2)
        for l in [1, 2, 3]
            ax_hist.add_artist(
                PyPlot.matplotlib.patches.Ellipse(;
                    xy = S_m,
                    width = 2 * l * ell_w,
                    height = 2 * l * ell_h,
                    angle = θ,
                    edgecolor = :red,
                    linestyle = :dashed,
                    facecolor = :none,
                    linewidth = 1.0,
                    zorder = 2,
                    label = if l == 1
                        "Sample"
                    else
                        ""
                    end,
                ),
            )
        end

        _, xedges, yedges, hm = ax_hist.hist2d(
            y_rels[1, :, i],
            y_rels[2, :, i];
            bins = 75,
            rasterized = true,
            density = true,
        )

        # Colorbar
        colorbar(hm; ax = ax_hist, location = "top", aspect = 40)

        fig.savefig("output/$name/hist_$ε.pdf"; bbox_inches = "tight", dpi = save_dpi)
        close(fig)
    end

    # Estimate strong errors
    # Compute first 4 moments of strong error
    rs = 1.0:4.0
    str_errs = Array{Float64}(undef, length(rs), length(εs))
    # l_rels = y_rels + permutedims(repeat(εs .^ 2; inner = (1, N, d_y)), (3, 2, 1))
    for (i, r) in enumerate(rs)
        str_errs[i, :] = mean(pnorm(y_rels - l_rels; dims = 1) .^ r; dims = 2)

        # ε plots
        fig, ax = subplots()
        ax.set_xlabel(L"\varepsilon")
        ax.set_ylabel(L"E_{%$(Int64(r))}\!\left(\varepsilon\right)")

        # Fit a LoBF of the form log(Γ) = a + b log(ε)
        # This is equivalent to Γ = ε^b exp(a)
        X = hcat(ones(length(εs)), log10.(εs))

        coefs = inv(X' * X) * X' * log10.(str_errs[i, :])

        ax.plot(εs, @.(10.0^(coefs[1]) * εs^coefs[2]), "r-")
        ax.scatter(εs, str_errs[i, :]; s = 2.5, c = :black, zorder = 2)

        # Annotate with the slope
        ax.text(
            minimum(εs),
            maximum(str_errs[i, :]),
            L"E_{%$(Int64(r))}\!\left(\varepsilon\right) \propto \varepsilon^{%$(round(coefs[2]; digits=2))}";
            color = :red,
            verticalalignment = "top",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        fig.savefig("output/$name/str_err_r_$(r).pdf"; bbox_inches = "tight", dpi = save_dpi)
        close(fig)
    end
end