using LinearAlgebra
using Random

using DifferentialEquations
using JLD
using LaTeXStrings

include("models.jl")
include("solve_sde.jl")
include("analysis.jl")

Random.seed!(3259245)

# Universal plot options - for consistent publication-sized figures
scale_factor = 2.5
fig_width_cm = 11.0
fig_height_cm = fig_width_cm / 1.4
resolution = scale_factor * 72.0 / 2.54 .* (fig_width_cm, fig_height_cm)
# 1 inch ≡ 72pt ⟺ 1 cm ≡ 72 / 2.54 pt
ptheme = Theme(;
    fonts = (; regular = "CM"),
    fontsize = scale_factor * 15,
    # ,
    Axis = (; width = resolution[1], height = resolution[2]),
)
set_theme!(ptheme)

# Specify the ODE and SDE solvers, as provided by DifferentialEquations.jl
# ODE solvers: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
# SDE solvers: https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/
# Note that the order of each solver must be compatible with the step sizes. See the supplementary
# materials for details.
ode_solver = Euler()
sde_solver = EM()

################## Model specification ##################
## ROSSBY MODEL WITH σ = Iₙ
function σ_id(_, _, _)
    SA[1.0 0.0; 0.0 1.0]
end
model = rossby(σ_id)

################## Data generation ##################
# If true, generate new data (takes some time). If false, attempt to load previously saved data.
GENERATE_DATA = false

# The number of realisations to work with (must match saved if loading data)
N = 10000

################## Theorem validation ##################
# Define the initial condition and finite-time interval
space_time = SpaceTime(SA[0.0, 1.0], 0.0, 1.0)

rs = [1, 2, 3, 4]
# The values of ε to consider
εs = [10^i for i in range(-1; stop = -4.0, step = -0.25)]
# Corresponding step sizes. Chosen automatically using the strong order of the SDE solver, but can
# be set manually if desired.
# println("Order of SDE solver: $(StochasticDiffEq.alg_order(sde_solver))")
# println("Order of ODE solver: $(OrdinaryDiffEq.alg_order(ode_solver))")
# γ = min(StochasticDiffEq.alg_order(sde_solver), OrdinaryDiffEq.alg_order(ode_solver))
# # γ = 2.0
# println("Step size: ε^($(2 / γ))")
dt = minimum(εs)^1

# Naming convention for data and figure outputs.
name = "$(model.name)_$(space_time.x₀)_[$(space_time.t₀),$(space_time.T)]_I"
data_fname = "data/$(name).jld"

# Pre-allocation of output
y_rels = Array{Float64}(undef, length(εs), model.d, N)
z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_z_rels = Array{Float64}(undef, length(εs), model.d, N)

if GENERATE_DATA
    # Solve the SDE to generate new data
    generate_ε_data!(y_rels, z_rels, gauss_z_rels, model, space_time, N, εs, dt)
    save(data_fname, "y", y_rels, "z", z_rels, "gauss_z", gauss_z_rels)

else
    # Reload previously saved data
    dat = load(data_fname)
    y_rels .= dat["y"]
    z_rels .= dat["z"]
    gauss_z_rels .= dat["gauss_z"]

    # Ensure sizes make sense
    @assert size(y_rels) == size(z_rels) == size(gauss_z_rels)
    @assert size(y_rels)[1] == length(εs)

    # Overwrite the number of realisations with whatever is in the file
    Nn = size(y_rels)[3]
    if N != Nn
        println("Warning: changing N to $(N)")
        N = Nn
    end
end

# Perform the analysis, generating and saving all appropriate plots
println("Calculating and generating plots...")
theorem_validation(
    y_rels,
    z_rels,
    gauss_z_rels,
    model,
    space_time,
    εs,
    dt,
    rs,
    ptheme,
    [
        L"(a) $r = 1$ (mean)",
        L"(b) $r = 2$ (covariance)",
        L"(c) $r = 3$ (skewness)",
        L"(d) $r = 4$ (kurtosis)",
    ];
    ode_solver = ode_solver,
    legend_idx = 2,
    hist_idxs = [1, 2, 5, 9],
)

################## Σ through time ##################
sde_solver = EM()
N = 10000
ε = 0.03
x₀ = SA[0.0, 1.0]
dt = ε^2
t₀ = 0.0
ts = 0.1:0.1:1.0
hist_idxs = 2:2:length(ts)

# The full figure
f = Figure()

for (i, (σ, g_label1, g_label2, σ_label)) in enumerate([
    (σ_id, L"(a) $\sigma(x,t) = I_2$", L"(c) $\sigma(x,t) = I_2$", "I"),
    (
        (x, _, t) -> SA[x[1] 0.5; x[1] 0.5 * x[1]+0.5 * x[2]],
        L"(b) $\sigma(x,t) = \sigma_M(x)$",
        L"(d) $\sigma(x,t) = \sigma_M(x)$",
        "M",
    ),
])
    model = rossby(σ)

    # Naming convention for data and figure outputs.
    name = "$(model.name)_$(x₀)_fixed$(ε)_through[$(minimum(ts)),$(maximum(ts))]_$(σ_label)"
    data_fname = "data/$(name).jld"

    # Pre-allocation of output
    time_rels = Array{Float64}(undef, length(ts), model.d, N)

    if GENERATE_DATA
        # Solve the SDE to generate new data
        generate_time_data!(time_rels, model, ε, x₀, t₀, ts, N, dt)
        save(data_fname, "rels", time_rels)
    else
        # Reload previously saved data
        dat = load(data_fname)
        time_rels .= dat["rels"]
    end

    Σ_through_time!(
        f,
        time_rels,
        model,
        ε,
        x₀,
        t₀,
        ts,
        dt,
        g_label1,
        g_label2,
        i;
        hist_idxs = hist_idxs,
        ode_solver = ode_solver,
    )
end

save_figure(f, "through_time.pdf")

################## Stochastic sensitivity calculations ##################
# Create a grid of initial conditions
xs = collect(range(0; stop = π, length = 800))
ys = collect(range(0; stop = π, length = 800))

# Time interval of interest
t₀ = 0.0
T = 1.0
# Temporal and spatial discretisation for stochastic sensitivity calculations
dt = 0.01
dx = 0.001

# Pick a threshold value for extracting coherent sets
threshold = 10.0

S²_grid_sets(model, xs, ys, t₀, T, threshold, dt, dx, "_$(ϵ)"; ode_solver = ode_solver)