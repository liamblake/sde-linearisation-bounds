using LinearAlgebra
using Random

using DifferentialEquations
using JLD
using LaTeXStrings
using Plots

include("models.jl")
include("solve_sde.jl")
include("analysis.jl")

Random.seed!(3259245)

# Universal plot options - for consistent publication-sized figures
# Default axis font size is 11 in Plots. Set the scale factor here to get the desired font size.
# Other text objects are scaled automatically.
Plots.scalefontsizes()
Plots.scalefontsizes(15.0 / 11.0)
# Specify the figure height and width in pixels
fig_height = 600
fig_width = fig_height / 1.4
# Include any other plot attributes here
plot_attrs = Dict(
    :size => (fig_height, fig_width),
    :markercolor => :black,
    # Margin padding when exporting figures
    :left_margin => 2Plots.mm,
    :top_margin => 2Plots.mm,
    :bottom_margin => 2Plots.mm,
    :top_margin => 2Plots.mm,
    # Colorbar options
    :colorbar => :top,
)

# Specify the ODE and SDE solvers, as provided by DifferentialEquations.jl
# ODE solvers: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
# SDE solvers: https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/
# Note that the order of each solver must be compatible with the step sizes. See the supplementary
# materials for details.
ode_solver = RK4()
sde_solver = SRA3()

# If true, generate new data (takes some time). If false, attempt to load previously saved data.
GENERATE_DATA = true

################## Theorem validation ##################
## ROSSBY MODEL WITH σ = Iₙ
function σ_id(_, _, _)
    SA[1.0 0.0; 0.0 1.0]
end
model = ex_rossby(σ_id)
# Define the initial condition and finite-time interval
space_time = SpaceTime(SA[0.0, 1.0], 0.0, 1.0)

# The number of realisations to work with (overwritten if loading data)
N = 1000

rs = [1, 2, 3, 4]
# The values of ε to consider
εs = [10^i for i in range(-1; stop = -4.0, step = -0.25)]
# Corresponding step sizes. Chosen automatically using the strong order of the SDE solver, but can
# be set manually if desired.
println("Order of SDE solver: $(StochasticDiffEq.alg_order(sde_solver))")
println("Order of ODE solver: $(OrdinaryDiffEq.alg_order(ode_solver))")
γ = min(StochasticDiffEq.alg_order(sde_solver), OrdinaryDiffEq.alg_order(ode_solver))
γ = 1.5
println("Step size: ε^($(2 / γ))")
dts = [ε^(2 / γ) for ε in εs]

# Check that the step size is small enough so that the numerical error does not dominate the
# theoretical.
# This is used to inform the range of ε values and the choice of γ above.
er = range(minimum(εs); stop = maximum(εs), step = 0.001)
dtr = er .^ (2 / γ)
plot(
    log10.(er),
    log10.(dtr .^ (1.5) + #StochasticDiffEq.alg_order(sde_solver)) +
           dtr .^ (OrdinaryDiffEq.alg_order(ode_solver)));
    label = "numerical",
    xlabel = L"\log(\varepsilon)",
    ylabel = L"Err",
)
plot!(log10.(er), log10.(er .^ 2); label = "theory")
plot!(
    log10.(er),
    log10.(er .^ 2 +
           dtr .^ (1.5) + #(StochasticDiffEq.alg_order(sde_solver)) +
           dtr .^ (OrdinaryDiffEq.alg_order(ode_solver)));
    label = "total",
)

# Naming convention for data and figure outputs.
name = "$(model.name)_$(space_time.x₀)_[$(space_time.t₀),$(space_time.T)]_I"
data_fname = "data/$(name).jld"

# Pre-allocation of output
y_rels = Array{Float64}(undef, length(εs), model.d, N)
z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_y_rels = Array{Float64}(undef, length(εs), model.d, N)

if GENERATE_DATA
    # Solve the SDE to generate new data
    generate_data!(y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, N, εs, dts)
    save(data_fname, "y", y_rels, "z", z_rels, "gauss_z", gauss_z_rels, "gauss_y", gauss_y_rels)

else
    # Reload previously saved data
    dat = load(data_fname)
    y_rels .= dat["y"]
    z_rels .= dat["z"]
    gauss_z_rels .= dat["gauss_z"]
    gauss_y_rels .= dat["gauss_y"]

    # Ensure sizes make sense
    @assert size(y_rels) == size(z_rels) == size(gauss_z_rels) == size(gauss_y_rels)
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
    # gauss_y_rels,
    model,
    space_time,
    εs,
    dts,
    rs;
    ode_solver = ode_solver,
    legend_idx = 5,
    plot_attrs = plot_attrs,
)

################## Stochastic sensitivity calculations ##################
include("analysis.jl")
# Create a grid of initial conditions
xs = collect(range(0; stop = π, length = 100))
ys = collect(range(0; stop = π, length = 100))
x₀_grid = reshape([collect(pairs) for pairs in Base.product(xs, ys)][:], length(xs), length(ys))

# Time interval of interest
t₀ = 0.0
T = 1.0
# Temporal and spatial discretisation for stochastic sensitivity calculations
dt = 0.01
dx = 0.001

# Pick a threshold value for extracting coherent sets
threshold = 10.0

ϵ = 0.3
model = ex_rossby(σ_id; ϵ = ϵ)

S²_grid_sets(
    model,
    x₀_grid,
    t₀,
    T,
    threshold,
    dt,
    dx,
    "_$(ϵ)";
    ode_solver = ode_solver,
    xlabel = L"x_1",
    ylabel = L"x_2",
    plot_attrs...,
)
