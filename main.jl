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
plot_attrs = Dict(:size => (fig_height, fig_width))

# Specify the ODE and SDE solvers, as provided by DifferentialEquations.jl
# ODE solvers: https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/
# SDE solvers: https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/
# Note that the order of each solver must be compatible with the step sizes. See the supplementary
# materials for details.
ode_solver = Euler()
sde_solver = EM()

################## Theorem validation ##################
## ROSSBY MODEL WITH σ = Iₙ
function σ_id(_, _, _)
    SA[1.0 0.0; 0.0 1.0]
end
model = ex_rossby(σ_id)
# Define the initial condition and finite-time interval
space_time = SpaceTime(SA[0.0, 1.0], 0.0, 1.0)

# The number of realisations to work with (overwritten if loading data)
N = 10000

rs = [1, 2, 3, 4]
# The values of ε to generate realisations for, and the corresponding step sizes to use in the
# SDE solver.
εs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
dts = εs
# dts = [ε^2 for ε in εs]

# Naming convention for data and figure outputs.
name = "$(model.name)_$(space_time.x₀)_[$(space_time.t₀),$(space_time.T)]_I"
data_fname = "data/$(name).jld"

# Pre-allocation of output
y_rels = Array{Float64}(undef, length(εs), model.d, N)
z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_y_rels = Array{Float64}(undef, length(εs), model.d, N)

# If true, generate new data (takes some time). If false, attempt to load previously saved data.
GENERATE = false
if GENERATE
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

# det_prob = ODEProblem(model.velocity, space_time.x₀, (space_time.t₀, space_time.T))
# det_sol = solve(det_prob, Euler(); dt = minimum(dts))
# w = last(det_sol.u)

# for (i, ε) in enumerate(εs)
#     z_rels[i, :, :] = sign.(y_rels[i, :, :] .- w) .* exp.(log.(abs.(y_rels[i, :, :] .- w)) .- log(ε))
# end

# Perform the analysis, generating and saving all appropriate plots
println("Calculating and generating plots...")
theorem_validation(
    y_rels,
    z_rels,
    gauss_z_rels,
    gauss_y_rels,
    model,
    space_time,
    εs,
    dts,
    rs;
    ode_solver = ode_solver,
    legend_idx = 2,
    plot_attrs = plot_attrs,
)

################## Stochastic sensitivity calculations ##################
# Create a grid of initial conditions
xs = collect(range(0; stop = π, length = 1000))
ys = collect(range(0; stop = π, length = 1000))
x₀_grid = reshape([collect(pairs) for pairs in Base.product(xs, ys)][:], length(xs), length(ys))

# Time interval of interest
t₀ = 0.0
T = 1.0
# Temporal and spatial discretisation for stochastic sensitivity calculations
dt = 0.01
dx = 0.001

# Pick a threshold value for extracting coherent sets
threshold = 10.0

# Varying the amplitude of the oscillatory perturbation
# for ϵ in [0.3]#[0.0, 0.3]
ϵ = 0.3
model = ex_rossby(σ_id; ϵ = ϵ)

# Compute the S² value for each initial condition, as the operator norm of Σ
Σ_eigs = map(x₀_grid) do x
    eigen(Σ_calculation(model, x, t₀, T, dt, dx, ode_solver)[2]).values
end
Sᵐ_grid = map(x -> x[1], Σ_eigs)'
S²_grid = map(x -> x[2], Σ_eigs)'

p = heatmap(xs, ys, log.(S²_grid); xlabel = L"x_1", ylabel = L"x_2", plot_attrs...)
savefig(p, "output/s2_field_$(ϵ).pdf")

# p = heatmap(xs, ys, log.(Sᵐ_grid); xlabel = L"x_1", ylabel = L"x_2", plot_attrs...)

# Extract robust sets and plot
R = S²_grid .< threshold
p = heatmap(
    xs,
    ys,
    R;
    xlabel = L"x_1",
    ylabel = L"x_2",
    c = cgrad([:white, :lightskyblue]),
    cbar = false,
    plot_attrs...,
)
savefig(p, "output/s2_robust_$(ϵ).pdf")

# end
