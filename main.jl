using LinearAlgebra
using Random

using JLD
using LaTeXStrings
using Plots

include("models.jl")
include("solve_sde.jl")
include("analysis.jl")

Random.seed!(3259245)

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
# Euler-Maruyama solver.
εs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
dts = [0.001, 0.0005, 0.0001, 0.00005, 1e-6, 1e-6, 1e-6]

# Naming convention for data and figure outputs.
name = "$(model.name)_$(space_time.x₀)_[$(space_time.t₀),$(space_time.T)]_I"
data_fname = "data/$(name).jld"

# Pre-allocation of output
y_rels = Array{Float64}(undef, length(εs), model.d, N)
z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_y_rels = Array{Float64}(undef, length(εs), model.d, N)

GENERATE = true
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

# Perform the analysis, generating and saving all appropriate plots
println("Calculating and generating plots...")
theorem_validation(y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, εs, dts, rs, legend_idx = 2)



################## Stochastic sensitivity calculations ##################
# Create a grid of initial conditions
xs = 0:0.01:π
ys = 0:0.01:π
x₀_grid = reshape([collect(pairs) for pairs in Base.product(xs, ys)][:], length(xs), length(ys))

# Time interval of interest
t₀ = 0.0
T = 2.5
# Temporal and spatial discretisation for stochastic sensitivity calculations
dt = 0.01
dx = 0.001

# Pick a threshold value for extracting coherent sets
threshold = 70.0

# Varying the amplitude of the oscillatory perturbation
for ϵ in [0.0, 0.3]
    model = ex_rossby(σ_id, ϵ = ϵ)

    # Compute the S² value for each initial condition, as the operator norm of Σ
    S² = x -> opnorm(Σ_calculation(model, x, t₀, T, dt)[2])
    S²_grid = S².(x₀_grid)'

    # Interpolate the scalar field for visualisation purposes
    p = heatmap(xs, ys, log.(S²_grid), xlabel = L"x_1", ylabel = L"x_2")
    savefig(p, "output/s2_field_$(ϵ).pdf")

    # Extract robust sets and plot
    R = S²_grid .< threshold
    p = heatmap(xs, ys, R, xlabel = L"x_1", ylabel = L"x_2", c = cgrad([:white, :lightskyblue]), cbar = false)
    savefig(p, "output/s2_robust_$(ϵ).pdf")

end
