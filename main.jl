using LinearAlgebra
using Random

using JLD
using LaTeXStrings
using Plots

include("models.jl")
include("solve_sde.jl")
include("analysis.jl")

Random.seed!(3259245)

"""

"""
function generate_or_load!(generate, y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, N, εs, dts, data_fname)
    if generate
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
    end
end


## ROSSBY MODEL WITH σ = Iₙ
function σ_id!(dW, _, _, _)
    dW .= 0.0
    dW[diagind(dW)] .= 1.0
    nothing
end
model = ex_rossby(σ_id!)
space_time = SpaceTime(SA[0.0, 1.0], 0.0, 1.0)

# The number of realisations to work with (overwritten if loading data)
N = 10000

rs = [1, 2, 3, 4]
εs = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
dts = [0.001, 0.0005, 0.0001, 0.00005, 1e-6, 1e-6, 1e-6]

# Identiifcation of the model and data
name = "$(model.name)_$(space_time.x₀)_[$(space_time.t₀),$(space_time.T)]_I"
data_fname = "data/$(name).jld"

# Pre-allocation of output
y_rels = Array{Float64}(undef, length(εs), model.d, N)
z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_y_rels = Array{Float64}(undef, length(εs), model.d, N)

GENERATE = true
generate_or_load!(GENERATE, y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, N, εs, dts, data_fname)
# Overwrite the number of realisations with whatever is in the file
Nn = size(y_rels)[3]
if N != Nn
    println("Warning: changing N to $(N)")
    N = Nn
end

# Perform the analysis, generating and saving all appropriate plots
println("Calculating and generating plots...")
theorem_validation(y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, εs, dts, rs, legend_idx = 2)



################## Stochastic sensitivity calculations ##################
# Create a grid of initial conditions
xs = 0:0.01:π
ys = 0:0.01:π
x₀_grid = reshape([collect(pairs) for pairs in Base.product(xs, ys)][:], length(xs), length(ys))

t₀ = 0.0
T = 2.5
dt = 0.01
dx = 0.001

# Pick a cut-off value
cutoff = 70.0

# Repeat for several T values
for ϵ in [0.0, 0.3]
    # T = 2.5
    model = ex_rossby(σ_id!, ϵ = ϵ)
    S² = x -> opnorm(Σ_calculation(model, x, t₀, T, dt)[2])
    hls

    S²_grid = S².(x₀_grid)'

    # Interpolate the scalar field for visualisation purposes
    p = heatmap(xs, ys, log.(S²_grid), xlabel = L"x_1", ylabel = L"x_2")
    savefig(p, "output/s2_field_$(ϵ).pdf")

    # Extract robust sets and plot
    R = S²_grid .< cutoff
    p = heatmap(xs, ys, R, xlabel = L"x_1", ylabel = L"x_2", c = cgrad([:white, :lightskyblue]), cbar = false)
    savefig(p, "output/s2_robust_$(ϵ).pdf")
end
