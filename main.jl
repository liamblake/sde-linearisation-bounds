using Random

using JLD
using Plots

include("models.jl")
include("solve_sde.jl")
include("analysis.jl")

Random.seed!(3259245)
GENERATE_DATA = true

# Specify the model, initial condition and time interval here
model = ex_rossby()
space_time = SpaceTime([0.0, 1.0], 0.0, 1.0)

# The number of realisations to work with (overwritten if loading data)
N = 10000

rs = [1, 2, 3, 4]
εs = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
dt = minimum(εs) / 10.0

# Identiifcation of the model and data
name = "$(model.name)_$(space_time.x₀)_[$(space_time.t₀),$(space_time.T)]"
data_fname = "data/$(name).jld"

# Pre-allocation of output
y_rels = Array{Float64}(undef, length(εs), model.d, N)
z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_z_rels = Array{Float64}(undef, length(εs), model.d, N)
gauss_y_rels = Array{Float64}(undef, length(εs), model.d, N)


if GENERATE_DATA
    # Solve the SDE to generate new data
    generate_data!(y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, N, εs, dt=dt)
    save(data_fname, "y", y_rels, "z", z_rels, "gauss_z", gauss_z_rels, "gauss_y", gauss_y_rels)

else
    # Reload previously saved data
    dat = load(data_fname)
    y_rels = dat["y"]
    z_rels = dat["z"]
    gauss_z_rels = dat["gauss_z"]
    gauss_y_rels = dat["gauss_y"]

    # Ensure sizes make sense
    @assert size(y_rels) == size(z_rels) == size(gauss_z_rels) == size(gauss_y_rels)
    @assert size(y_rels)[1] == length(εs)
    # Overwrite the number of realisations with whatever is in the file
    N = size(y_rels)[3]
    println("Warning: changing N to $(N)")
end

# Perform the analysis, generating and saving all appropriate plots
println("Calculating and generating plots...")
full_analysis(y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, εs, rs, dt)