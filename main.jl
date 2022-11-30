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
function generate_or_load!(generate::Bool, y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, N, εs, dts, data_fname::String)
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
N = 100

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

# det_prob = ODEProblem(model.velocity!, space_time.x₀, (space_time.t₀, space_time.T))
# det_sol = solve(det_prob, Euler(), dt=minimum(dts))
# w = last(det_sol.u)
# for (i, ε) in enumerate(εs)
#     z_rels[i, :, :] = sign.(y_rels[i, :, :] .- w) .* exp.(log.(abs.(y_rels[i, :, :] .- w)) .- log.(ε))
# end


# Perform the analysis, generating and saving all appropriate plots
println("Calculating and generating plots...")
theorem_validation(y_rels, z_rels, gauss_z_rels, gauss_y_rels, model, space_time, εs, dts, rs, legend_idx=2)


# ################## Many initial conditions ##################
# N = 1000
# ε = 0.05
# dt = 0.0005
# many_sts = Vector{SpaceTime}(undef, 3)
# many_sts[1] = remake(space_time, T=2.5)
# many_sts[2] = remake(many_sts[1], x₀=[1.15, 2.0])
# many_sts[3] = remake(many_sts[1], x₀=[0.4, 1.0])

# function σ!(dW, _, _, _)
#     dW .= 0.0
#     dW[diagind(dW)] .= ε
#     nothing
# end

# # Generate/load data for each one
# GENERATE = false
# many_y = Array{Float64}(undef, 3, model.d, N)

# for (i, st) in enumerate(many_sts)
#     name = "$(model.name)_$(st.x₀)_[$(st.t₀),$(st.T)]_I"
#     data_fname = "data/$(name).jld"

#     if GENERATE
#         sde_realisations!(@view(many_y[i, :, :]), model.velocity!, σ!, N, model.d, model.d, st.x₀, st.t₀, st.T, dt)
#         save(data_fname, "many_y", many_y[i, :, :])
#     else
#         dat = load(data_fname)
#         many_y[i, :, :] = dat["many_y"]
#     end
# end


# # Unlikely to be used
# many_points_plot(many_y, model, many_sts, ε, dt, (0, π), (0, π + 1))


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
Threads.@threads for ϵ in [0.0, 0.3]
    # T = 2.5
    model = ex_rossby(σ_id!, ϵ=ϵ)
    S² = x -> opnorm(Σ_calculation(model, x, t₀, T, dt, dx)[2])
    S²_grid = S².(x₀_grid)'

    # Interpolate the scalar field for visualisation purposes
    p = heatmap(xs, ys, log.(S²_grid), xlabel=L"x_1", ylabel=L"x_2")
    savefig(p, "output/s2_field_$(ϵ).pdf")

    # Extract robust sets and plot
    R = S²_grid .< cutoff
    p = heatmap(xs, ys, R, xlabel=L"x_1", ylabel=L"x_2", c=cgrad([:white, :lightskyblue]), cbar=false)
    savefig(p, "output/s2_robust_$(ϵ).pdf")
end


diffs = Array{Float64}(undef, length(εs), N)
for (i, ε) in enumerate(εs)
    diffs[i, :] = pnorm(y_rels[i, :, :] .- w, dims=1)
end
scatter(log.(εs), log.(mean(diffs, dims=2)), legend=false)
plot!(log.(εs), log.(εs))