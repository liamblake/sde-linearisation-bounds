"""
Run various performance tests, using the BenchmarkTools pacakge.
"""

using Random

using BenchmarkTools
using StaticArrays

include("models.jl")
include("covariance.jl")
include("solve_sde.jl")

Random.seed!(128345)

suite = BenchmarkGroup()

println("nthreads: $(Threads.nthreads())")

# An example SDE model
function σ(_, _, _)
    SA[1.0 0.0; 0.0 1.0]
end
model = ex_rossby(σ)
x₀ = SA[1.0, 0.0]
t₀ = 0.0
T = 1.0

# Calculation of theoretical covariance matrix
suite["Σ_calculation"] = @benchmarkable Σ_calculation(model, x₀, t₀, T, 0.001)

# Solving of SDE
N = 10
dest = Array{Float64}(undef, (2, N))
suite["sde_realisations"] = @benchmarkable sde_realisations!(dest, model.velocity, σ, N, 2, 2, 1.0, x₀, t₀, T, 0.01)


# Upon running/including this file, tune and run the benchmark suite
tune!(suite)
results = run(suite)
