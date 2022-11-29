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

# An example SDE model
model = ex_rossby()
x₀ = SA[1.0, 0.0]
t₀ = 0.0
T = 1.0
function σ!(_, _, _)
    SA[1.0 0.0; 0.0 1.0]
end

# Calculation of theoretical covariance matrix
println("nthreads: $(Threads.nthreads())")
suite["Σ_calculation"] = @benchmarkable Σ_calculation(model, x₀, t₀, T, 0.001)

# Solving of SDE
N = 10
dest = Array{Float64}(undef, (2, N))
suite["sde_realisations"] = @benchmarkable sde_realisations!(dest, model.velocity!, σ!, N, 2, 2, x₀, t₀, T, 0.01)


# Upon running/including this file, tune and run the benchmark suite
tune!(suite)
results = run(suite)
