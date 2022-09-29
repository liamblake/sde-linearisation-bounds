"""
Run various performance tests, using the BenchmarkTools pacakge.
"""

using Random

using BenchmarkTools
using StaticArrays

include("main.jl")

Random.seed!(128345)

model = ex_rossby()
function σ(_, _, _)
	dW = SA[1.0 0.0; 0.0 1.0]
	return dW
end

# Calculation of theoretical covariance matrix
# println("Σ_calculation")
# @benchmark Σ_calculation(model, 0.001, 0.001)

# Solving of SDE 
# println("sde_realisations")
N = 10
x₀ = [1.0, 0.0]
t₀ = 0.0
T = 1.0
dest = Array{Float64}(undef, (2, N))
@benchmark sde_realisations(
    dest,
    model.velocity!,
    σ,
    N,
    2,
    2,
    x₀,
    t₀,
    T,
    0.01,
)
