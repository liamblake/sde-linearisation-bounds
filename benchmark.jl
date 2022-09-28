"""
Run various performance tests, using the BenchmarkTools pacakge.
"""

using Random

using BenchmarkTools

include("main.jl")

Random.seed!(128345)

model = ex_rossby()
function σ!(dW, _, _, _)
    dW[1, 1] = 1.0
    dW[2, 2] = 1.0
    dW[1, 2] = 0.0
    dW[2, 1] = 0.0
    nothing
end

# Calculation of theoretical covariance matrix
# println("Σ_calculation")
# @benchmark Σ_calculation(model, 0.001, 0.001)

# Solving of SDE 
# println("sde_realisations")
N = 10
dest = Array{Float64}(undef, (2, N))
@benchmark sde_realisations(
    dest,
    model.velocity!,
    σ!,
    N,
    2,
    2,
    model.x₀,
    model.t₀,
    model.T,
    0.01,
)
