"""
Run various performance tests, using the BenchmarkTools pacakge.
"""

using BenchmarkTools

include("main.jl")

model = ex_rossby()
function σ!(dW, _, _, _)
    dW[1, 1] = 1.0
    dW[2, 2] = 1.0
    dW[1, 2] = 0.0
    dW[2, 1] = 0.0
    nothing
end

# Calculation of theoretical covariance matrix
println("Σ_calculation")
@benchmark Σ_calculation(model, 0.001, 0.001)

# Solving of SDE 
# println("sde_realisations")
# @benchmark sde_realisations(model.velocity!, σ!, 10, (2, 2), model.x₀, model.t₀, model.T, 0.01)
