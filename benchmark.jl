"""
Run various performance tests, using the BenchmarkTools pacakge.
"""

using Random

using BenchmarkTools

include("main.jl")

Random.seed!(128345)

suite = BenchmarkGroup()

# An example SDE model
model = ex_rossby()
x₀ = [1.0, 0.0]
t₀ = 0.0
T = 1.0
function σ!(dW, _, _, _)
    dW[1, 1] = 1.0
    dW[2, 2] = 1.0
    dW[1, 2] = 0.0
    dW[2, 1] = 0.0
    nothing
end

# Calculation of theoretical covariance matrix
suite["Σ_calculation"] = @benchmarkable Σ_calculation(model, x₀, t₀, T, 0.001)

# Solving of SDE 
N = 10
dest = Array{Float64}(undef, (2, N))
suite["sde_realisations"] =
    @benchmarkable sde_realisations(dest, model.velocity!, σ!, N, 2, 2, x₀, t₀, T, 0.01)


# The full validation procedure (with no loading or saving)
# suite["convergence_validation"] = @benchmarkable convergence_validation(
#     model,
#     [x₀],
#     t₀,
#     T,
#     N,
#     quiet = true,
#     save_plots = false,
#     attempt_reload = false,
#     save_on_generation = false,
# )


# Upon running/including this file, tune and run the benchmark suite
tune!(suite)
results = run(suite)
