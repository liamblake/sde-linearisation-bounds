"""
Run various performance tests, using the BenchmarkTools pacakge.
"""

using BenchmarkTools

include("main.jl")

function run_benchmarks()

    let model = ex_rossby()

        # Stochastic sensitivity calculation
        @benchmark Σ_calculation($model, 0.001, 0.001)

    end
end

