using Test

using DifferentialEquations
using StaticArrays

include("covariance.jl")
include("models.jl")

@testset "star grid" begin
    x = [1.0, 2.0, 3.0]
    δx = 0.01

    sg = star_grid(x, δx)

    @test sg == [1.01 2.0 3.0; 0.99 2.0 3.0; 1.0 2.01 3.0; 1.0 1.99 3.0; 1.0 2.0 3.01; 1.0 2.0 2.99]
end

@testset "gradient approximation" begin
    δx = 1.0
    star = [[2.0 0.0]; [0.8 0.2]; [1.6 0.6]; [1.4 -0.2]]

    expected = [[0.6 0.1]; [-0.1 0.4]]

    @test isapprox(∇F(star, 2, δx), expected, atol = 1e-10)
end

"""
Tests the calculation of Σ with an OU process, for which Σ can be computed exactly. See the
supplementary materials for more details.
"""
@testset "OU calculations" begin
    # Define OU process
    A = SA[1.0 0.0; 0.0 2.3]
    σ = SA[1.0 0.5; -0.2 1.4]
    model = linear_vel(A, σ)

    x = SA[1.1, -0.2]
    t = 1.5
    ode_solver = Euler()

    # Exact forms of terms involved
    Fe = SA[exp(A[1, 1] * t) * x[1]; exp(A[2, 2] * t) * x[2]]
    ∇Fe = [exp(A[1, 1] * t) 0.0; 0.0 exp(A[2, 2] * t)]
    Σe = [
        (σ[1, 1]^2 + σ[1, 2]^2) / (2 * A[1, 1])*(exp(2 * A[1, 1] * t) - 1) (σ[1, 1] * σ[2, 1] + σ[1, 2] * σ[2, 2]) / (A[1, 1] + A[2, 2])*(exp((A[1, 1] + A[2, 2]) * t) - 1)
        (σ[1, 1] * σ[2, 1] + σ[1, 2] * σ[2, 2]) / (A[1, 1] + A[2, 2])*(exp((A[1, 1] + A[2, 2]) * t) - 1) (σ[2, 1]^2 + σ[2, 2]^2) / (2 * A[2, 2])*(exp(2 * A[2, 2] * t) - 1)
    ]

    # Test full w, Σ calculation
    w, Σ = Σ_calculation(model, x, 0, t, 0.001, 0.001, ode_solver)

    @test isapprox(w, Fe, atol = 1e-0)
    @test isapprox(Σ, Σe, atol = 1e-0)
end