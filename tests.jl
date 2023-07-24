using Test

include("gaussian_computation.jl")

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

@testset "OU calculations" begin
    """
    Tests the calculation of Σ with an OU process, for which Σ can be computed exactly. See the
    supplementary materials for more details.
    """
    # Define OU process
    A = [1.0 0.0; 0.0 2.3]
    σ = [1.0 0.5; -0.2 1.4]
    model = linear_vel(A, σ)

    x = [1.1, -0.2]
    t = 1.5

    # Exact forms of terms involved
    Fe = [exp(A[1, 1] * t) * x[1]; exp(A[2, 2] * t) * x[2]]
    ∇Fe = [exp(A[1, 1] * t) 0.0; 0.0 exp(A[2, 2] * t)]
    Σe = [
        (σ[1, 1]^2 + σ[1, 2]^2) / (2 * A[1, 1])*(exp(2 * A[1, 1] * t) - 1) (σ[1, 1] * σ[2, 1] + σ[1, 2] * σ[2, 2]) / (A[1, 1] + A[2, 2])*(exp((A[1, 1] + A[2, 2]) * t) - 1)
        (σ[1, 1] * σ[2, 1] + σ[1, 2] * σ[2, 2]) / (A[1, 1] + A[2, 2])*(exp((A[1, 1] + A[2, 2]) * t) - 1) (σ[2, 1]^2 + σ[2, 2]^2) / (2 * A[2, 2])*(exp(2 * A[2, 2] * t) - 1)
    ]

    ts = 0:0.001:t
    ws = Vector{Vector{Float64}}(undef, length(ts))
    Σs = Vector{Matrix{Float64}}(undef, length(ts))

    for Σ₀ in [zeros(2, 2), [1.2 0.5; 0.5 2.1]]
        # Test full w, Σ calculation
        gaussian_computation!(ws, Σs, model, x, Σ₀, ts)

        @test ws[1] == x
        @test Σs[1] == Σ₀
        @test isapprox(ws[end], Fe, atol = 1e-2)
        @test isapprox(Σs[end], ∇Fe * Σ₀ * ∇Fe' + Σe, atol = 1e-2)
    end
end
