using Random

include("validation.jl")

function main(; regenerate = true)
    Random.seed!(1308234)

    # Define the Rossby model
    # Parameters
    A = 1.0
    c = 0.5
    K = 4.0
    l₁ = 2.0
    c₁ = π
    k₁ = 1.0
    ϵ = 0.3

    function u!(s, x, t)
        s[1] =
            c - A * sin(K * x[1]) * cos(x[2]) + ϵ * l₁ * sin(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
        s[2] =
            A * K * cos(K * x[1]) * sin(x[2]) + ϵ * k₁ * cos(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
    end

    function ∇u!(s, x, t)
        s[1, 1] =
            -A * K * cos(K * x[1]) * cos(x[2]) +
            ϵ * k₁ * l₁ * cos(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
        s[1, 2] =
            A * sin(K * x[1]) * sin(x[2]) - ϵ * l₁^2 * sin(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
        s[2, 1] =
            -A * K^2 * sin(K * x[1]) * sin(x[2]) -
            ϵ * k₁^2 * sin(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
        s[2, 2] =
            A * K * cos(K * x[1]) * cos(x[2]) +
            ϵ * k₁ * l₁ * cos(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
    end

    function σ!(s, x, _)
        s[1, 1] = 1.0
        s[1, 2] = sin(K * x[1]) * cos(x[2])
        s[2, 1] = 0.0
        s[2, 2] = K * cos(K * x[1]) * sin(x[2])
    end

    function σσᵀ!(s, x, _)
        s[1, 1] = 1 + sin(K * x[1])^2 * cos(x[2])^2
        s[1, 2] = K * sin(K * x[1]) * cos(x[2]) * cos(K * x[1]) * sin(x[2])
        s[2, 1] = s[1, 2]
        s[2, 2] = K^2 * cos(K * x[1])^2 * sin(x[2])^2
    end

    # Initial condition and final time
    x₀ = [0.0, 1.0]
    T = 1.0

    # Noise scale parameter
    εs = [10^(-i) for i in 1.0:0.1:3.0]

    bound_validation_2d(
        "rossby",
        x₀,
        T,
        u!,
        ∇u!,
        σ!,
        σσᵀ!,
        2,
        2,
        εs,
        10000;
        dt = 10^(-4.0),
        regenerate = regenerate,
    )
end

main(; regenerate = true)