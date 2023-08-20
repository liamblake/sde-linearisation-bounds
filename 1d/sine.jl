using Distributions

include("validation.jl")

function main(; regenerate = true)
    Random.seed!(3259245)

    # The initial condition - a Gaussian centered around 0.5 with variance scaling by δ
    x₀ = 0.5
    init_dist = δ -> Normal(x₀, δ)

    # Time span
    T = 1.5

    # Model definition
    function u(x, _)
        return sin(x)
    end

    function ∇u(x, _)
        return cos(x)
    end

    function σ(_, _)
        return 1.0
    end

    # Flow map and linearisation details
    tan_μ₀ = tan(x₀ / 2)
    function F!(s, t)
        s[1] = 2 * atan(exp(t) * tan_μ₀)
    end

    function ∇F(t)
        exp(t) * sec(x₀ / 2)^2 / (exp(2 * t) * tan_μ₀^2 + 1)
    end

    bound_validation_1d(
        "sine",
        x₀,
        init_dist,
        T,
        u,
        ∇u,
        σ,
        F!,
        ∇F,
        [10^(-i) for i in 0.0:0.1:2.5],
        [10^(-i) for i in 0.0:0.1:2.5],
        10000,
        10,
        j -> j > 10 && j < 24 && j % 2 == 0,
        [1, 8, 16, 26];
        linear = false,
        multiplicative = false,
        regenerate = regenerate,
        msize = 12.5,
        lwidth = 1.0,
    )
end

main(; regenerate = false)