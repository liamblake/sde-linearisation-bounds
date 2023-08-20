using Distributions

include("validation.jl")

function main(; regenerate = true)
    Random.seed!(3259245)

    # The initial condition - a Gaussian centered around 2.0 with variance scaling by δ
    x₀ = 2.0
    init_dist = δ -> Normal(x₀, δ)

    # Time span
    T = 1.0

    # Model definition
    function u(x, _)
        return 0.5 * x
    end

    function ∇u(_, _)
        return 0.5
    end

    function σ(x, _)
        return cos(x)
    end

    # Flow map and linearisation details
    function F!(s, t)
        s[1] = exp(t / 2.0) * x₀
    end

    function ∇F(t)
        exp(t / 2.0)
    end

    bound_validation_1d(
        "multiplicative",
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
        [1, 6, 10, 26];
        linear = true,
        multiplicative = true,
        regenerate = regenerate,
        msize = 12.5,
        lwidth = 1.0,
    )
end

main(; regenerate = false)