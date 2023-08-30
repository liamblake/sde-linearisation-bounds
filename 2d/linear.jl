using LinearAlgebra
using Random

include("validation.jl")

function main(; regenerate = true)
    Random.seed!(1308234)

    A = [1.5 -0.7; 1.1 -2.1]
    function u!(s, x, _)
        mul!(s, A, x)
    end

    function ∇u!(s, x, t)
        s .= A
    end

    B = [1.0 0.5; 0.7 0.9]
    BBT = B * B'
    function σ!(s, _, _)
        s .= B
    end

    function σσᵀ!(s, _, _)
        s .= BBT
    end

    # Initial condition and final time
    x₀ = [0.0, 1.0]
    T = 1.0

    # Noise scale parameter
    εs = [10^(-i) for i in 1.0:0.25:3.0]

    bound_validation_2d(
        "linear2d",
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
        dt = 10^(-3.0),
        regenerate = regenerate,
    )
end

main(; regenerate = true)