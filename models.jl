using Parameters

@with_kw struct Model
    name::String
    d::UInt8
    velocity!::Function
    ∇u::Function
    σ!::Function
end

@with_kw struct SpaceTime
    x₀::AbstractVector
    t₀::Float64
    T::Float64
end

"""
	ex_rossby()

Two dimensional example: perturbed Rossby wave.
"""
function ex_rossby(σ!::Function; A=1.0, c=0.5, K=4.0, l₁=2.0, c₁=π, k₁=1.0, ϵ=0.3)::Model
    # The velocity field, with an in-place update.
    # Much faster this way:
    # https://diffeq.sciml.ai/stable/tutorials/faster_ode_example/#Example-Accelerating-a-Non-Stiff-Equation:-The-Lorenz-Equation
    function rossby!(dx, x, _, t)
        dx[1] =
            c - A * sin(K * x[1]) * cos(x[2]) +
            ϵ * l₁ * sin(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
        dx[2] =
            A * K * cos(K * x[1]) * sin(x[2]) +
            ϵ * k₁ * cos(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
        nothing
    end


    # The velocity gradient matrix is known exactly
    ∇u =
        (x, t) -> [
            -A*K*cos(K * x[1])*cos(x[2])+ϵ*k₁*l₁*cos(k₁ * (x[1] - c₁ * t))*cos(l₁ * x[2]) A*sin(K * x[1])*sin(x[2])-ϵ*l₁^2*sin(k₁ * (x[1] - c₁ * t))*sin(l₁ * x[2])
            -A*K^2*sin(K * x[1])*sin(x[2])-ϵ*k₁^2*sin(k₁ * (x[1] - c₁ * t))*sin(l₁ * x[2]) A*K*cos(K * x[1])*cos(x[2])+ϵ*k₁*l₁*cos(k₁ * (x[1] - c₁ * t))*cos(l₁ * x[2])
        ]

    # Bounds on the velocity and gradient norms
    vel_bound = sqrt(2) * max(abs(c) + abs(A) + abs(ϵ * l₁), abs(A * K) + abs(ϵ * k₁))
    grad_bound =
        2 * max(
            abs(A * K) + abs(ϵ * k₁ * l₁),
            abs(A) + abs(ϵ * l₁^2),
            abs(A * K^2) + abs(ϵ * k₁ * l₁),
            abs(A * K) + abs(ϵ * k₁ * l₁),
        )

    return Model("rossby", 2, rossby!, ∇u, σ!)
end

"""
	ex_lorenz()

40-dimensional example: Lorenz 96 system. Currently unused
"""
function ex_lorenz()::Model
    # Parameters
    d = 4
    F = 8

    # In-place velocity field
    function lorenz!(dx, x, _, _)

        # 3 edge cases explicitly
        @inbounds dx[1] = (x[2] - x[d-1]) * x[d] - x[1] + F
        @inbounds dx[2] = (x[3] - x[d]) * x[1] - x[2] + F
        @inbounds dx[d] = (x[1] - x[d-2]) * x[d-1] - x[d] + F
        # The general case.
        for n = 3:(d-1)
            @inbounds dx[n] = (x[n+1] - x[n-2]) * x[n-1] - x[n] + F
        end

        nothing
    end

    # TODO: Need to test this construction.
    # Some magic using the diagm function from the LinearAlgebra library.
    ∇u =
        (x, _) -> diagm(
            1 - d => [x[d-1]],
            -2 => circshift(x, -1)[1:(d-2)],
            -1 => circshift(x, -1)[2:d] - circshift(x[2:d], 1),
            0 => -ones(d),
            1 => circshift(x, 1)[1:(d-1)],
            d - 2 => -[x[d], x[1]],
            d - 1 => [x[2] - x[d-1]],
        )


    return Model("lorenz", d, lorenz!, ∇u, 1.0)

end
