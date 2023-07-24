"""
Information describing an SDE model of the form
    dxₜ = u(xₜ, t)dt + σ(xₜ, t)dWₜ,
stored as
    n: The dimension of the state variable xₜ.
    m: The dimension of the Wiener process Wₜ.
    u!: A function with signature u!(dest, x, t), which evaluates the drift function at state x and
        time t and stores the resulting vector in dest.
    ∇u!: A function with signature ∇u!(dest, x, t), which evaluates the spatial Jacobian of the
         drift function u at state x and time t and evaluates the resulting matrix in dest.
    σ!: A function with signature σ!(dest, x, t), which evaluates the diffusion matrix σ at state x
        and time t and evaluates the resulting matrix in dest.
    σσᵀ!: A function with signature σσᵀ!(dest, x, t), which evaluates the product σσᵀ at state x and
          time t and evaluates the resulting matrix in dest. Specification of this in-place allows
          for more efficient solving of the state-covariance equations.

"""
struct SDEModel
    n::Int
    m::Int
    # In-place version of u
    u!::Function
    # In-place version of ∇u
    ∇u!::Function
    # In-place calculation of σ
    σ!::Function
    # In-place calculation of σσᵀ
    σσᵀ!::Function
end

"""
    fill_Id!(s)

Fill a generic Matrix s as the identity in place, using the diagind function provided by
LinearAlgebra.
"""
function fill_Id!(s::AbstractMatrix)
    s .= 0.0
    s[diagind(s)] .= 1.0
end

"""
	ex_rossby(; A = 1.0, c = 0.5, K = 4.0, l₁ = 2.0, c₁ = π, k₁ = 1.0, ϵ = 0.3)

A kinematic travelling wave in a co-moving frame, with oscillatory determinsitic perturbations.
Taken directly from Chapter 5 of

    R. M. Samelson and S. Wiggins. Lagrangian Transport in Geophysical Jets and Waves: The Dynamical
    Systems Approach, volume 31 of Interdisciplinary Applied Mathematics. Springer, New York, NY,
    2006.

The parameters are
    - A: amplitude of the primary wave.
    - c: phase speed of the primary wave.
    - K: wavenumber of the primary wave in the x₁-direction.
    - k₁: wavenumber of the oscillatory perturbation in the x₁-direction.
    - l₁: wavenumber of the oscillatory perturbation in the x₂-direction.
    - c₁: phase speed of the oscillatory perturbation.
    - ϵ: amplitude of the oscillatory perturbation.

"""
function rossby_Id(; A = 1.0, c = 0.5, K = 4.0, l₁ = 2.0, c₁ = π, k₁ = 1.0, ϵ = 0.3)
    return SDEModel(
        2,
        2,
        function (s, x, t)
            s[1] =
                c - A * sin(K * x[1]) * cos(x[2]) +
                ϵ * l₁ * sin(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
            s[2] =
                A * K * cos(K * x[1]) * sin(x[2]) +
                ϵ * k₁ * cos(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
        end,
        function (s, x, t)
            s[1, 1] =
                -A * K * cos(K * x[1]) * cos(x[2]) +
                ϵ * k₁ * l₁ * cos(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
            s[1, 2] =
                A * sin(K * x[1]) * sin(x[2]) -
                ϵ * l₁^2 * sin(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
            s[2, 1] =
                -A * K^2 * sin(K * x[1]) * sin(x[2]) -
                ϵ * k₁^2 * sin(k₁ * (x[1] - c₁ * t)) * sin(l₁ * x[2])
            s[2, 2] =
                A * K * cos(K * x[1]) * cos(x[2]) +
                ϵ * k₁ * l₁ * cos(k₁ * (x[1] - c₁ * t)) * cos(l₁ * x[2])
        end,
        function (s, _, _)
            fill_Id!(s)
        end,
        function (s, _, _)
            fill_Id!(s)
        end,
    )

    return Model("rossby", 2, rossby, ∇u, σ)
end

"""
    linear_vel(A::StaticMatrix, σ::StaticMatrix)

An Ornstein-Uhlenbeck process
    dyₜ = Ayₜdt + σdWₜ,
where A and σ are constant d×d matrices. The analytical solution is a zero-mean Gaussian process.

"""
function linear_vel(A, σ)
    # Ensure matrix sizes are compatable
    @assert issquare(A)
    @assert size(σ, 1) == size(A, 1)

    return SDEModel(
        size(A, 1),
        size(σ, 1),
        function (s, x, _)
            mul!(s, A, x)
        end,
        function (s, _, _)
            s .= A
        end,
        function (s, _, _)
            s .= σ
        end,
        function (s, _, _)
            mul!(s, σ, σ')
        end,
    )
end
