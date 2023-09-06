using Random

include("validation.jl")

# Define the double gyre
# Parameters
A = 1.0
ω = 2π
ϵ = 5.0

function u!(s, x, t)
    s[1] =
        -π *
        A *
        sin(π * (ϵ * sin(ω * t) * x[1]^2 + (1 - 2 * ϵ * sin(ω * t)) * x[1])) *
        cos(π * x[2])
    s[2] =
        π *
        A *
        cos(π * (ϵ * sin(ω * t) * x[1]^2 + (1 - 2 * ϵ * sin(ω * t)) * x[1])) *
        sin(π * x[2]) *
        (2 * ϵ * sin(ω * t) * x[1] + 1 - 2 * ϵ * sin(ω * t))
end

function ∇u!(s, x, t)
    ϕ = ϵ * sin(ω * t) * x[1]^2 + (1 - 2 * ϵ * sin(ω * t)) * x[1]
    dϕ = 2 * ϵ * sin(ω * t) * x[1] + 1 - 2 * ϵ * sin(ω * t)
    d2ϕ = 2 * ϵ * sin(ω * t)

    s[1, 1] = -π^2 * A * cos(π * x[2]) * cos(π * ϕ) * dϕ
    s[1, 2] = π^2 * A * sin(π * x[2]) * sin(π * ϕ)
    s[2, 1] =
        -π^2 * A * sin(π * ϕ) * sin(π * x[2]) * dϕ^2 + π * A * cos(π * ϕ) * sin(π * x[2]) * d2ϕ
    s[2, 2] = π^2 * A * cos(π * ϕ) * cos(π * x[2]) * dϕ
end

function σ!(s, _, _)
    s[1, 1] = 1.0
    s[1, 2] = 0.0
    s[2, 1] = 0.0
    s[2, 2] = 1.0
end

function σσᵀ!(s, _, _)
    σ!(s, 0.0, 0.0)
end

### FTLE versus S²
include("ftle.jl")

begin
    # Declare a grid of initial conditions
    xgrid = collect(range(0; stop = 2.0, length = 100))
    ygrid = collect(range(0; stop = 1.0, length = 100))
    ftle_S2_comparison("double_gyre", xgrid, ygrid, 0.0, 5.0, u!, ∇u!, σσᵀ!)
end