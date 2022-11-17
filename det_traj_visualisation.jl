"""
Visualisation of deterministic trajectories. This was used to select initial conditions.
"""

using DifferentialEquations
using Plots

include("models.jl")

# Rossby wave example
rossby = ex_rossby(x -> x)

t₀ = 0.0
T = 2.5

x₀ = [0.0, 1.0]
prob = ODEProblem(rossby.velocity!, x₀, (t₀, T))
sol1 = solve(prob)

prob = remake(prob, u0=[1.1, π/2])
sol2 = solve(prob)

prob = remake(prob, u0=[0.4, 1.0])
sol3 = solve(prob)

p = plot(sol1, idxs=(1, 2), legend=false)
plot!(p, sol2, idxs=(1, 2), legend=false, xlims=(0, π), ylims=(0, π))
plot!(p, sol3, idxs=(1, 2), legend=false, xlims=(0, π), ylims=(0, π))


# Streamfunction
A = 1.0
c = 0.5
K = 4.0
l₁ = 2.0
c₁ = π
k₁ = 1.0
ϵ = 0.3
ψ = (x, y, t) -> -c * y + A * sin(K * x) * sin(y) + ϵ * sin(k₁ * (x - c₁ * t)) * sin(l₁ * y)
x = 0:0.01:π
y = 0:0.01:π

ps = [contour(x, y, (x, y) -> ψ(x, y, t), legend=false) for t in 0:0.2:1]
plot(ps..., layout=(3, 2))

