using Random

include("validation.jl")

function validate_all(
    N::Int64;
    attempt_reload::Bool = true,
    save_on_generation::Bool = true,
)
    Random.seed!(20220805)

    ### Perturbed Rossby wave ###
    # Initial conditions
    x₀s = [[1.5, 1.0], [0.0, 1.0]]
    # Time interval of interest
    t₀ = 0.0
    T = 1.0

    convergence_validation(
        ex_rossby(),
        x₀s,
        t₀,
        T,
        N,
        attempt_reload = attempt_reload,
        save_on_generation = save_on_generation,
    )

end
