Numerical validation of the convergence of a small-noise SDE to the linearisation about the deterministic trajectory.

## File structure
The Julia files in the root directory serve the following purposes:

- `analysis.jl`: Validation of the main results, including generation and saving of plots.

- `covariance.jl`: Computation of the limiting covariance matrix $\Sigma(x,t)$.

- `main.jl`:

- `models.jl`:

- `solve_sde.jl`: Generation of data by directly solving the original SDE and the
