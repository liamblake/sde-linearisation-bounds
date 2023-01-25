Julia code implementing the results of

> L. Blake, J. Maclean, and S. Balasuriya, *Explicit Gaussian characterisation of model uncertainty
in the limit of small noise*, SIAM/ASA Journal on Uncertainty Quantification, (2023), submitted for publication.


## File structure
The Julia files in the root directory serve the following purposes:

- `analysis.jl`: Validation of the main results, including generation and saving of plots.

- `covariance.jl`: Computation of the limiting covariance matrix $\Sigma(x,t)$, with options to use either the direct integral expression or matrix ODE.

- `main.jl`: The main setup and call to validation functions. Run everything in this file to generate all the figures.

- `models.jl`: Definition of the model interface and the 2D Rossby wave example.

- `solve_sde.jl`: Generation of all realisation data by directly solving the original SDE and the limiting SDE jointly.
