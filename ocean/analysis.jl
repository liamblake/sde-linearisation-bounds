"""
A complete analysis of a trajectory in the Gulf Stream, using an SDE model constructed from
altimetry-derived velocity data.

Note that this script relies upon a NetCDF file containing the pre-processed altimetry data, the
path of which is specified as `nc_data`. This file is not included in the repository due to its
size, but can be downloaded from the Copernicus Marine Environment Monitoring Service (CMEMS) at
    https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description

"""

using Dates
using LinearAlgebra
using Random

using ColorSchemes
using DifferentialEquations
using Interpolations
using ProgressMeter
using PyPlot
using StatsBase

include("../computations/gaussian_computation.jl")
include("process_ocean.jl")

include("../pyplot_setup.jl")

Random.seed!(1328472345)

# Signed square root
ssqrt(x) = sign(x) * sqrt(abs(x))

#################################### LOAD DATA AND SETUP MODELS ####################################
# Load data
nc_data = "data/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_1680670084978.nc"
lon, lat, days, dates, u, v, v_err, u_err, ssh, land, dx, _ = parse_glo_phy_l4_daucs(nc_data);

# Construct interpolations
u_interp = linear_interpolation((lon, lat, days), u; extrapolation_bc = 0.0)
v_interp = linear_interpolation((lon, lat, days), v; extrapolation_bc = 0.0)
u_err_interp = linear_interpolation((lon, lat, days), u_err; extrapolation_bc = 0.0)
v_err_interp = linear_interpolation((lon, lat, days), v_err; extrapolation_bc = 0.0)
# Meshgrid-style arrangement of longitude and latitude values - useful for contour maps
glon = repeat(lon, 1, length(lat))
glat = repeat(lat; inner = (1, length(lon)))'

function vel!(s, x, t)
    s[1] = u_interp(x[1], x[2], t)
    s[2] = v_interp(x[1], x[2], t)
    nothing
end

# Finite-difference approximation of ∇u
function ∇u!(s, x, t)
    s[1, 1] = u_interp(x[1] + dx, x[2], t) - u_interp(x[1] - dx, x[2], t)
    s[1, 2] = u_interp(x[1], x[2] + dx, t) - u_interp(x[1], x[2] - dx, t)
    s[2, 1] = v_interp(x[1] + dx, x[2], t) - v_interp(x[1] - dx, x[2], t)
    s[2, 2] = v_interp(x[1], x[2] + dx, t) - v_interp(x[1], x[2] - dx, t)
    rmul!(s, 1 / (2 * dx))
    nothing
end

# Interpolated ssh
ssh_interp = linear_interpolation((lon, lat, days), ssh)

# Diffusion - from data
ε = sqrt(dx)
function σ!(s, x, t)
    s[1, 1] = ssqrt(u_err_interp(x[1], x[2], t))
    s[2, 2] = ssqrt(v_err_interp(x[1], x[2], t))
    s[1, 2] = 0.0
    s[2, 1] = 0.0
    rmul!(s, ε)
    nothing
end

function σσᵀ!(s, x, t)
    s[1, 1] = abs(u_err_interp(x[1], x[2], t))
    s[2, 2] = abs(v_err_interp(x[1], x[2], t))
    s[1, 2] = 0.0
    s[2, 1] = 0.0
    rmul!(s, ε^2)
    nothing
end

# Window for plotting
lon_range = (-66, -52)
lat_range = (34, 46)
biglon = minimum(lon_range):0.01:maximum(lon_range)
biglat = minimum(lat_range):0.01:maximum(lat_range)

# Output settings
fdir = "output/ocean"
dpi = 600

# Timeframe details
i1 = 1
i2 = 8
t1 = days[i1]
t2 = days[i2]
dt = 1 / (1 * 24)
tspan = t1:dt:t2

################################## STOCHASTIC SENSITIVITY FIELDS ###################################
# Plot the stochastic sensitivity field over the plotting window, at both the resolution of the data
# and a higher resolution.
if false
    wtmp = Vector{Vector}(undef, length(tspan))
    Σtmp = Vector{Matrix}(undef, length(tspan))

    function zero_σσ!(s, _, _)
        s .= 0.0
    end

    for (str, res) in [("grid", dx), ("high", dx / 10.0)]
        x_grid = lon_range[1]:res:lon_range[2]
        y_grid = lat_range[1]:res:lat_range[2]
        inits = [[x, y] for x in x_grid, y in y_grid][:]

        S2 = Vector{Float64}(undef, length(inits))
        s2 = Vector{Float64}(undef, length(inits))
        cov2 = Vector{Float64}(undef, length(inits))
        ftle = Vector{Float64}(undef, length(inits))

        @showprogress desc = "Computing matrices for $str resolution..." for (i, x0) in
                                                                             enumerate(inits)
            gaussian_computation!(wtmp, Σtmp, 2, vel!, ∇u!, σσᵀ!, x0, zeros(2, 2), tspan)
            s2[i], S2[i] = eigvals(Σtmp[end])
            cov2[i] = Σtmp[end][1, 2]

            # Let's also compute the FTLE
            gaussian_computation!(wtmp, Σtmp, 2, vel!, ∇u!, zero_σσ!, x0, [1.0 0.0; 0.0 1.0], tspan)
            _, ftle[i] = eigvals(Σtmp[end])
        end

        # Stochastic sensitivity
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(S2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"S^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/S2_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Minimum eigenvalue
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(s2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"S^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/s2min_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Covariance
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(cov2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"s^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/cov_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Ratio between eigenvalues
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(s2 ./ S2, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"S^2/s^2")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/ratio_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Finite-time Lyapunov exponent
        fig, ax = subplots()
        ax.set_xlabel("°W")
        ax.set_ylabel("°N")

        pc = ax.pcolormesh(
            x_grid,
            y_grid,
            reshape(ftle, length(x_grid), length(y_grid))';
            cmap = :pink,
            norm = "log",
            rasterized = true,
        )
        colorbar(pc; ax = ax, location = "top", aspect = 40, label = "FTLE stretching rate")

        # Overlay land
        ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

        xlim(lon_range)
        ylim(lat_range)
        ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
        fig.savefig("$fdir/ftle_field_$str.pdf"; dpi = dpi, bbox_inches = "tight")
        close(fig)

        # Extract some robust sets
        for R in [0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
            fig, ax = subplots()
            ax.set_xlabel("°W")
            ax.set_ylabel("°N")

            ax.pcolormesh(
                x_grid,
                y_grid,
                (reshape(S2, length(x_grid), length(y_grid)) .< R)';
                cmap = "twocolor_blue",
                rasterized = true,
            )

            # Overlay land
            ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

            # Trying to get an invisible colourbar - overlay a transparent copy of the field, but
            # with a completely transparent colormap. A dirty hack, according to my Copilot overlord.
            p = ax.pcolormesh(
                x_grid,
                y_grid,
                (reshape(S2, length(x_grid), length(y_grid)) .< R)';
                cmap = ColorMap("nothing", [RGBA(0.0, 0.0, 0.0, 0.0), RGBA(0.0, 0.0, 0.0, 0.0)]),
                rasterized = true,
            )

            cb = colorbar(p; ax = ax, location = "top", aspect = 40, label = L"S^2")

            # Hide label, ticks, and outline
            cb.ax.xaxis.label.set_color(:white)
            cb.ax.tick_params(; axis = "x", colors = :white)
            cb.outline.set_visible(false)

            xlim(lon_range)
            ylim(lat_range)
            ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))

            fig.savefig("$fdir/S2_robust_$(str)_$R.pdf"; dpi = dpi, bbox_inches = "tight")
            close(fig)
        end
    end
end

### Repeat analysis, but with a simple eddy parameterisation
eddyT = 8.0  # Eddy lifetime
# eddyS = 18000m    # Eddy diffusivity, converted to degrees in each direction
middle_lat = lat_range[1] + (lat_range[2] - lat_range[1]) / 2
eddyS1 = sqrt(arc_to_meridonal(WGS84, middle_lat) * 18000)
eddyS2 = sqrt(arc_to_parallel(WGS84, middle_lat) * 18000)

res = 0.05

# dx = dx * 2

function vel_eddy!(s, x, t)
    # Velocity
    s[1] = u_interp(x[1], x[2], t) + x[3]
    s[2] = v_interp(x[1], x[2], t) + x[4]

    # Perturbations
    s[3] = -x[3] / eddyT
    s[4] = -x[4] / eddyT
end

# Finite-difference approximation of ∇u from interpolated data + analytic derivatives
function ∇u_eddy!(s, x, t)
    s .= 0

    s[1, 1] = u_interp(x[1] + dx, x[2], t) - u_interp(x[1] - dx, x[2], t)
    s[1, 2] = u_interp(x[1], x[2] + dx, t) - u_interp(x[1], x[2] - dx, t)
    s[2, 1] = v_interp(x[1] + dx, x[2], t) - v_interp(x[1] - dx, x[2], t)
    s[2, 2] = v_interp(x[1], x[2] + dx, t) - v_interp(x[1], x[2] - dx, t)
    rmul!(s, 1 / (2 * dx))

    s[1, 3] = 1.0
    s[2, 4] = 1.0

    s[3, 3] = -1 / eddyT
    s[4, 4] = -1 / eddyT
end

# Diffusion matrix
function σ_eddy!(s, x, t)
    s .= 0.0

    s[3, 1] = eddyS1
    s[4, 2] = eddyS2
end

function σσᵀ_eddy!(s, x, t)
    s .= 0.0

    s[3, 3] = eddyS1^2
    s[4, 4] = eddyS2^2
end

begin
    wtmp = Vector{Vector}(undef, length(tspan))
    Σtmp = Vector{Matrix}(undef, length(tspan))

    x_grid = lon_range[1]:res:lon_range[2]
    y_grid = lat_range[1]:res:lat_range[2]
    inits = [[x, y] for x in x_grid, y in y_grid][:]

    S2 = Vector{Float64}(undef, length(inits))
    s2 = Vector{Float64}(undef, length(inits))

    @showprogress desc = "Computing eddy matrices..." for (i, x0) in enumerate(inits)
        gaussian_computation!(wtmp, Σtmp, 4, vel_eddy!, ∇u_eddy!, σσᵀ_eddy!, [x0; 0.0; 0.0], zeros(4, 4), tspan)
        s2[i], S2[i] = eigvals(Σtmp[end][1:2, 1:2])
    end

    # Stochastic sensitivity
    fig, ax = subplots()
    ax.set_xlabel("°W")
    ax.set_ylabel("°N")

    pc = ax.pcolormesh(
        x_grid,
        y_grid,
        reshape(S2, length(x_grid), length(y_grid))';
        cmap = :pink,
        norm = "log",
        rasterized = true,
    )
    colorbar(pc; ax = ax, location = "top", aspect = 40, label = L"S^2")

    # Overlay land
    ax.pcolor(lon, lat, land'; cmap = "twocolor", rasterized = true, zorder = 1)

    xlim(lon_range)
    ylim(lat_range)
    ax.set_xticks(ax.get_xticks(), -Int64.(ax.get_xticks()))
    fig.savefig("$fdir/S2_field_eddy.pdf"; dpi = dpi, bbox_inches = "tight")
    close(fig)
end
