using LinearAlgebra
using Statistics

using NCDatasets

"""
Describes an ellipsoidal model of the Earth, with semi-major axis a and inverse flattening inv_f.
"""
struct EarthModel
    a::Float64
    inv_f::Float64
end

"""
    e(self::EarthModel)::Float64
Calculate the eccentricity of a symmetric cross-sectional ellipse of an EarthModel.
"""
function e(self::EarthModel)::Float64
    f = 1.0 / self.inv_f
    return 2 * f - f^2
end

WGS84 = EarthModel(6378187.0, 298.257223563);

"""
    arc_to_meridonal(self::EarthModel, φ)
Compute the length (in degrees) of a 1 metre change in latitude, at latitude φ.
"""
function arc_to_meridonal(self::EarthModel, φ)
    e² = e(self)^2
    return @. 180 / pi * (1 - e²) / self.a * (1 - e² * sind(φ)^2)^(3 / 2)
end
"""
    arc_to_meridonal(self::EarthModel, φ)
Compute the length (in degrees) of a 1 metre change in longitude, at latitude φ.
"""
function arc_to_parallel(self::EarthModel, φ)
    e² = e(self)^2
    return @. 180 / pi * (1 - e²) / self.a * (1 - e² * sind.(φ)^2)^(1 / 2) / cosd.(φ)
end

"""
    parse_glo_phy_l4_daucs(file::String, interp_func::Function, extras = []; interp_kwargs...)

Parse a NetCDF file containing a subset of the "Global Ocean Gridded L 4 Sea Surface Heights And
Derived Variables Reprocessed 1993 Ongoing" Copernicus data. The NetCDF file is read from the path
`file`. The function used to interpolate the velocity data is given by `interp_func`, which should
match the interface provided by the `Interpolations.jl` package. Any extra variables of interest to
also extract from the NetCDF file can be specified by passing their names as strings in the `extras`
array. Any keyword arguments to be passed to `interp_func` can be specified in `interp_kwargs`.

The output is a tuple containing:
    - A vector of longitudinal gridpoints
    - A vector of latitudinal gridpoints
    - A vector of temporal gridpoints, as the number of days since the start of the data.
    - A vector of the raw temporal gridpoints, as DateTime objects.
    - The zonal velocity data, in degrees per day and evaluated at each spatial and temporal
      gridpoint.
    - The meridional velocity data, in degrees per day and evaluated at each spatial and temporal
      gridpoint.
    - The zonal velocity error field, in degrees per day and evaluated at each spatial and temporal
      gridpoint.
    - The meridional velocity error field, in degrees per day and evaluated at each spatial and
      temporal gridpoint.
    - The sea surface height data, in metres and evaluated at each spatial and temporal gridpoint.
    - A boolean array indicating whether each spatial gridpoint is land or not. Note that the
      presence of land is determined by finding the gridpoints for which the either velocity
      component is missing for ANY time.
    - The spatial resolution of the data, in degrees.
    - Any additional variables specified in `extras`, all stored in a vector.

"""
function parse_glo_phy_l4_daucs(file::String, extras = [])
    nc = Dataset(file)
    lat = Array(nc["latitude"])
    lon = Array(nc["longitude"])
    time = Array(nc["time"])
    v = Array(nc["vgos"])
    u = Array(nc["ugos"])
    v_err = Array(nc["err_vgosa"])
    u_err = Array(nc["err_ugosa"])
    # Sea surface height
    ssh = Array(nc["adt"])

    # Load any extra variables of interest
    extras_out = []
    for e in extras
        push!(extras_out, nc[e][:, :])
    end

    # Convert velocities to deg/day
    vsize = (length(lon), length(lat), length(time))
    v = v .* repeat(arc_to_parallel(WGS84, lat)'; outer = (vsize[1], 1, vsize[3])) * 86400
    u = u .* repeat(arc_to_meridonal(WGS84, lat)'; outer = (vsize[1], 1, vsize[3])) * 86400
    v_err = v_err .* repeat(arc_to_parallel(WGS84, lat)'; outer = (vsize[1], 1, vsize[3])) * 86400
    u_err = u_err .* repeat(arc_to_meridonal(WGS84, lat)'; outer = (vsize[1], 1, vsize[3])) * 86400

    # Extract the land and set to zero velocity
    land_u = isnan.(u)
    land_v = isnan.(v)
    land = prod(land_u .|| land_v; dims = 3)[:, :, 1]
    u[land_u] .= 0.0
    v[land_v] .= 0.0

    u_err[isnan.(u_err)] .= mean(u_err[.!isnan.(u_err)])
    v_err[isnan.(v_err)] .= mean(v_err[.!isnan.(v_err)])

    # rescale time to start from 0
    mil_to_days = s -> s.value / (1000 * 60 * 60 * 24)
    days = mil_to_days.(time .- time[1])

    # Resolution of data
    dx = max(abs(lon[2] - lon[1]), abs(lat[2] - lat[1]))

    println("Successfully loaded and parsed $(file)!")
    return lon, lat, days, time, u, v, v_err, u_err, ssh, land, dx, extras_out
end