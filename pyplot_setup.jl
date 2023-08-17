using ColorSchemes
using PyPlot

# Register colormaps
# Thermal
PyPlot.matplotlib.colormaps.register(
    ColorMap(ColorSchemes.thermal.colors);
    name = "thermal",
    force = true,
)

# Set universal default parameters to enforce consistent plot style
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["font.family"] = "serif"
rcParams["font.size"] = "14"

rcParams["image.cmap"] = "thermal"

# Legend
# rcParams["legend.facecolor"] = "white"
rcParams["legend.fancybox"] = false
rcParams["legend.framealpha"] = 1.0