using ColorSchemes
using PyPlot

ENV["PYTHON"] = "/usr/bin/python3"

# Register colormaps
# Thermal
register_cmap(:thermal, ColorMap(ColorSchemes.thermal.colors))

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

save_dpi = 600