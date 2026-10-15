# l1c test: no cross validation with the reference outputs (the l1c points are not ordered)
# 1. plot the l1b grid and the l1c grid over the same plot
# 2. plot the spatial sampling distance (gsd) of the l1b product, central row and central column

import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from geopy.distance import geodesic

# path settings
base_folder = r'C:/Users/Usuario/Documents/GitHub/EODP/EODP_TER_2021/EODP-TS-L1C'
gm_folder = os.path.join(base_folder, "input", "gm_alt100_act_150")
test_folder = os.path.join(base_folder, "output_Boris")
log_file = os.path.join(test_folder, "l1c_test.log")

# band used to read the l1c grid (the grid is the same for all the bands)
grid_band = "VNIR-0"

# central row and column used for the gsd
gsd_row = 50
gsd_col = 75

# lines written to the log file
log_lines = []


def log(msg):
    # print a message and keep it for the log file
    print(msg)
    log_lines.append(msg)


def read_l1b_grid():
    # latitude and longitude of the centre of projection of each l1b pixel (output of the geometry module)
    with Dataset(os.path.join(gm_folder, "geolocation.nc"), "r") as src:
        lat = np.array(src.groups["projection"].variables["latitude"][:])
        lon = np.array(src.groups["projection"].variables["longitude"][:])
    return lat, lon


def read_l1c_grid(band):
    # latitude and longitude of the l1c points (mgrs grid)
    with Dataset(os.path.join(test_folder, "l1c_toa_" + band + ".nc"), "r") as src:
        lat = np.array(src.variables["lat"][:])
        lon = np.array(src.variables["lon"][:])
    return lat, lon


def plot_grids(lat_l1b, lon_l1b, lat_l1c, lon_l1c):
    # l1b grid (blue) and l1c grid (red) over the same plot
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(lon_l1b.flatten(), lat_l1b.flatten(), '.', color='tab:blue', markersize=2, label='L1B grid')
    ax.plot(lon_l1c, lat_l1c, '.', color='tab:red', markersize=6, label='L1C grid (MGRS)')
    # same scale in metres in both axes (1 deg of longitude is shorter than 1 deg of latitude)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat_l1b))))
    ax.set_title('L1B and L1C grids', fontsize=20)
    ax.set_xlabel('Longitude [deg]', fontsize=16)
    ax.set_ylabel('Latitude [deg]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid()
    ax.legend(fontsize=16, markerscale=4)
    out_path = os.path.join(test_folder, "l1b_l1c_grids.png")
    plt.savefig(out_path)
    plt.close(fig)
    log(f"plot saved to: {out_path}")


def gsd(lat, lon):
    # geodesic distance [m] (wgs84 ellipsoid) between consecutive points of a line of pixels
    return np.array([geodesic((lat[k], lon[k]), (lat[k + 1], lon[k + 1])).meters for k in range(len(lat) - 1)])


def plot_gsd(values, title_str, xlabel_str, color, file_name):
    # plot one gsd vector and save it
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(np.arange(len(values)), values, '-', color=color)
    ax.set_title(title_str, fontsize=20)
    ax.set_xlabel(xlabel_str, fontsize=16)
    ax.set_ylabel('gsd [m]', fontsize=16)
    ax.tick_params(labelsize=14)
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.grid()
    out_path = os.path.join(test_folder, file_name)
    plt.savefig(out_path)
    plt.close(fig)
    log(f"plot saved to: {out_path}")


def run_gsd(lat, lon):
    # gsd across track (central row) and along track (central column)
    gsd_act = gsd(lat[gsd_row, :], lon[gsd_row, :])
    gsd_alt = gsd(lat[:, gsd_col], lon[:, gsd_col])

    log(f"gsd act (row {gsd_row}):    mean = {gsd_act.mean():.3f} m, min = {gsd_act.min():.3f} m, max = {gsd_act.max():.3f} m")
    log(f"gsd alt (column {gsd_col}): mean = {gsd_alt.mean():.3f} m, min = {gsd_alt.min():.3f} m, max = {gsd_alt.max():.3f} m")

    plot_gsd(gsd_act, f'gsd across track (central row {gsd_row})', 'ACT pixel [-]', 'tab:blue', f"gsd_l1b_row{gsd_row}.png")
    plot_gsd(gsd_alt, f'gsd along track (central column {gsd_col})', 'ALT pixel [-]', 'tab:red', f"gsd_l1b_col{gsd_col}.png")


if __name__ == "__main__":
    lat_l1b, lon_l1b = read_l1b_grid()
    lat_l1c, lon_l1c = read_l1c_grid(grid_band)
    log("l1c test (no cross validation with the reference outputs)")
    log(f"l1b points: {lat_l1b.size}, l1c points: {lat_l1c.size}")
    log("-" * 80)

    plot_grids(lat_l1b, lon_l1b, lat_l1c, lon_l1c)
    run_gsd(lat_l1b, lon_l1b)

    # write the log file of the test
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"log saved to: {log_file}")