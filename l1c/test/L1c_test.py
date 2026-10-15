import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from geopy.distance import geodesic

# Directory configuration
root_dir = r'C:/Users/Usuario/Documents/GitHub/EODP/EODP_TER_2021/EODP-TS-L1C'
geometry_dir = os.path.join(root_dir, "input", "gm_alt100_act_150")
results_dir = os.path.join(root_dir, "output_Boris")
report_path = os.path.join(results_dir, "l1c_test.log")

# L1C band selected for the geographical grid
selected_band = "VNIR-0"

# Pixel indices corresponding to the central row and column
central_row = 50
central_col = 75

# Store messages for the final report
report = []


def write_log(message):
    print(message)
    report.append(message)


def load_l1b_coordinates():
    # Extract pixel-centre geographical coordinates from the geometry module
    file_path = os.path.join(geometry_dir, "geolocation.nc")

    with Dataset(file_path, "r") as data:
        projection = data.groups["projection"]
        latitude = np.asarray(projection.variables["latitude"][:])
        longitude = np.asarray(projection.variables["longitude"][:])

    return latitude, longitude


def load_l1c_coordinates(band_name):
    # Retrieve the MGRS coordinates associated with the selected L1C band
    file_path = os.path.join(results_dir, f"l1c_toa_{band_name}.nc")

    with Dataset(file_path, "r") as data:
        latitude = np.asarray(data.variables["lat"][:])
        longitude = np.asarray(data.variables["lon"][:])

    return latitude, longitude


def compare_geographical_grids(lat_b, lon_b, lat_c, lon_c):
    # Display the two geographical grids in the same coordinate system
    fig, axis = plt.subplots(figsize=(20, 10))

    axis.plot(
        lon_b.ravel(), lat_b.ravel(),
        '.', color='tab:blue', markersize=2,
        label='L1B grid'
    )

    axis.plot(
        lon_c.ravel(), lat_c.ravel(),
        '.', color='tab:red', markersize=6,
        label='L1C grid (MGRS)'
    )

    # Correct the aspect ratio to account for latitude-dependent longitude distances
    average_lat = np.mean(lat_b)
    axis.set_aspect(1 / np.cos(np.deg2rad(average_lat)))

    axis.set(
        title='L1B and L1C grids',
        xlabel='Longitude [deg]',
        ylabel='Latitude [deg]'
    )

    axis.title.set_fontsize(20)
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.tick_params(labelsize=14)
    axis.grid(True)
    axis.legend(fontsize=16, markerscale=2)

    destination = os.path.join(results_dir, "l1b_l1c_grids.png")
    fig.savefig(destination)
    plt.close(fig)

    write_log(f"plot saved to: {destination}")


def calculate_gsd(latitudes, longitudes):
    # Determine the WGS84 surface distance between successive pixel centres
    coordinates = list(zip(latitudes, longitudes))

    distances = [
        geodesic(point_a, point_b).meters
        for point_a, point_b in zip(coordinates[:-1], coordinates[1:])
    ]

    return np.asarray(distances)


def save_gsd_graph(distances, heading, x_label, line_color, output_name):
    # Generate and export a graph of the spatial sampling distances
    fig, axis = plt.subplots(figsize=(20, 8))

    pixel_indices = np.arange(distances.size)
    axis.plot(pixel_indices, distances, color=line_color, linewidth=1.5)

    axis.set_title(heading, fontsize=20)
    axis.set_xlabel(x_label, fontsize=16)
    axis.set_ylabel('GSD [m]', fontsize=16)
    axis.tick_params(labelsize=14)
    axis.ticklabel_format(axis='y', useOffset=False)
    axis.grid(True)

    destination = os.path.join(results_dir, output_name)
    fig.savefig(destination)
    plt.close(fig)

    write_log(f"plot saved to: {destination}")


def analyse_l1b_sampling(latitude, longitude):
    # Extract the across-track and along-track geographical profiles
    across_track_lat = latitude[central_row, :]
    across_track_lon = longitude[central_row, :]

    along_track_lat = latitude[:, central_col]
    along_track_lon = longitude[:, central_col]

    # Calculate distances between neighbouring pixel centres
    act_distances = calculate_gsd(across_track_lat, across_track_lon)
    alt_distances = calculate_gsd(along_track_lat, along_track_lon)

    # Record statistical information for both directions
    for direction, index, values in [
        ("ACT", central_row, act_distances),
        ("ALT", central_col, alt_distances)
    ]:
        write_log(
            f"GSD {direction} (index {index}): "
            f"mean = {np.mean(values):.3f} m, "
            f"min = {np.min(values):.3f} m, "
            f"max = {np.max(values):.3f} m"
        )

    # Export the across-track profile
    save_gsd_graph(
        act_distances,
        f'GSD across track (central row {central_row})',
        'ACT pixel [-]',
        'tab:blue',
        f'gsd_l1b_row{central_row}.png'
    )

    # Export the along-track profile
    save_gsd_graph(
        alt_distances,
        f'GSD along track (central column {central_col})',
        'ALT pixel [-]',
        'tab:red',
        f'gsd_l1b_col{central_col}.png'
    )


def main():
    # Read both geographical datasets
    lat_b, lon_b = load_l1b_coordinates()
    lat_c, lon_c = load_l1c_coordinates(selected_band)

    write_log("L1C grid assessment (without reference-output cross-validation)")
    write_log(f"L1B pixels: {lat_b.size}, L1C pixels: {lat_c.size}")
    write_log("-" * 80)

    # Compare grid locations and evaluate L1B spatial sampling
    compare_geographical_grids(lat_b, lon_b, lat_c, lon_c)
    analyse_l1b_sampling(lat_b, lon_b)

    # Save the collected messages to a text log
    with open(report_path, "w") as output:
        output.write("\n".join(report) + "\n")

    print(f"log saved to: {report_path}")


if __name__ == "__main__":
    main()