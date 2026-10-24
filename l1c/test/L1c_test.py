import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from geopy.distance import geodesic
import mgrs

# Directory configuration
root_dir = r'C:/Users/Usuario/Documents/GitHub/EODP/EODP_TER_2021/EODP-TS-L1C'
geometry_dir = os.path.join(root_dir, 'input', 'gm_alt100_act_150')
results_dir = os.path.join(root_dir, 'output_Boris')
report_path = os.path.join(results_dir, 'l1c_test.log')

selected_band = 'VNIR-0'
central_row = 50
central_col = 75

# The L1C product uses MGRS precision 3 (100 m grid cells).
# Change this if the processor's MGRS precision changes.
l1c_grid_spacing = 100
report = []

def write_log(message):
    print(message)
    report.append(message)


def load_l1b_coordinates():
    file_path = os.path.join(geometry_dir, 'geolocation.nc')
    with Dataset(file_path, 'r') as data:
        projection = data.groups['projection']
        latitude = np.asarray(projection.variables['latitude'][:])
        longitude = np.asarray(projection.variables['longitude'][:])
    return latitude, longitude


def load_l1c_coordinates(band_name):
    file_path = os.path.join(results_dir, f'l1c_toa_{band_name}.nc')
    with Dataset(file_path, 'r') as data:
        latitude = np.asarray(data.variables['lat'][:])
        longitude = np.asarray(data.variables['lon'][:])
    return latitude, longitude


def compare_geographical_grids(lat_b, lon_b, lat_c, lon_c):
    fig, axis = plt.subplots(figsize=(20, 10))

    axis.plot(lon_b.ravel(), lat_b.ravel(), '.', color='tab:blue',
              markersize=2, label='L1B grid')
    axis.plot(lon_c.ravel(), lat_c.ravel(), '.', color='tab:red',
              markersize=6, label='L1C grid (MGRS)')

    # Trace the actual L1B pixel-centre perimeter, closing the polygon.
    perimeter_lon = np.concatenate((
        lon_b[0, :], lon_b[1:, -1],
        lon_b[-1, -2::-1], lon_b[-2:0:-1, 0], lon_b[0, :1]
    ))
    perimeter_lat = np.concatenate((
        lat_b[0, :], lat_b[1:, -1],
        lat_b[-1, -2::-1], lat_b[-2:0:-1, 0], lat_b[0, :1]
    ))
    axis.plot(perimeter_lon, perimeter_lat, '-', color='black',
              linewidth=1.5, label='L1B footprint', zorder=5)

    average_lat = np.mean(lat_b)
    axis.set_aspect(1 / np.cos(np.deg2rad(average_lat)))
    axis.set(title='L1B and L1C grids',
             xlabel='Longitude [deg]', ylabel='Latitude [deg]')
    axis.title.set_fontsize(20)
    axis.xaxis.label.set_size(16)
    axis.yaxis.label.set_size(16)
    axis.tick_params(labelsize=14)
    axis.grid(True)
    axis.legend(fontsize=16, markerscale=2)

    destination = os.path.join(results_dir, 'l1b_l1c_grids.png')
    fig.savefig(destination)
    plt.close(fig)
    write_log(f'plot saved to: {destination}')


def calculate_gsd(latitudes, longitudes):
    coordinates = list(zip(latitudes, longitudes))
    return np.asarray([
        geodesic(point_a, point_b).meters
        for point_a, point_b in zip(coordinates[:-1], coordinates[1:])
    ])


def save_gsd_graph(distances, heading, x_label, line_color, output_name):
    fig, axis = plt.subplots(figsize=(20, 8))
    axis.plot(np.arange(distances.size), distances,
              color=line_color, linewidth=1.5)
    axis.set_title(heading, fontsize=20)
    axis.set_xlabel(x_label, fontsize=16)
    axis.set_ylabel('GSD [m]', fontsize=16)
    axis.tick_params(labelsize=14)
    axis.ticklabel_format(axis='y', useOffset=False)
    axis.grid(True)

    destination = os.path.join(results_dir, output_name)
    fig.savefig(destination)
    plt.close(fig)
    write_log(f'plot saved to: {destination}')


def log_gsd(direction, values):
    if values.size == 0:
        raise ValueError(f'No adjacent grid points found for {direction}')
    write_log(
        f'{direction}: mean = {values.mean():.5f} m, '
        f'min = {values.min():.5f} m, max = {values.max():.5f} m'
    )


def analyse_l1b_sampling(latitude, longitude):
    gsd_act = calculate_gsd(latitude[central_row, :], longitude[central_row, :])
    gsd_alt = calculate_gsd(latitude[:, central_col], longitude[:, central_col])

    log_gsd(f'L1B GSD ACT (row {central_row})', gsd_act)
    log_gsd(f'L1B GSD ALT (column {central_col})', gsd_alt)

    save_gsd_graph(gsd_act, f'GSD across track (central row {central_row})',
                   'ACT pixel [-]', 'tab:blue', f'gsd_l1b_row{central_row}.png')
    save_gsd_graph(gsd_alt, f'GSD along track (central column {central_col})',
                   'ALT pixel [-]', 'tab:red', f'gsd_l1b_col{central_col}.png')


def analyse_l1c_sampling(latitude, longitude):
    # MGRS is already a dependency of the L1C processing module.
    # Encode each unordered L1C coordinate at 1 m precision, then round
    # its MGRS easting/northing within the 100 km square to the 100 m grid.
    converter = mgrs.MGRS()
    references = [
        converter.toMGRS(float(la), float(lo), MGRSPrecision=5)
        for la, lo in zip(latitude.ravel(), longitude.ravel())
    ]
    references = [
        ref.decode('ascii') if isinstance(ref, bytes) else ref
        for ref in references
    ]

    # This particular small scene should lie in a single 100 km square.
    squares = {ref[:-10] for ref in references}
    if len(squares) != 1:
        raise ValueError('L1C grid crosses MGRS 100 km squares; '
                         'the grid indices need a multi-square conversion.')

    east_m = np.array([int(ref[-10:-5]) for ref in references])
    north_m = np.array([int(ref[-5:]) for ref in references])
    east_idx = np.rint(east_m / l1c_grid_spacing).astype(int)
    north_idx = np.rint(north_m / l1c_grid_spacing).astype(int)

    # Ensure each reconstructed cell has exactly one data point.
    if len(set(zip(east_idx, north_idx))) != len(latitude):
        raise ValueError('Repeated L1C grid indices after MGRS reconstruction.')

    unique_rows = np.unique(north_idx)
    unique_cols = np.unique(east_idx)
    middle_row = unique_rows[len(unique_rows) // 2]
    middle_col = unique_cols[len(unique_cols) // 2]

    row_points = np.flatnonzero(north_idx == middle_row)
    col_points = np.flatnonzero(east_idx == middle_col)
    row_points = row_points[np.argsort(east_idx[row_points])]
    col_points = col_points[np.argsort(north_idx[col_points])]

    # Calculate GSD only between truly adjacent cells (not across holes).
    row_adjacent = np.diff(east_idx[row_points]) == 1
    col_adjacent = np.diff(north_idx[col_points]) == 1
    row_gsd = calculate_gsd(latitude[row_points], longitude[row_points])[row_adjacent]
    col_gsd = calculate_gsd(latitude[col_points], longitude[col_points])[col_adjacent]

    log_gsd('L1C east-west GSD (middle MGRS row)', row_gsd)
    log_gsd('L1C north-south GSD (middle MGRS column)', col_gsd)
    save_gsd_graph(row_gsd, 'L1C GSD - central MGRS row',
                   'ALT pixel [-]', 'tab:red',
                   'gsd_l1c_row.png')
    save_gsd_graph(col_gsd, 'L1C GSD - central MGRS column',
                   'ACT pixel [-]', 'tab:blue',
                   'gsd_l1c_col.png')

def main():
    lat_b, lon_b = load_l1b_coordinates()
    lat_c, lon_c = load_l1c_coordinates(selected_band)

    write_log('L1C grid assessment (without reference-output cross-validation)')
    write_log(f'L1B pixels: {lat_b.size}, L1C pixels: {lat_c.size}')
    write_log('-' * 80)

    compare_geographical_grids(lat_b, lon_b, lat_c, lon_c)
    analyse_l1b_sampling(lat_b, lon_b)
    analyse_l1c_sampling(lat_c, lon_c)

    with open(report_path, 'w') as output:
        output.write('\n'.join(report) + '\n')
    print(f'log saved to: {report_path}')


if __name__ == '__main__':
    main()
