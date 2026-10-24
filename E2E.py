
import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

# Directory containing the L1B TOA products
base_folder = r"C:/Users/Usuario/Documents/GitHub/EODP/EODP_TER_2021/EODP-TS-E2E/output_Boris_l1b"

# Geolocation file from the geometry module
geo_file = r"C:/Users/Usuario/Documents/GitHub/EODP/EODP_TER_2021/EODP-TS-E2E/gm_out/geolocation.nc"

# Plot font sizes
TITLE_SIZE = 16
LABEL_SIZE = 14
TICK_SIZE = 12
CBAR_SIZE = 13

# Read geographical coordinates and altitude
with Dataset(geo_file, "r") as nc:
    projection = nc.groups["projection"]

    latitude = np.ma.asarray(
        projection.variables["latitude"][:], dtype=float
    ).filled(np.nan)

    longitude = np.ma.asarray(
        projection.variables["longitude"][:], dtype=float
    ).filled(np.nan)

    altitude = np.ma.asarray(
        projection.variables["altitude"][:], dtype=float
    ).filled(np.nan)

    altitude_units = getattr(projection.variables["altitude"], "units", "")

# Approximate equal-distance scaling
mean_lat = np.nanmean(latitude)
geo_aspect = 1 / np.cos(np.deg2rad(mean_lat))

# ---------------- GEOLOCATED ALTITUDE ----------------

# Only this figure uses a 16:9 canvas
fig_alt, ax_alt = plt.subplots(figsize=(8, 12))

im_alt = ax_alt.pcolormesh(
    longitude,
    latitude,
    altitude,
    cmap="terrain",
    shading="auto"
)

ax_alt.set_aspect(geo_aspect)
ax_alt.set_xlabel("Longitude [deg]", fontsize=LABEL_SIZE)
ax_alt.set_ylabel("Latitude [deg]", fontsize=LABEL_SIZE)
ax_alt.set_title("Geolocated altitude", fontsize=TITLE_SIZE)
ax_alt.tick_params(labelsize=TICK_SIZE)
ax_alt.grid(True, alpha=0.3)

cbar_alt = fig_alt.colorbar(im_alt, ax=ax_alt)
cbar_alt.set_label(
    f"Altitude [{altitude_units}]" if altitude_units else "Altitude",
    fontsize=CBAR_SIZE
)
cbar_alt.ax.tick_params(labelsize=TICK_SIZE)

alt_path = os.path.join(base_folder, "geolocated_altitude.png")
fig_alt.savefig(alt_path, dpi=300, bbox_inches="tight")
plt.close(fig_alt)

print(f"Saved geolocated altitude: {alt_path}")

# ---------------- TOA MAPS AND HISTOGRAMS ----------------

for band in range(4):

    file_path = os.path.join(base_folder, f"l1b_toa_VNIR-{band}.nc")

    # Read TOA radiance
    with Dataset(file_path, "r") as nc:
        toa = np.ma.asarray(
            nc.variables["toa"][:], dtype=float
        ).filled(np.nan)

    # Verify grid compatibility
    if toa.shape != latitude.shape or toa.shape != longitude.shape:
        raise ValueError(
            f"VNIR{band}: TOA shape {toa.shape} does not match "
            f"geolocation shapes {latitude.shape}, {longitude.shape}"
        )

    # ---------------- GEOLOCATED TOA RADIANCE ----------------

    fig, ax = plt.subplots(figsize=(8, 12))

    im = ax.pcolormesh(
        longitude,
        latitude,
        toa,
        cmap="jet",
        shading="auto"
    )

    ax.set_aspect(geo_aspect)

    ax.set_xlabel("Longitude [deg]", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitude [deg]", fontsize=LABEL_SIZE)
    ax.set_title(f"TOA radiance - VNIR{band}", fontsize=TITLE_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(
        r"TOA radiance [mW m$^{-2}$ sr$^{-1}$]",
        fontsize=CBAR_SIZE
    )
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    # Save geolocated radiance map
    output_path = os.path.join(base_folder, f"toaVNIR{band}.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved geolocated TOA VNIR{band}: {output_path}")

    # ---------------- PIXEL HISTOGRAM ----------------

    # Remove invalid values
    toa_1d = toa[np.isfinite(toa)]

    # Define bins of width 0.1
    bin_width = 0.1
    bin_start = np.floor(toa_1d.min() / bin_width) * bin_width
    bin_end = np.ceil(toa_1d.max() / bin_width) * bin_width
    bins = np.arange(bin_start, bin_end + 1.5 * bin_width, bin_width)

    # Create histogram
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))

    ax_hist.hist(toa_1d, bins=bins, edgecolor="black")

    ax_hist.set_xlabel(
        r"TOA radiance [mW m$^{-2}$ sr$^{-1}$]",
        fontsize=LABEL_SIZE
    )
    ax_hist.set_ylabel("Number of pixels", fontsize=LABEL_SIZE)
    ax_hist.set_title(
        f"Pixel histogram - VNIR{band}",
        fontsize=TITLE_SIZE
    )
    ax_hist.tick_params(labelsize=TICK_SIZE)
    ax_hist.grid(True)

    # Save histogram
    hist_path = os.path.join(base_folder, f"histVNIR{band}.png")
    fig_hist.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close(fig_hist)

    print(f"Saved histogram VNIR{band}: {hist_path}")

print("All nine figures saved successfully.")
