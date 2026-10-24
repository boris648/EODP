
# CROSS-VALIDATE ISM OUTPUTS AGAINST REFERENCE

import os
import numpy as np
import netCDF4 as nc

# Directory configuration
DIR_A = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-ISM\output"
DIR_B = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-ISM\outputBoris"

RTOL = 1e-6
ATOL = 1e-10


def compare_file(path_a, path_b):

    file_ok = True

    with nc.Dataset(path_a, "r") as ds_a, nc.Dataset(path_b, "r") as ds_b:

        vars_a = set(ds_a.variables)
        vars_b = set(ds_b.variables)

        # Check for missing variables
        missing_a = vars_b - vars_a
        missing_b = vars_a - vars_b

        for var in sorted(missing_a):
            print(f"  [MISSING IN REFERENCE] {var}")
            file_ok = False

        for var in sorted(missing_b):
            print(f"  [MISSING IN OUTPUT] {var}")
            file_ok = False

        # Compare common variables
        common_vars = sorted(vars_a & vars_b)

        for var in common_vars:

            data_a = np.ma.asarray(
                ds_a.variables[var][:], dtype=float
            ).filled(np.nan)

            data_b = np.ma.asarray(
                ds_b.variables[var][:], dtype=float
            ).filled(np.nan)

            # Check dimensions
            if data_a.shape != data_b.shape:
                print(
                    f"  [SHAPE MISMATCH] {var}: "
                    f"{data_a.shape} vs {data_b.shape}"
                )
                file_ok = False
                continue

            # Compare numerical values
            comparison = np.isclose(
                data_a,
                data_b,
                rtol=RTOL,
                atol=ATOL,
                equal_nan=True
            )

            if np.all(comparison):
                print(f"  [OK]   {var}")

            else:
                file_ok = False
                n_diff = np.sum(~comparison)

                print(f"  [DIFF] {var}: {n_diff} values differ")

    return file_ok


from math import pi
from scipy.constants import h, c


def report_analytical_conversion_factors():

    # ---------------------------------------------------------
    # Instrument parameters
    # ---------------------------------------------------------

    D = 0.150                 # Telescope diameter [m]
    f = 0.5262                # Focal length [m]
    Tr = 0.99                 # Optical transmittance [-]

    pixel_size = 30e-6        # Pixel size [m]
    A = pixel_size**2         # Pixel area [m^2]
    tint = 0.00672            # Integration time [s]

    wavelengths = {
        "VNIR-0": 0.490e-6,
        "VNIR-1": 0.665e-6,
        "VNIR-2": 0.865e-6,
        "VNIR-3": 0.945e-6
    }

    QE = 0.8                  # Quantum efficiency [-]
    OCF_GADC = 3.024e-6       # Electron-to-voltage coefficient [V/e-]

    bit_depth = 12
    min_voltage = 0.0         # [V]
    max_voltage = 0.86        # [V]

    # ---------------------------------------------------------
    # Analytical conversion factors
    # ---------------------------------------------------------

    # Radiance -> Irradiance
    C_LI = Tr * (pi / 4) * (D / f)**2

    # Photons -> Electrons
    C_phe = QE

    # Electrons -> Voltage
    C_eV = OCF_GADC

    # Voltage -> DN
    C_VDN = (2**bit_depth - 1) / (max_voltage - min_voltage)

    # ---------------------------------------------------------
    # Prepare terminal and TXT report
    # ---------------------------------------------------------

    lines = []

    lines.append("=" * 110)
    lines.append("ISM ANALYTICAL RADIOMETRIC CONVERSION FACTORS")
    lines.append("=" * 110)

    lines.append("")
    lines.append("Instrument parameters:")
    lines.append(f"  Telescope diameter:       {D} m")
    lines.append(f"  Focal length:             {f} m")
    lines.append(f"  Transmittance:            {Tr}")
    lines.append(f"  Pixel size:               {pixel_size:.3e} m")
    lines.append(f"  Integration time:         {tint} s")
    lines.append(f"  Quantum efficiency:       {QE}")
    lines.append(f"  OCF * G_ADC:              {OCF_GADC:.6e} V/e-")
    lines.append(f"  ADC bit depth:            {bit_depth}")
    lines.append(f"  ADC voltage range:        {min_voltage} - {max_voltage} V")

    lines.append("")
    lines.append("Conversion factors:")

    header = (
        f"{'Band':<9}"
        f"{'Rad->Irr':>15}"
        f"{'Irr->Ph':>15}"
        f"{'Ph->e-':>12}"
        f"{'e-->V':>16}"
        f"{'V->DN':>15}"
        f"{'Total':>17}"
    )

    lines.append(header)
    lines.append("-" * len(header))

    for band, wavelength in wavelengths.items():

        # Irradiance -> Photons
        C_Iph = 1e-3 * A * tint * wavelength / (h * c)

        # Total radiance -> DN coefficient
        C_total = C_LI * C_Iph * C_phe * C_eV * C_VDN

        lines.append(
            f"{band:<9}"
            f"{C_LI:>15.8f}"
            f"{C_Iph:>15.3f}"
            f"{C_phe:>12.4f}"
            f"{C_eV:>16.6e}"
            f"{C_VDN:>15.3f}"
            f"{C_total:>17.6f}"
        )

    lines.append("-" * len(header))
    lines.append("")
    lines.append("Units:")
    lines.append("  Rad->Irr : sr")
    lines.append("  Irr->Ph  : photons / (mW/m^2)")
    lines.append("  Ph->e-   : electrons / photon")
    lines.append("  e-->V    : V / electron")
    lines.append("  V->DN    : DN / V")
    lines.append("  Total    : DN / (mW m^-2 sr^-1)")
    lines.append("")
    lines.append("NOTE: Factors describe the nominal linear conversion.")
    lines.append("MTF, detector nonuniformity, clipping and rounding are excluded.")
    lines.append("=" * 110)

    report = "\n".join(lines)

    # Print to terminal
    print("\n" + report)

    # Save to TXT file
    os.makedirs(DIR_B, exist_ok=True)

    txt_path = os.path.join(
        DIR_B,
        "ISM_analytical_conversion_factors.txt"
    )

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(report + "\n")

    print(f"\nConversion factors saved to: {txt_path}")



def main():

    files_a = {
        f for f in os.listdir(DIR_A)
        if f.endswith(".nc")
    }

    files_b = {
        f for f in os.listdir(DIR_B)
        if f.endswith(".nc")
    }

    common_files = sorted(files_a & files_b)

    print("ISM NetCDF COMPARISON REPORT")
    print(f"Reference: {DIR_A}")
    print(f"Output:    {DIR_B}")
    print(f"RTOL = {RTOL}, ATOL = {ATOL}\n")

    # Check missing files
    missing_a = files_b - files_a
    missing_b = files_a - files_b

    for fname in sorted(missing_a):
        print(f"[MISSING IN REFERENCE] {fname}")

    for fname in sorted(missing_b):
        print(f"[MISSING IN OUTPUT] {fname}")

    print()

    n_match = 0
    n_mismatch = 0

    # Compare files
    for fname in common_files:

        print(f"File: {fname}")

        ok = compare_file(
            os.path.join(DIR_A, fname),
            os.path.join(DIR_B, fname)
        )

        print("  => MATCH\n" if ok else "  => MISMATCH\n")

        n_match += int(ok)
        n_mismatch += int(not ok)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print(f"Matching files:      {n_match}")
    print(f"Mismatching files:   {n_mismatch}")
    print(f"Missing files:       {len(missing_a) + len(missing_b)}")
    print(f"Total common files:  {len(common_files)}")

    if n_mismatch == 0 and not missing_a and not missing_b and n_match > 0:
        print("\nALL ISM OUTPUTS MATCH THE REFERENCE")
    else:
        print("\nISM VALIDATION FAILED OR INCOMPLETE")


if __name__ == "__main__":
    main()
    report_analytical_conversion_factors()
