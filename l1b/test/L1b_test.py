# CROSS VALIDATE L1B OUTPUTS EQUALIZED

import os
import numpy as np
import netCDF4 as nc


DIR_A = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\output"
DIR_B = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\outputBoris"

RTOL = 1e-8
ATOL = 1e-10


def compare_file(path_a, path_b):
    ds_a = nc.Dataset(path_a, "r")
    ds_b = nc.Dataset(path_b, "r")

    common_vars = set(ds_a.variables) & set(ds_b.variables)
    file_ok = True

    for var in sorted(common_vars):
        data_a = np.array(ds_a.variables[var][:], dtype=float)
        data_b = np.array(ds_b.variables[var][:], dtype=float)

        if np.allclose(data_a, data_b, rtol=RTOL, atol=ATOL, equal_nan=True):
            print(f"  [OK]   {var}")
        else:
            file_ok = False
            n_diff = np.sum(~np.isclose(data_a, data_b, rtol=RTOL, atol=ATOL, equal_nan=True))
            print(f"  [DIFF] {var}: {n_diff} values differ")

    ds_a.close()
    ds_b.close()
    return file_ok


def main():
    files_a = {f for f in os.listdir(DIR_A) if f.endswith(".nc")}
    files_b = {f for f in os.listdir(DIR_B) if f.endswith(".nc")}
    common_files = sorted(files_a & files_b)

    print("NetCDF COMPARISON REPORT")
    print(f"A: {DIR_A}")
    print(f"B: {DIR_B}\n")

    n_match, n_mismatch = 0, 0
    for fname in common_files:
        print(f"File: {fname}")
        ok = compare_file(os.path.join(DIR_A, fname), os.path.join(DIR_B, fname))
        print("  => MATCH\n" if ok else "  => MISMATCH\n")
        n_match += ok
        n_mismatch += not ok

    print(f"SUMMARY: {n_match} matching, {n_mismatch} mismatching, {len(common_files)} total")


if __name__ == "__main__":
    main()


# PLOT FROM YOUR OUTPUTS THE EQUALISED OUTPUT VERSUS NOT EQUALISED VERSUS THE TRUTH

import matplotlib.pyplot as plt

DIR_INPUT = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\input"
DIR_NOTEQUAL = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\outputBoris_notequal"
DIR_EQUAL = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\outputBoris"

# Set this if the variable name inside the files is known, to skip auto-detection.
VAR_NAME = None


def load_data(path):
    ds = nc.Dataset(path, "r")
    var_name = VAR_NAME
    if var_name is None:
        # pick the largest numeric variable
        var_name = max(
            ds.variables,
            key=lambda n: ds.variables[n].size if np.issubdtype(ds.variables[n].dtype, np.number) else -1,
        )
    data = np.array(ds.variables[var_name][:], dtype=float)
    ds.close()

    while data.ndim > 1:  # reduce to a 1D profile for plotting
        data = data.mean(axis=0)
    return data


def plot_band(band):
    true_path = os.path.join(DIR_INPUT, f"ism_toa_isrf_VNIR-{band}.nc")
    notequal_path = os.path.join(DIR_NOTEQUAL, f"l1b_toa_VNIR-{band}.nc")
    equal_path = os.path.join(DIR_EQUAL, f"l1b_toa_VNIR-{band}.nc")

    plt.figure(figsize=(10, 6))
    plt.plot(load_data(true_path), label="True", color="blue")
    plt.plot(load_data(notequal_path), label="Non-equalized", color="red")
    plt.plot(load_data(equal_path), label="Equalized", color="black")

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"VNIR-{band}: True vs Non-equalized vs Equalized")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    for band in range(4):
        plot_band(band)
    plt.show()  # shows all 4 figures at once


if __name__ == "__main__":
    main()



# TRUTH = EODP-TS-L1B\input\ism_toa_isrf_VNIR-0.nc