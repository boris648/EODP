
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


# ============================================================
# CROSS VALIDATE L1B OUTPUTS EQUALIZED
# ============================================================

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
            n_diff = np.sum(
                ~np.isclose(data_a, data_b, rtol=RTOL, atol=ATOL, equal_nan=True)
            )
            print(f"  [DIFF] {var}: {n_diff} values differ")

    ds_a.close()
    ds_b.close()
    return file_ok


def main_validation():
    files_a = {f for f in os.listdir(DIR_A) if f.endswith(".nc")}
    files_b = {f for f in os.listdir(DIR_B) if f.endswith(".nc")}
    common_files = sorted(files_a & files_b)

    print("NetCDF COMPARISON REPORT")
    print(f"A: {DIR_A}")
    print(f"B: {DIR_B}\n")

    n_match, n_mismatch = 0, 0

    for fname in common_files:
        print(f"File: {fname}")
        ok = compare_file(
            os.path.join(DIR_A, fname),
            os.path.join(DIR_B, fname)
        )

        print("  => MATCH\n" if ok else "  => MISMATCH\n")
        n_match += ok
        n_mismatch += not ok

    print(
        f"SUMMARY: {n_match} matching, "
        f"{n_mismatch} mismatching, {len(common_files)} total"
    )


# ============================================================
# PLOT EQUALIZED VS NON-EQUALIZED VS TRUTH
# ============================================================

DIR_INPUT = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\input"
DIR_NOTEQUAL = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\outputBoris_notequal"
DIR_EQUAL = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B\outputBoris"

VAR_NAME = None


def load_data(path):
    ds = nc.Dataset(path, "r")
    var_name = VAR_NAME

    if var_name is None:
        # Pick the largest numeric variable
        var_name = max(
            ds.variables,
            key=lambda n: (
                ds.variables[n].size
                if np.issubdtype(ds.variables[n].dtype, np.number)
                else -1
            ),
        )

    data = np.array(ds.variables[var_name][:], dtype=float)
    ds.close()

    # Reduce to a 1D profile
    while data.ndim > 1:
        data = data.mean(axis=0)

    return data


def get_band_data(band):
    true_path = os.path.join(
        DIR_INPUT, f"ism_toa_isrf_VNIR-{band}.nc"
    )

    notequal_path = os.path.join(
        DIR_NOTEQUAL, f"l1b_toa_VNIR-{band}.nc"
    )

    equal_path = os.path.join(
        DIR_EQUAL, f"l1b_toa_VNIR-{band}.nc"
    )

    true_data = load_data(true_path)
    notequal_data = load_data(notequal_path)
    equal_data = load_data(equal_path)

    return true_data, notequal_data, equal_data


def add_lines(ax, true_data, notequal_data, equal_data):
    # Original colours, line styles and legend names
    ax.plot(true_data, label="Target", color="blue")
    ax.plot(notequal_data, label="Non-equalized", color="red")
    ax.plot(equal_data, label="Equalized", color="black")


# ============================================================
# INDIVIDUAL VNIR PLOTS
# ============================================================

def plot_band(band):
    true_data, notequal_data, equal_data = get_band_data(band)

    fig, ax = plt.subplots(figsize=(10, 6))

    add_lines(ax, true_data, notequal_data, equal_data)

    ax.set_xlabel("ACT pixel [-]", fontsize=14)
    ax.set_ylabel(
        r"TOA radiance [mW m$^{-2}$ sr$^{-1}$]",
        fontsize=14
    )

    ax.set_title(
        f"VNIR-{band}: True vs Non-equalized vs Equalized",
        fontsize=16
    )

    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = os.path.join(
        DIR_EQUAL, f"l1b_VNIR{band}.png"
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved VNIR-{band}: {output_path}")


# ============================================================
# COMBINED 2x2 VNIR FIGURE
# ============================================================

def plot_combined_figure():

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    for band in range(4):

        ax = axes[band]
        true_data, notequal_data, equal_data = get_band_data(band)

        # Exactly the same lines as the individual plots
        add_lines(ax, true_data, notequal_data, equal_data)

        ax.set_title(f"VNIR-{band}", fontsize=16)
        ax.set_xlabel("ATC Pixel [-]", fontsize=13)
        ax.set_ylabel(
            r"TOA radiance [mW m$^{-2}$ sr$^{-1}$]",
            fontsize=13
        )

        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=13,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01)
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = os.path.join(
        DIR_EQUAL, "l1b_comparison_all_VNIR.png"
    )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved combined figure: {output_path}")

def calculate_adc_saturation():

        import os
        import numpy as np
        from scipy.constants import h, c

        # ---------------------------------------------------------
        # Instrument parameters (same as ISM analytical calculation)
        # ---------------------------------------------------------

        D = 0.150
        f = 0.5262
        Tr = 0.99

        pixel_size = 30e-6
        A = pixel_size ** 2
        tint = 0.00672

        wavelengths = {
            "VNIR-0": 0.490e-6,
            "VNIR-1": 0.665e-6,
            "VNIR-2": 0.865e-6,
            "VNIR-3": 0.945e-6
        }

        QE = 0.8
        FWC = 420000
        OCF_GADC = 3.024e-6

        bit_depth = 12
        min_voltage = 0.0
        max_voltage = 0.86

        # ---------------------------------------------------------
        # Analytical conversion coefficients
        # ---------------------------------------------------------

        C_LI = Tr * (np.pi / 4) * (D / f) ** 2
        C_phe = QE
        C_eV = OCF_GADC
        C_VDN = (2 ** bit_depth - 1) / (max_voltage - min_voltage)

        DN_max = 2 ** bit_depth - 1

        # ADC electron limit
        electrons_ADC = (max_voltage - min_voltage) / OCF_GADC

        # ---------------------------------------------------------
        # Output report
        # ---------------------------------------------------------

        lines = []

        lines.append("=" * 78)
        lines.append("L1B ANALYTICAL ADC SATURATION THRESHOLDS")
        lines.append("=" * 78)
        lines.append("")
        lines.append(f"ADC bit depth:             {bit_depth} bits")
        lines.append(f"Maximum digital count:     {DN_max} DN")
        lines.append(f"ADC voltage range:         {min_voltage} - {max_voltage} V")
        lines.append(f"Detector FWC:              {FWC:,} electrons")
        lines.append(f"ADC electron limit:        {electrons_ADC:,.3f} electrons")
        lines.append(f"ADC limit / FWC:           {100 * electrons_ADC / FWC:.2f}%")
        lines.append("")

        header = (
            f"{'Band':<12}"
            f"{'DN / Radiance':>22}"
            f"{'Clipping radiance':>24}"
        )

        lines.append(header)
        lines.append("-" * len(header))

        for band, wavelength in wavelengths.items():
            # Irradiance -> Photons
            C_Iph = 1e-3 * A * tint * wavelength / (h * c)

            # Total radiance -> DN
            C_total = C_LI * C_Iph * C_phe * C_eV * C_VDN

            # Radiance corresponding to maximum ADC count
            L_clip = DN_max / C_total

            lines.append(
                f"{band:<12}"
                f"{C_total:>22.6f}"
                f"{L_clip:>24.6f}"
            )

        lines.append("-" * len(header))
        lines.append("")
        lines.append("Conversion factor: DN / (mW m^-2 sr^-1)")
        lines.append("Clipping radiance: mW m^-2 sr^-1")
        lines.append("")
        lines.append("Saturation is limited by the ADC, not the detector FWC.")
        lines.append("Thresholds refer to the nominal linear full-scale signal.")
        lines.append("=" * 78)

        report = "\n".join(lines)

        # Print in terminal
        print("\n" + report)

        # Save TXT report
        output_dir = (
            r"C:\Users\Usuario\Documents\GitHub\EODP"
            r"\EODP_TER_2021\EODP-TS-L1B\outputBoris"
        )

        os.makedirs(output_dir, exist_ok=True)

        txt_path = os.path.join(
            output_dir,
            "L1B_ADC_saturation_thresholds.txt"
        )

        with open(txt_path, "w", encoding="utf-8") as file:
            file.write(report + "\n")

        print(f"\nSaturation thresholds saved to: {txt_path}")


# =============================================================
# L1B: ARA AND EQUALIZATION-RESIDUAL (SNR PROXY) ANALYSIS
# Append to L1B_test.py, above its __main__ block.
# =============================================================

def calculate_l1b_ara_snr():
    import os
    import numpy as np
    from netCDF4 import Dataset

    # Same directories used for the L1B validation and comparison plots
    root = r"C:\Users\Usuario\Documents\GitHub\EODP\EODP_TER_2021\EODP-TS-L1B"
    eq_dir = os.path.join(root, "outputBoris")
    noneq_dir = os.path.join(root, "outputBoris_notequal")
    input_dir = os.path.join(root, "input")
    ism_dir = os.path.join(os.path.dirname(root), "EODP-TS-ISM", "outputBoris")

    DN_MAX = 4095

    def read_toa(path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required NetCDF file not found: {path}")
        with Dataset(path, "r") as ds:
            if "toa" not in ds.variables:
                raise KeyError(f"Variable 'toa' not found in {path}")
            return np.ma.asarray(ds.variables["toa"][:], dtype=np.float64).filled(np.nan)

    def find_ism_dn(band):
        filename = f"ism_toa_{band}.nc"
        candidates = [
            os.path.join(input_dir, filename),
            os.path.join(ism_dir, filename),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Cannot locate raw ISM digital counts for {band}. Tried:\n"
            + "\n".join(candidates)
        )

    results = []

    for k in range(4):
        band = f"VNIR-{k}"

        target = read_toa(os.path.join(input_dir, f"ism_toa_isrf_{band}.nc"))
        equalized = read_toa(os.path.join(eq_dir, f"l1b_toa_{band}.nc"))
        non_equalized = read_toa(os.path.join(noneq_dir, f"l1b_toa_{band}.nc"))
        raw_dn = read_toa(find_ism_dn(band))

        if not (target.shape == equalized.shape == non_equalized.shape == raw_dn.shape):
            raise ValueError(
                f"{band}: array shapes do not match: "
                f"target={target.shape}, equalized={equalized.shape}, "
                f"non_equalized={non_equalized.shape}, raw_DN={raw_dn.shape}"
            )

        # Use the full 2D images, not the plotted average profiles.
        finite = (np.isfinite(target) & np.isfinite(equalized)
                  & np.isfinite(non_equalized) & np.isfinite(raw_dn))
        valid = finite & (target > 0)
        unsaturated = valid & (raw_dn < DN_MAX)
        saturated = np.isfinite(raw_dn) & (raw_dn >= DN_MAX)

        n_valid = int(valid.sum())
        n_unsat = int(unsaturated.sum())
        n_dn = int(np.isfinite(raw_dn).sum())
        if n_valid == 0 or n_unsat < 2 or n_dn == 0:
            raise ValueError(f"{band}: insufficient valid/unsaturated pixels for statistics")

        # ARA: equalized retrieval against the original ISRF target.
        residual_all = equalized[valid] - target[valid]
        residual_unsat = equalized[unsaturated] - target[unsaturated]
        ara_all_pct = 100 * np.mean(np.abs(residual_all) / target[valid])
        ara_unsat_pct = 100 * np.mean(np.abs(residual_unsat) / target[unsaturated])
        ara_unsat_mae = np.mean(np.abs(residual_unsat))

        # SNR-LIKE PROXY: spread of the spatial correction removed by equalization.
        # This is NOT the temporal, photon-shot, or read-noise detector SNR.
        correction = non_equalized[unsaturated] - equalized[unsaturated]
        sigma_pattern = float(np.std(correction, ddof=1))
        mean_signal = float(np.mean(equalized[unsaturated]))
        snr_pattern = mean_signal / sigma_pattern if sigma_pattern > 0 else float("inf")

        clipped_pct = 100 * int(saturated.sum()) / n_dn

        results.append(dict(
            band=band, valid=n_valid, unsat=n_unsat,
            clipped_pct=clipped_pct,
            ara_unsat_mae=ara_unsat_mae, ara_unsat_pct=ara_unsat_pct,
            ara_all_pct=ara_all_pct, sigma_pattern=sigma_pattern,
            snr_pattern=snr_pattern,
        ))
    lines = [
        "=" * 108,
        "L1B RADIOMETRIC ACCURACY AND EQUALIZATION-RESIDUAL ANALYSIS",
        "=" * 108,
        "",
        (f"{'Band':<9}{'N valid':>9}{'N unsat':>10}{'Clipped %':>12}"
         f"{'ARA abs':>12}{'ARA unsat %':>14}{'ARA all %':>12}"
         f"{'Sigma corr':>14}{'SNR proxy':>12}"),
        "-" * 108,
    ]

    for r in results:
        lines.append(
            f"{r['band']:<9}{r['valid']:>9d}{r['unsat']:>10d}"
            f"{r['clipped_pct']:>12.2f}{r['ara_unsat_mae']:>12.4f}"
            f"{r['ara_unsat_pct']:>14.3f}{r['ara_all_pct']:>12.3f}"
            f"{r['sigma_pattern']:>14.4f}{r['snr_pattern']:>12.2f}"
        )

    lines += [
        "-" * 108,
        "ARA abs     : mean |equalized - target| over unsaturated valid pixels [mW m^-2 sr^-1]",
        "ARA unsat % : mean 100*|equalized - target|/target over unsaturated valid pixels",
        "ARA all %   : same relative error including saturated pixels (positive target only)",
        "Sigma corr  : sample std(non-equalized - equalized) over unsaturated pixels [mW m^-2 sr^-1]",
        "SNR proxy   : mean(equalized)/Sigma corr over unsaturated pixels, dimensionless",
        "Clipped %   : fraction of finite raw ISM DN pixels at or above 4095",
        "NOTE: SNR proxy measures equalization-correction pattern, NOT physical temporal detector SNR.",
        "NOTE: ARA also includes spatial/MTF and calibration residuals, not only gain error.",
        "=" * 108,
    ]

    report = "\n".join(lines) + "\n"
    print("\n" + report)
    os.makedirs(eq_dir, exist_ok=True)
    txt_path = os.path.join(eq_dir, "L1B_ARA_SNR_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

# ============================================================
# MAIN
# ============================================================

def main():
    # Cross-validation
    main_validation()

    # Save the four individual figures
    for band in range(4):
        plot_band(band)

    # Save the combined figure
    plot_combined_figure()

    print("All five L1B comparison figures saved successfully.")


if __name__ == "__main__":
    main()
    calculate_adc_saturation()
    calculate_l1b_ara_snr()
