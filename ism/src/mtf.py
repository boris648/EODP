from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion # dummy

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        #TODO
        fstepAlt = 1 / nlines / w
        fstepAct = 1 / ncolumns / w

        eps = 1e-6
        fAlt = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAlt)
        fAct = np.arange(-1 / (2 * w), 1 / (2 * w) - eps, fstepAct)

        [fAltxx, fActxx] = np.meshgrid(fAlt, fAct, indexing='ij')
        f2D = np.sqrt(fAltxx * fAltxx + fActxx * fActxx)

        fc = D/(lambd*focal)

        fn2D = f2D/(1/w)
        fr2D = f2D/fc
        fnAct = fAct/(1/w)
        fnAlt = fAlt/(1/w)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        #TODO
        Hdiff = (2/np.pi)*(np.acos(fr2D)-fr2D*(1-fr2D**2)**(1/2))
        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        #TODO
        x = np.pi * defocus * fr2D * (1 - fr2D)
        Hdefoc = 2 * j1(x) / x
        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        #TODO
        Hwfe = np.exp(-fr2D * (1 - fr2D) * (kLF * (wLF / lambd) ** 2 + kHF * (wHF / lambd) ** 2))

        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        #TODO
        Hdet = abs(np.sinc(fn2D))
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        #TODO
        Hsmear = np.zeros((np.size(fnAlt), ncolumns))
        for n in range(ncolumns):
            Hsmear[:, n] = np.sinc(fnAlt * ksmear)
        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        #TODO
        Hmotion = np.sinc(kmotion*fn2D)
        return Hmotion


    def plotMtf(self, Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion,
                Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):

        # ---------------------------------------------------------
        # Extract ACT MTF profiles (ALT frequency = 0)
        # ---------------------------------------------------------

        row = nlines // 2
        col = ncolumns // 2

        # Include the Nyquist frequency at +0.5.
        # The original frequency array contains -0.5 but not +0.5.
        frequency = np.append(fnAct[col:], 0.5)

        def act_slice(matrix):
            return np.append(matrix[row, col:], matrix[row, 0])

        # Store profiles from all four VNIR bands
        if not hasattr(mtf, "_mtf_profiles") or band == "VNIR-0":
            mtf._mtf_profiles = {}

        mtf._mtf_profiles[band] = {
            "frequency": frequency,
            "Diffraction": act_slice(Hdiff),
            "Defocus": act_slice(Hdefoc),
            "WFE": act_slice(Hwfe),
            "Detector": act_slice(Hdet),
            "Smearing": act_slice(Hsmear),
            "Motion blur": act_slice(Hmotion),
            "System": act_slice(Hsys)
        }

        # ---------------------------------------------------------
        # Plot all four VNIR bands in a 2x2 figure
        # ---------------------------------------------------------

        styles = {
            "Diffraction": ("tab:blue", "-", 1.7),
            "Defocus": ("tab:orange", "-", 1.7),
            "WFE": ("tab:green", "-", 1.7),
            "Detector": ("tab:red", "-", 1.7),
            "Smearing": ("tab:purple", "-", 1.7),
            "Motion blur": ("tab:brown", "-", 1.7),
            "System": ("black", "-", 2.8)
        }

        fig, axes = plt.subplots(2, 2, figsize=(12, 6.8))

        legend_handles = []
        legend_labels = []

        for i, ax in enumerate(axes.flat):

            band_name = f"VNIR-{i}"
            ax.set_title(band_name, fontsize=16)

            if band_name not in mtf._mtf_profiles:
                ax.set_visible(False)
                continue

            profiles = mtf._mtf_profiles[band_name]
            freq = profiles["frequency"]

            for name, (color, linestyle, linewidth) in styles.items():

                ax.plot(
                    freq,
                    profiles[name],
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=name
                )

            if not legend_handles:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 1.04)

            ax.set_xticks(np.arange(0, 0.51, 0.1))
            ax.set_yticks(np.arange(0, 1.01, 0.25))

            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.3)

        # Shared axis labels
        fig.supxlabel(
            "Spatial frequencies f/(1/w) [-]",
            fontsize=14,
            y=0.16
        )

        fig.supylabel(
            "MTF [-]",
            fontsize=14,
            x=0.04
        )

        # Shared legend
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=12,
            frameon=False
        )

        fig.subplots_adjust(
            left=0.11,
            right=0.97,
            top=0.94,
            bottom=0.25,
            wspace=0.20,
            hspace=0.34
        )

        # Save the combined figure
        os.makedirs(directory, exist_ok=True)

        fig.savefig(
            os.path.join(directory, "mtf_ACT_all_VNIR.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        # ---------------------------------------------------------
        # Save all seven MTF matrices as NetCDF files
        # ---------------------------------------------------------

        mtf_components = {
            "Hdiff": Hdiff,
            "Hdefoc": Hdefoc,
            "Hwfe": Hwfe,
            "Hdet": Hdet,
            "Hsmear": Hsmear,
            "Hmotion": Hmotion,
            "Hsys": Hsys
        }

        for name, matrix in mtf_components.items():

            filename = f"{name}_{band}"

            writeMat(directory, filename, matrix)

            self.logger.info(f"Saved MTF: {filename}")
