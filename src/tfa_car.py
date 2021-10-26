from typing import List, Literal, Optional, Tuple, TypedDict

import numpy as np
import numpy.typing as npt
from scipy.fft import fft
from scipy.signal import detrend, filtfilt
from scipy.signal.windows import boxcar


def hanning_car(M: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]:
    return (1 - np.cos(2 * np.pi * (np.arange(0, M).T) / M)) / 2


class WelchCDict(TypedDict, total=False):
    Pxx: npt.NDArray[np.float_]
    Pxy: npt.NDArray[np.float_]
    Pyy: npt.NDArray[np.float_]
    coh: npt.NDArray[np.complex_]


def welch1(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    window: npt.NDArray[np.complex_],
    overlap: int,
    fs: float,
    Nfft: Optional[int] = None,
) -> Tuple[WelchCDict, npt.NDArray[np.float_], int]:
    M: int
    if len(window) == 1:
        M = window[0]
        window = boxcar(M)
    else:
        M = len(window)
    M = int(M)

    if Nfft is None:
        Nfft = M

    shift = np.round((1 - overlap) * M)
    x = x.flatten("F")
    y = y.flatten("F")
    window = window.flatten("F")
    N = len(x)

    X = fft(np.multiply(x[:M], window))
    Y = fft(np.multiply(y[:M], window))
    L = 1

    if shift > 0:
        i_start = int(shift)
        while i_start + M <= N:
            X = np.vstack(
                [X, fft(np.multiply(x[i_start : int(i_start + M)], window), Nfft)]
            )
            Y = np.vstack(
                [Y, fft(np.multiply(y[i_start : int(i_start + M)], window), Nfft)]
            )
            i_start = int(i_start + shift)
            L = L + 1

    f = np.arange(0, Nfft).T / Nfft * fs
    X = X.T
    Y = Y.T
    C: WelchCDict = {}
    if L == 1:
        C["Pxx"] = (np.multiply(X, np.conj(X))) / L / np.sum(np.power(window, 2)) / fs
        C["Pyy"] = (np.multiply(Y, np.conj(Y))) / L / np.sum(np.power(window, 2)) / fs
        C["Pxy"] = (np.multiply(np.conj(X), Y)) / L / np.sum(np.power(window, 2)) / fs
        C["coh"] = np.true_divide(
            C["Pxy"], np.power((np.abs((np.multiply(C["Pxx"], C["Pyy"])))), 0.5)
        )
    else:
        C["Pxx"] = (
            np.sum(np.multiply(X, np.conj(X)), 1) / L / np.sum(np.power(window, 2)) / fs
        )
        C["Pyy"] = (
            np.sum(np.multiply(Y, np.conj(Y)), 1) / L / np.sum(np.power(window, 2)) / fs
        )
        C["Pxy"] = (
            np.sum(np.multiply(np.conj(X), Y), 1) / L / np.sum(np.power(window, 2)) / fs
        )
        C["coh"] = np.true_divide(
            C["Pxy"], np.power((np.abs((np.multiply(C["Pxx"], C["Pyy"])))), 0.5)
        )

    return C, f, L


def tfa1(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    wind: npt.NDArray[np.complex_],
    overlap: int,
    M_smooth: int,
    fs: float,
    Nfft: Optional[int] = None,
) -> Tuple[
    npt.NDArray[np.complex_],
    npt.NDArray[np.complex_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    npt.NDArray[np.float_],
    int,
]:
    M: int
    if len(wind) == 1:
        M = wind[0]
        wind = boxcar(wind)
    else:
        M = len(wind)

    if Nfft is None:
        Nfft = 0
    if Nfft == 0:
        Nfft = M
    C, f, no_windows = welch1(x, y, wind, overlap, fs)
    Pxx = C["Pxx"]
    Pyy = C["Pyy"]
    Pxy = C["Pxy"]

    if M_smooth > 1:
        h = np.ones(int(np.floor((M_smooth + 1) / 2)))
        h = h / np.sum(h)

        Pxx1: npt.NDArray[np.float_] = np.copy(Pxx)
        Pxx1[0] = Pxx[1]
        Pyy1: npt.NDArray[np.float_] = np.copy(Pyy)
        Pyy1[0] = Pyy[1]
        Pxy1: npt.NDArray[np.float_] = np.copy(Pxy)
        Pxy1[0] = Pxy[1]

        Pxx1 = filtfilt(h.T, 1, Pxx1)
        Pyy1 = filtfilt(h.T, 1, Pyy1)
        Pxy1 = filtfilt(h.T, 1, Pxy1)

        Pxx1[0] = Pxx[0]
        Pxx = Pxx1
        Pyy1[0] = Pyy[0]
        Pyy = Pyy1
        Pxy1[0] = Pxy[0]
        Pxy = Pxy1

    H: npt.NDArray[np.complex_] = np.true_divide(Pxy, Pxx)
    C_2: npt.NDArray[np.complex_] = np.true_divide(
        Pxy, np.power(np.abs(np.multiply(Pxx, Pyy)), 0.5)
    )

    return H, C_2, f, Pxx, Pxy, Pyy, no_windows


class TFAParamDict(TypedDict, total=False):
    vlf: List[float]
    lf: List[float]
    hf: List[float]
    detrend: int
    spectral_smoothing: int
    coherence2_thresholds: npt.NDArray[np.float_]
    apply_coherence2_thresholds: bool
    remove_negative_phase: bool
    remove_negative_phase_f_cutoff: float
    normalize_ABP: bool
    normalize_CBFV: bool
    window_type: Literal["HANNING", "BOXCAR"]
    window_length: float
    overlap: float
    overlap_adjust: bool
    plot: bool
    plot_f_range: List[float]
    plot_title: str


class TFAOutDict(TypedDict, total=False):
    Mean_abp: float
    Std_abp: float
    Mean_cbfv: float
    Std_cbfv: float
    Gain_vlf: float
    Phase_vlf: float
    Coh2_vlf: float
    P_abp_vlf: float
    P_cbfv_vlf: float
    Gain_lf: float
    Phase_lf: float
    Coh2_lf: float
    P_abp_lf: float
    P_cbfv_lf: float
    Gain_hf: float
    Phase_hf: float
    Coh2_hf: float
    P_abp_hf: float
    P_cbfv_hf: float
    Gain_vlf_not_norm: float
    Gain_lf_not_norm: float
    Gain_hf_not_norm: float
    Gain_vlf_norm: float
    Gain_lf_norm: float
    Gain_hf_norm: float
    overlap: float
    H: npt.NDArray[np.complex_]
    C: npt.NDArray[np.complex_]
    f: npt.NDArray[np.float_]
    Pxx: npt.NDArray[np.float_]
    Pxy: npt.NDArray[np.float_]
    Pyy: npt.NDArray[np.float_]
    No_windows: int


def tfa_car(
    ABP: npt.NDArray[np.float_],
    CBFV: npt.NDArray[np.float_],
    fs: float,
    params: Optional[TFAParamDict] = None,
) -> TFAOutDict:
    default_params: TFAParamDict = {}
    default_params["vlf"] = [0.02, 0.07]
    default_params["lf"] = [0.07, 0.2]
    default_params["hf"] = [0.2, 0.5]
    default_params["detrend"] = 0
    default_params["spectral_smoothing"] = 3
    default_params["coherence2_thresholds"] = np.vstack(
        [
            np.arange(3, 16),
            np.array(
                [
                    0.51,
                    0.40,
                    0.34,
                    0.29,
                    0.25,
                    0.22,
                    0.20,
                    0.18,
                    0.17,
                    0.15,
                    0.14,
                    0.13,
                    0.12,
                ]
            ),
        ]
    ).T
    default_params["apply_coherence2_thresholds"] = True
    default_params["remove_negative_phase"] = True
    default_params["remove_negative_phase_f_cutoff"] = 0.1
    default_params["normalize_ABP"] = False
    default_params["normalize_CBFV"] = False
    default_params["window_type"] = "HANNING"
    default_params["window_length"] = 102.4
    default_params["overlap"] = 59.99
    default_params["overlap_adjust"] = True
    default_params["plot"] = True
    default_params["plot_f_range"] = [0, 0.5]
    default_params["plot_title"] = ""

    if params is None:
        params = default_params
    # Set defaults:
    param_list = [
        "vlf",
        "lf",
        "hf",
        "detrend",
        "spectral_smoothing",
        "coherence2_thresholds",
        "apply_coherence2_thresholds",
        "remove_negative_phase",
        "remove_negative_phase_f_cutoff",
        "normalize_ABP",
        "normalize_CBFV",
        "window_type",
        "window_length",
        "overlap",
        "overlap_adjust",
        "plot",
        "plot_f_range",
        "plot_title",
    ]
    for p in param_list:
        params[p] = params.get(p, default_params[p])
    tfa_out: TFAOutDict = {}
    tfa_out["Mean_abp"] = np.mean(ABP)
    tfa_out["Std_abp"] = np.std(ABP)
    if params["detrend"]:
        ABP = detrend(ABP)
    else:
        ABP = ABP - np.mean(ABP)
    if params["normalize_ABP"] == 1:
        ABP = np.true_divide(ABP, tfa_out["Mean_abp"]) * 100
    tfa_out["Mean_cbfv"] = np.mean(CBFV)
    tfa_out["Std_cbfv"] = np.std(CBFV)
    if params["detrend"]:
        CBFV = detrend(CBFV)
    else:
        CBFV = CBFV - np.mean(CBFV)
    if params["normalize_ABP"] == 1:
        CBFV = np.true_divide(CBFV, tfa_out["Mean_cbfv"]) * 100
    window_length = np.round(params["window_length"] * fs)
    if params["window_type"].upper() == "HANNING":
        wind = hanning_car(window_length)
    if params["window_type"].upper() == "BOXCAR":
        wind = boxcar(window_length)
    if params["overlap_adjust"] == 1:
        L = (
            np.floor(
                (len(ABP) - window_length)
                / (window_length * (1 - params["overlap"] / 100))
            )
            + 1
        )
        if L > 1:
            shift = np.floor((len(ABP) - window_length) / (L - 1))
            overlap = (window_length - shift) / window_length * 100
            tfa_out["overlap"] = overlap
    else:
        overlap = params["overlap"]
    overlap = overlap / 100
    M_smooth = params["spectral_smoothing"]
    N_fft = window_length
    H, C, f, Pxx, Pxy, Pyy, no_windows = tfa1(
        ABP, CBFV, wind, overlap, M_smooth, fs, N_fft
    )
    tfa_out["H"] = np.copy(H)
    tfa_out["C"] = C
    tfa_out["f"] = f
    tfa_out["Pxx"] = Pxx
    tfa_out["Pyy"] = Pyy
    tfa_out["Pxy"] = Pxy
    tfa_out["No_windows"] = no_windows

    i = np.argwhere(params["coherence2_thresholds"][:, 0] == no_windows)
    if len(i) == 0:
        print(
            "Warning:no coherence threshold defined for the number of windows obtained - all frequencies will be included"
        )
        coherence2_threshold = 0
    else:
        coherence2_threshold = params["coherence2_thresholds"][i, 1]

    if params["apply_coherence2_thresholds"]:
        i = np.argwhere(np.power(abs(C), 2) < coherence2_threshold)
        H[i] = np.nan
    P = np.angle(H)
    if params["remove_negative_phase"]:
        n = np.argwhere(f < params["remove_negative_phase_f_cutoff"])
        k = np.argwhere(P[n] < 0)
        if len(k) > 0:
            P[n[k]] = np.nan
    i = np.argwhere(np.logical_and((f >= params["vlf"][0]), (f < params["vlf"][1])))
    tfa_out["Gain_vlf"] = np.nanmean(abs(H[i]))
    tfa_out["Phase_vlf"] = np.nanmean(P[i]) / (2 * np.pi) * 360
    tfa_out["Coh2_vlf"] = np.nanmean(np.power(abs(C[i]), 2))
    tfa_out["P_abp_vlf"] = 2 * sum(Pxx[i]) * f[1]
    tfa_out["P_cbfv_vlf"] = 2 * sum(Pyy[i]) * f[1]

    i = np.argwhere(np.logical_and((f >= params["lf"][0]), (f < params["lf"][1])))
    tfa_out["Gain_lf"] = np.nanmean(abs(H[i]))
    tfa_out["Phase_lf"] = np.nanmean(P[i]) / (2 * np.pi) * 360
    tfa_out["Coh2_lf"] = np.nanmean(np.power(abs(C[i]), 2))
    tfa_out["P_abp_lf"] = 2 * sum(Pxx[i]) * f[1]
    tfa_out["P_cbfv_lf"] = 2 * sum(Pyy[i]) * f[1]

    i = np.argwhere(np.logical_and((f >= params["hf"][0]), (f < params["hf"][1])))
    tfa_out["Gain_hf"] = np.nanmean(abs(H[i]))
    dummy = np.angle(H[i])
    tfa_out["Phase_hf"] = np.nanmean(P[i]) / (2 * np.pi) * 360
    tfa_out["Coh2_hf"] = np.nanmean(np.power(abs(C[i]), 2))
    tfa_out["P_abp_hf"] = 2 * sum(Pxx[i]) * f[1]
    tfa_out["P_cbfv_hf"] = 2 * sum(Pyy[i]) * f[1]

    if params["normalize_CBFV"]:
        tfa_out["Gain_vlf_norm"] = tfa_out["Gain_vlf"]
        tfa_out["Gain_lf_norm"] = tfa_out["Gain_lf"]
        tfa_out["Gain_hf_norm"] = tfa_out["Gain_hf"]
        tfa_out["Gain_vlf_not_norm"] = tfa_out["Gain_vlf"] * tfa_out["Mean_cbfv"] / 100
        tfa_out["Gain_lf_not_norm"] = tfa_out["Gain_lf"] * tfa_out["Mean_cbfv"] / 100
        tfa_out["Gain_hf_not_norm"] = tfa_out["Gain_hf"] * tfa_out["Mean_cbfv"] / 100
    else:
        tfa_out["Gain_vlf_not_norm"] = tfa_out["Gain_vlf"]
        tfa_out["Gain_lf_not_norm"] = tfa_out["Gain_lf"]
        tfa_out["Gain_hf_not_norm"] = tfa_out["Gain_hf"]
        tfa_out["Gain_vlf_norm"] = tfa_out["Gain_vlf"] / tfa_out["Mean_cbfv"] * 100
        tfa_out["Gain_lf_norm"] = tfa_out["Gain_lf"] / tfa_out["Mean_cbfv"] * 100
        tfa_out["Gain_hf_norm"] = tfa_out["Gain_hf"] / tfa_out["Mean_cbfv"] * 100

    return tfa_out
