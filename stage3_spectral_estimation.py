"""
Stage 3 — Spectral Estimation
FM Broadcast Centre-Frequency Estimation Pipeline
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.signal import windows as sig_windows

from dataclasses import dataclass
from typing import Optional

from stage1_acquisition import AcquisitionConfig
from stage2_preprocessing import PreprocessingResult, NoiseFloorResult

logger = logging.getLogger(__name__)

FFT_SIZE_DEFAULT:         int = 8192
FFT_SIZE_HIGH_RES:        int = 65536
AVERAGING_FRAMES_MIN:     int = 4
AVERAGING_FRAMES_DEFAULT: int = 16

WINDOW_DEFAULT: str = "blackmanharris"
WINDOW_ALTERNATIVES: dict[str, float] = {
    "blackmanharris": -92.0,
    "flattop":        -93.6,
    "hann":           -31.5,
    "hamming":        -43.0,
    "boxcar":         -13.3,
}

ADC_FULL_SCALE_DBM: float = -10.0
FREQ_AXIS_CORRECTION_FLOOR_HZ: float = 1.0


@dataclass
class GainCalibration:
    receiver_gain_db: float
    cable_loss_db: float = 0.0
    antenna_gain_dbi: float = 0.0
    frequency_response_correction_db: Optional[np.ndarray] = None
    calibration_state_id: str = "default"
    apply_antenna_gain_correction: bool = False

    @property
    def scalar_correction_db(self) -> float:
        antenna_term = -self.antenna_gain_dbi if self.apply_antenna_gain_correction else 0.0
        return ADC_FULL_SCALE_DBM - self.receiver_gain_db + self.cable_loss_db + antenna_term

    @classmethod
    def from_acquisition_config(
        cls,
        cfg: AcquisitionConfig,
        cable_loss_db: float = 0.0,
        antenna_gain_dbi: float = 0.0,
        apply_antenna_gain_correction: bool = False,
    ) -> "GainCalibration":
        return cls(
            receiver_gain_db              = cfg.total_gain_db,
            cable_loss_db                 = cable_loss_db,
            antenna_gain_dbi              = antenna_gain_dbi,
            calibration_state_id          = f"from_acq_{cfg.total_gain_db}dB",
            apply_antenna_gain_correction = apply_antenna_gain_correction,
        )


@dataclass
class WindowSpec:
    name:                 str
    fft_size:             int
    coefficients:         np.ndarray
    window_power_sum:     float
    power_norm_factor:    float
    amplitude_correction: float
    sidelobe_db:          float
    coherent_gain:        float
    enbw_bins:            float


def build_window(name: str, fft_size: int) -> WindowSpec:
    if name not in WINDOW_ALTERNATIVES and name not in ("bartlett", "nuttall", "parzen", "triang"):
        logger.warning("Window '%s' is not in the recommended set for FM monitoring.", name)

    w = sig_windows.get_window(name, fft_size, fftbins=True).astype(np.float64)

    window_power_sum = float(np.sum(w ** 2))
    power_norm_factor = 1.0 / max(window_power_sum, 1e-30)
    amp_corr = 1.0 / max(float(np.mean(np.abs(w))), 1e-30)
    coherent_gain = float(np.sum(w)) / fft_size
    sidelobe = WINDOW_ALTERNATIVES.get(name, float("nan"))
    enbw_bins = fft_size * window_power_sum / max(float(np.sum(w)) ** 2, 1e-30)

    logger.debug(
        "Window '%s' N=%d: power_norm=%.6e  amp_corr=%.4f  coherent_gain=%.4f  "
        "ENBW=%.3f bins  sidelobe=%.1f dB",
        name, fft_size, power_norm_factor, amp_corr, coherent_gain, enbw_bins, sidelobe,
    )

    return WindowSpec(
        name                 = name,
        fft_size             = fft_size,
        coefficients         = w,
        window_power_sum     = window_power_sum,
        power_norm_factor    = power_norm_factor,
        amplitude_correction = amp_corr,
        sidelobe_db          = sidelobe,
        coherent_gain        = coherent_gain,
        enbw_bins            = enbw_bins,
    )


@dataclass
class SpectralFrame:
    psd_db:         np.ndarray
    frame_index:    int
    peak_bin:       int
    peak_power_db:  float


@dataclass
class SpectralEstimationResult:
    psd_dbm_hz:         np.ndarray
    freq_axis_hz:       np.ndarray

    fft_size:           int
    averaging_frames:   int
    window_name:        str
    window_spec:        WindowSpec

    gain_calibration:   GainCalibration
    scalar_correction_db: float
    freq_response_applied: bool

    target_frequency_hz:    int
    centre_frequency_hz:    int
    offset_correction_hz:   int
    bin_spacing_hz:         float

    noise_floor_db:         np.ndarray
    noise_floor_calibrated: np.ndarray

    frames:                 list[SpectralFrame]

    sample_rate_hz:         int
    peak_psd_dbm_hz:        float
    peak_frequency_hz:      float
    dynamic_range_db:       float
    computation_time_s:     float

    @property
    def bin_containing(self):
        def _find(freq_hz: float) -> int:
            return int(np.argmin(np.abs(self.freq_axis_hz - freq_hz)))
        return _find

    @property
    def psd_at(self):
        def _get(freq_hz: float) -> float:
            return float(self.psd_dbm_hz[self.bin_containing(freq_hz)])
        return _get


def build_frequency_axis(
    fft_size:             int,
    sample_rate_hz:       int,
    centre_frequency_hz:  int,
    offset_correction_hz: int,
) -> tuple[np.ndarray, float]:
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    bin_spacing = sample_rate_hz / fft_size
    baseband = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    freq_axis = baseband + centre_frequency_hz - offset_correction_hz

    if abs(offset_correction_hz) < FREQ_AXIS_CORRECTION_FLOOR_HZ:
        logger.warning(
            "offset_correction_hz=%.1f Hz is negligible; verify whether Stage 1 offset tuning "
            "was intentionally disabled.",
            offset_correction_hz,
        )

    return freq_axis.astype(np.float64), float(bin_spacing)


def welch_average(
    samples:         np.ndarray,
    sample_rate_hz:  int,
    window:          WindowSpec,
    n_frames:        int,
    collect_frames:  bool = True,
) -> tuple[np.ndarray, list[SpectralFrame]]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    N = window.fft_size
    w = window.coefficients
    denom = float(sample_rate_hz) * max(window.window_power_sum, 1e-30)

    available = len(samples) // N
    m_actual = min(n_frames, available)

    if m_actual < n_frames:
        logger.warning("Welch: only %d complete frames available (requested %d).", m_actual, n_frames)
    if m_actual < AVERAGING_FRAMES_MIN:
        raise ValueError(
            f"Welch averaging requires at least {AVERAGING_FRAMES_MIN} complete frames; only {m_actual} available."
        )

    psd_accum = np.zeros(N, dtype=np.float64)
    frame_list: list[SpectralFrame] = []

    for m in range(m_actual):
        block = samples[m * N : (m + 1) * N]
        spectrum = np.fft.fftshift(np.fft.fft(block * w, n=N))
        psd_m = (np.abs(spectrum) ** 2) / denom

        psd_accum += psd_m

        if collect_frames:
            psd_m_db = 10.0 * np.log10(np.maximum(psd_m, 1e-30))
            peak_bin = int(np.argmax(psd_m_db))
            frame_list.append(
                SpectralFrame(
                    psd_db        = psd_m_db,
                    frame_index   = m,
                    peak_bin      = peak_bin,
                    peak_power_db = float(psd_m_db[peak_bin]),
                )
            )

    psd_avg = psd_accum / m_actual
    return psd_avg, frame_list


def apply_gain_calibration(
    psd_linear: np.ndarray,
    cal:        GainCalibration,
) -> tuple[np.ndarray, float, bool]:
    psd_dbfs_hz = 10.0 * np.log10(np.maximum(psd_linear, 1e-30))
    scalar_corr_db = cal.scalar_correction_db
    psd_dbm_hz = psd_dbfs_hz + scalar_corr_db

    freq_response_applied = False
    if cal.frequency_response_correction_db is not None:
        fr = cal.frequency_response_correction_db
        if len(fr) == len(psd_dbm_hz):
            psd_dbm_hz = psd_dbm_hz + fr
            freq_response_applied = True
            logger.debug("Per-bin frequency-response correction applied.")
        else:
            logger.warning(
                "frequency_response_correction_db length %d != FFT size %d; skipping per-bin correction.",
                len(fr), len(psd_dbm_hz),
            )

    logger.debug(
        "Gain calibration: scalar_corr=%.2f dB  (ADC_FS=%.1f  Grx=%.1f  Lcable=%.1f  antenna_term=%s)",
        scalar_corr_db,
        ADC_FULL_SCALE_DBM,
        cal.receiver_gain_db,
        cal.cable_loss_db,
        "on" if cal.apply_antenna_gain_correction else "off",
    )

    return psd_dbm_hz.astype(np.float64), float(scalar_corr_db), freq_response_applied


def resample_noise_floor_relative(
    noise_result:         NoiseFloorResult,
    target_fft_size:      int,
    sample_rate_hz:       int,
    centre_frequency_hz:  int,
    offset_correction_hz: int,
) -> np.ndarray:
    src_size = noise_result.fft_size
    noise_db = noise_result.noise_floor_db

    if src_size <= 0:
        raise ValueError("noise_result.fft_size must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    if src_size == target_fft_size:
        return noise_db.copy().astype(np.float64)

    src_freq = noise_result.freq_axis_hz - offset_correction_hz
    tgt_freq, _ = build_frequency_axis(
        target_fft_size,
        sample_rate_hz,
        centre_frequency_hz,
        offset_correction_hz,
    )

    resampled_db = np.interp(
        tgt_freq,
        src_freq,
        noise_db,
        left=noise_db[0],
        right=noise_db[-1],
    )
    return resampled_db.astype(np.float64)


def calibrate_relative_noise_floor(
    noise_floor_relative_db: np.ndarray,
    sample_rate_hz:          int,
    stage2_fft_size:         int,
    scalar_correction_db:    float,
) -> np.ndarray:
    stage2_density_offset_db = -10.0 * np.log10(float(sample_rate_hz) * float(stage2_fft_size))
    calibrated = noise_floor_relative_db + stage2_density_offset_db + scalar_correction_db
    return calibrated.astype(np.float64)


def validate_spectral_consistency(
    frames:      list[SpectralFrame],
    target_bin:  int,
    bin_window:  int = 20,
) -> tuple[float, float]:
    if not frames:
        raise ValueError("frames must be non-empty")

    lo = max(target_bin - bin_window, 0)
    hi = min(target_bin + bin_window + 1, len(frames[0].psd_db))

    per_frame_peaks = np.array([float(np.max(f.psd_db[lo:hi])) for f in frames], dtype=np.float64)
    return float(np.std(per_frame_peaks)), float(np.mean(per_frame_peaks))


def run_stage3(
    pre_result:        PreprocessingResult,
    gain_calibration:  Optional[GainCalibration] = None,
    fft_size:          int = FFT_SIZE_DEFAULT,
    n_frames:          int = AVERAGING_FRAMES_DEFAULT,
    window_name:       str = WINDOW_DEFAULT,
    collect_frames:    bool = True,
) -> SpectralEstimationResult:
    t0 = time.perf_counter()
    cfg = pre_result.acquisition_config

    if fft_size < FFT_SIZE_DEFAULT:
        raise ValueError(f"fft_size={fft_size} is below the specification minimum of {FFT_SIZE_DEFAULT}.")
    if fft_size & (fft_size - 1):
        raise ValueError(f"fft_size={fft_size} is not a power of two.")
    if n_frames < AVERAGING_FRAMES_MIN:
        raise ValueError(f"n_frames={n_frames} < minimum {AVERAGING_FRAMES_MIN}.")
    if window_name == "boxcar":
        raise ValueError("Boxcar window rejected: sidelobe suppression is inadequate for FM monitoring.")

    samples_needed = fft_size * n_frames
    if len(pre_result.samples) < samples_needed:
        raise ValueError(f"Insufficient samples: need {samples_needed:,}, have {len(pre_result.samples):,}.")

    if gain_calibration is None:
        logger.warning(
            "No GainCalibration provided; deriving an approximate calibration from Stage 1 gains. "
            "Absolute dBm/Hz accuracy should be treated as approximate."
        )
        gain_calibration = GainCalibration.from_acquisition_config(cfg)

    window = build_window(window_name, fft_size)

    logger.info(
        "Stage 3: FFT N=%d  M=%d  window='%s'  sidelobe=%.1f dB  ENBW=%.3f bins  bin_spacing=%.2f Hz",
        fft_size, n_frames, window_name, window.sidelobe_db, window.enbw_bins, cfg.sample_rate_hz / fft_size,
    )

    psd_linear, frame_records = welch_average(
        samples        = pre_result.samples,
        sample_rate_hz = cfg.sample_rate_hz,
        window         = window,
        n_frames       = n_frames,
        collect_frames = collect_frames,
    )

    freq_axis, bin_spacing = build_frequency_axis(
        fft_size             = fft_size,
        sample_rate_hz       = cfg.sample_rate_hz,
        centre_frequency_hz  = cfg.centre_frequency_hz,
        offset_correction_hz = cfg.offset_frequency_hz,
    )

    psd_dbm_hz, scalar_corr_db, fr_applied = apply_gain_calibration(
        psd_linear = psd_linear,
        cal        = gain_calibration,
    )

    noise_floor_rel = resample_noise_floor_relative(
        noise_result         = pre_result.noise_result,
        target_fft_size      = fft_size,
        sample_rate_hz       = cfg.sample_rate_hz,
        centre_frequency_hz  = cfg.centre_frequency_hz,
        offset_correction_hz = cfg.offset_frequency_hz,
    )

    noise_floor_cal = calibrate_relative_noise_floor(
        noise_floor_relative_db = noise_floor_rel,
        sample_rate_hz          = cfg.sample_rate_hz,
        stage2_fft_size         = pre_result.noise_result.fft_size,
        scalar_correction_db    = scalar_corr_db,
    )

    peak_bin = int(np.argmax(psd_dbm_hz))
    peak_psd = float(psd_dbm_hz[peak_bin])
    peak_freq = float(freq_axis[peak_bin])
    noise_mean = float(np.mean(noise_floor_cal))
    dynamic_range = peak_psd - noise_mean
    elapsed = time.perf_counter() - t0

    if frame_records:
        target_bin = int(np.argmin(np.abs(freq_axis - cfg.target_frequency_hz)))
        frame_peak_std_db, frame_peak_mean_db = validate_spectral_consistency(
            frame_records,
            target_bin=target_bin,
            bin_window=20,
        )
        logger.debug(
            "Frame consistency near target: mean=%.2f dBFS/Hz  std=%.2f dB",
            frame_peak_mean_db,
            frame_peak_std_db,
        )

    logger.info(
        "Stage 3 complete: dominant bin=%.1f dBm/Hz @ %.4f MHz  noise baseline=%.1f dBm/Hz  "
        "DR=%.1f dB  elapsed=%.0f ms",
        peak_psd, peak_freq / 1e6, noise_mean, dynamic_range, elapsed * 1e3,
    )

    return SpectralEstimationResult(
        psd_dbm_hz             = psd_dbm_hz,
        freq_axis_hz           = freq_axis,
        fft_size               = fft_size,
        averaging_frames       = len(frame_records) if collect_frames else n_frames,
        window_name            = window_name,
        window_spec            = window,
        gain_calibration       = gain_calibration,
        scalar_correction_db   = scalar_corr_db,
        freq_response_applied  = fr_applied,
        target_frequency_hz    = cfg.target_frequency_hz,
        centre_frequency_hz    = cfg.centre_frequency_hz,
        offset_correction_hz   = cfg.offset_frequency_hz,
        bin_spacing_hz         = bin_spacing,
        noise_floor_db         = noise_floor_rel,
        noise_floor_calibrated = noise_floor_cal,
        frames                 = frame_records,
        sample_rate_hz         = cfg.sample_rate_hz,
        peak_psd_dbm_hz        = peak_psd,
        peak_frequency_hz      = peak_freq,
        dynamic_range_db       = dynamic_range,
        computation_time_s     = elapsed,
    )
