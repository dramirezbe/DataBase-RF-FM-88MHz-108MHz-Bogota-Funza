"""
Stage 2 — Preprocessing
FM Broadcast Centre-Frequency Estimation Pipeline
"""

from __future__ import annotations

import datetime
import logging
import uuid

import numpy as np
from scipy.signal import windows as sig_windows

from dataclasses import dataclass
from typing import Optional

from stage1_acquisition import AcquisitionResult, AcquisitionConfig

logger = logging.getLogger(__name__)

DC_GUARD_BINS_DEFAULT:     int   = 5
DC_INTERP_CONTEXT_BINS:    int   = 4

IQ_MIRROR_REJECTION_FLOOR: float = -40.0
IQ_REF_TONE_OFFSET_HZ:     float = 1_000_000.0

NOISE_PERCENTILE_DEFAULT:  float = 10.0
NOISE_FRAME_COUNT_DEFAULT: int   = 16
FFT_SIZE_NOISE:            int   = 4096


@dataclass
class IQCalibration:
    amplitude_imbalance_factor: float = 1.0
    phase_error_rad:            float = 0.0
    calibration_timestamp_utc:  str   = "uncalibrated"
    calibration_state_id:       str   = "default"

    @property
    def amplitude_imbalance_db(self) -> float:
        return 20.0 * np.log10(max(self.amplitude_imbalance_factor, 1e-12))

    @property
    def phase_error_deg(self) -> float:
        return float(np.degrees(self.phase_error_rad))

    def build_correction_matrix(self) -> np.ndarray:
        phi = float(self.phase_error_rad)
        g   = float(max(self.amplitude_imbalance_factor, 1e-12))
        return np.array([
            [1.0,           0.0],
            [-np.sin(phi),  np.cos(phi) / g],
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "amplitude_imbalance_factor": self.amplitude_imbalance_factor,
            "phase_error_rad":            self.phase_error_rad,
            "calibration_timestamp_utc":  self.calibration_timestamp_utc,
            "calibration_state_id":       self.calibration_state_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IQCalibration":
        return cls(**d)


def estimate_iq_calibration(
    samples:            np.ndarray,
    sample_rate_hz:     int,
    ref_tone_offset_hz: float = IQ_REF_TONE_OFFSET_HZ,
    fft_size:           int   = 65536,
    window_type:        str   = "blackmanharris",
) -> IQCalibration:
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")

    n = min(len(samples), fft_size)
    if n < 16:
        raise ValueError("not enough samples for calibration estimation")

    window = sig_windows.get_window(window_type, n, fftbins=True)
    window = window / np.sqrt(np.mean(window ** 2))

    spectrum = np.fft.fftshift(np.fft.fft(samples[:n] * window, n=n))

    bin_spacing = sample_rate_hz / n
    ref_bin     = int(round( ref_tone_offset_hz / bin_spacing)) + n // 2
    mirror_bin  = int(round(-ref_tone_offset_hz / bin_spacing)) + n // 2

    if not (0 <= ref_bin < n and 0 <= mirror_bin < n):
        raise ValueError("ref_tone_offset_hz lies outside the FFT span")

    ref_val    = spectrum[ref_bin]
    mirror_val = spectrum[mirror_bin]

    g   = float(np.abs(ref_val) / max(np.abs(mirror_val), 1e-12))
    phi = float(np.angle(mirror_val) - np.angle(ref_val) + np.pi) / 2.0

    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    logger.info(
        "I/Q calibration estimated from reference tone: g=%.4f (%.2f dB)  φ=%.4f rad (%.2f°)",
        g, 20 * np.log10(max(g, 1e-12)), phi, np.degrees(phi),
    )

    return IQCalibration(
        amplitude_imbalance_factor = g,
        phase_error_rad            = phi,
        calibration_timestamp_utc  = timestamp_utc,
        calibration_state_id       = f"cal_{uuid.uuid4().hex[:12]}",
    )


@dataclass
class DCSpikeRemovalResult:
    samples_corrected:   np.ndarray
    dc_guard_band_hz:    float
    guard_bins_applied:  int
    peak_dc_power_db:    float


def remove_dc_spike(
    samples:         np.ndarray,
    sample_rate_hz:  int,
    fft_size:        int = FFT_SIZE_NOISE,
    guard_bins:      int = DC_GUARD_BINS_DEFAULT,
    context_bins:    int = DC_INTERP_CONTEXT_BINS,
) -> DCSpikeRemovalResult:
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")
    if guard_bins < 0:
        raise ValueError("guard_bins must be non-negative")
    if context_bins < 1:
        raise ValueError("context_bins must be at least 1")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")
    if 2 * (guard_bins + context_bins) + 1 >= fft_size:
        raise ValueError("guard/context region is too large for the chosen fft_size")

    logger.debug(
        "2a DC mitigation: guard_bins=±%d  context_bins=%d  fft_size=%d",
        guard_bins, context_bins, fft_size,
    )

    n_complete = len(samples) // fft_size
    corrected_blocks: list[np.ndarray] = []
    peak_dc_powers:   list[float]      = []

    dc_idx      = fft_size // 2
    lo_guard    = max(dc_idx - guard_bins, 0)
    hi_guard    = min(dc_idx + guard_bins, fft_size - 1)
    lo_ctx_s    = max(lo_guard - context_bins, 0)
    hi_ctx_e    = min(hi_guard + 1 + context_bins, fft_size)

    lo_ctx_bins = np.arange(lo_ctx_s,     lo_guard)
    hi_ctx_bins = np.arange(hi_guard + 1, hi_ctx_e)
    ctx_bins    = np.concatenate([lo_ctx_bins, hi_ctx_bins])
    fill_bins   = np.arange(lo_guard, hi_guard + 1)

    if ctx_bins.size < 2:
        raise ValueError("insufficient interpolation context around the DC guard region")

    for i in range(n_complete):
        block    = samples[i * fft_size : (i + 1) * fft_size].copy()
        spectrum = np.fft.fftshift(np.fft.fft(block))

        peak_dc_powers.append(
            float(10.0 * np.log10(max(np.abs(spectrum[dc_idx]) ** 2, 1e-30)))
        )

        ctx_vals     = spectrum[ctx_bins]
        amp_interp   = np.interp(fill_bins, ctx_bins, np.abs(ctx_vals))
        phase_interp = np.interp(fill_bins, ctx_bins, np.unwrap(np.angle(ctx_vals)))
        spectrum[fill_bins] = amp_interp * np.exp(1j * phase_interp)

        corrected_blocks.append(
            np.fft.ifft(np.fft.ifftshift(spectrum)).astype(np.complex64)
        )

    remainder = samples[n_complete * fft_size :]
    if len(remainder):
        corrected_blocks.append(remainder.astype(np.complex64, copy=False))

    corrected = np.concatenate(corrected_blocks) if corrected_blocks else samples.astype(np.complex64, copy=False)
    peak_dc_power_db = float(np.mean(peak_dc_powers)) if peak_dc_powers else float("nan")
    guard_band_hz = (2 * guard_bins + 1) * sample_rate_hz / fft_size

    logger.debug(
        "DC mitigation complete: guard_band=%.1f kHz  mean_DC_bin_power=%.1f dB(rel)",
        guard_band_hz / 1e3, peak_dc_power_db,
    )

    return DCSpikeRemovalResult(
        samples_corrected  = corrected,
        dc_guard_band_hz   = guard_band_hz,
        guard_bins_applied = guard_bins,
        peak_dc_power_db   = peak_dc_power_db,
    )


@dataclass
class IQCorrectionResult:
    samples_corrected:      np.ndarray
    amplitude_imbalance_db: float
    phase_imbalance_deg:    float
    mirror_power_before_db: float
    mirror_power_after_db:  float
    mirror_suppression_db:  float
    iq_residual_flag:       bool
    diagnostic_tone_used:   bool


def correct_iq_imbalance(
    samples:              np.ndarray,
    calibration:          IQCalibration,
    sample_rate_hz:       int,
    fft_size:             int              = FFT_SIZE_NOISE,
    eval_tone_offset_hz:  Optional[float]  = None,
) -> IQCorrectionResult:
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if len(samples) == 0:
        raise ValueError("samples must be non-empty")

    logger.debug(
        "2b I/Q correction: g=%.4f (%.2f dB)  φ=%.4f rad (%.2f°)",
        calibration.amplitude_imbalance_factor,
        calibration.amplitude_imbalance_db,
        calibration.phase_error_rad,
        calibration.phase_error_deg,
    )

    C = calibration.build_correction_matrix()

    iq_matrix = np.vstack([
        samples.real.astype(np.float64),
        samples.imag.astype(np.float64),
    ])
    iq_corrected = C @ iq_matrix
    corrected    = (iq_corrected[0] + 1j * iq_corrected[1]).astype(np.complex64)

    diagnostic_tone_used = eval_tone_offset_hz is not None

    if not diagnostic_tone_used:
        logger.debug("No diagnostic tone specified for I/Q evaluation; mirror metrics set to NaN.")
        return IQCorrectionResult(
            samples_corrected      = corrected,
            amplitude_imbalance_db = calibration.amplitude_imbalance_db,
            phase_imbalance_deg    = calibration.phase_error_deg,
            mirror_power_before_db = float("nan"),
            mirror_power_after_db  = float("nan"),
            mirror_suppression_db  = float("nan"),
            iq_residual_flag       = False,
            diagnostic_tone_used   = False,
        )

    n = min(len(samples), fft_size)
    bin_spacing = sample_rate_hz / n
    mirror_bin  = int(round(-eval_tone_offset_hz / bin_spacing)) + n // 2
    ref_bin     = int(round( eval_tone_offset_hz / bin_spacing)) + n // 2

    if not (0 <= ref_bin < n and 0 <= mirror_bin < n):
        raise ValueError("eval_tone_offset_hz lies outside the FFT span for the chosen fft_size")

    def _mirror_power(s: np.ndarray) -> tuple[float, float]:
        spec     = np.fft.fftshift(np.fft.fft(s[:n], n=n))
        ref_p    = float(np.abs(spec[ref_bin]) ** 2)
        mirror_p = float(np.abs(spec[mirror_bin]) ** 2)
        return (
            10.0 * np.log10(max(ref_p, 1e-30)),
            10.0 * np.log10(max(mirror_p, 1e-30)),
        )

    ref_before, mirror_before = _mirror_power(samples)
    ref_after,  mirror_after  = _mirror_power(corrected)

    mirror_before_dbc = mirror_before - ref_before
    mirror_after_dbc  = mirror_after  - ref_after
    suppression_db    = mirror_before_dbc - mirror_after_dbc

    iq_residual_flag = bool(mirror_after_dbc > IQ_MIRROR_REJECTION_FLOOR)

    if iq_residual_flag:
        logger.warning(
            "IQ residual remains high: mirror %.1f dBc > %.1f dBc threshold.",
            mirror_after_dbc, IQ_MIRROR_REJECTION_FLOOR,
        )
    else:
        logger.debug(
            "Diagnostic tone mirror: before=%.1f dBc  after=%.1f dBc  suppression=%.1f dB",
            mirror_before_dbc, mirror_after_dbc, suppression_db,
        )

    return IQCorrectionResult(
        samples_corrected      = corrected,
        amplitude_imbalance_db = calibration.amplitude_imbalance_db,
        phase_imbalance_deg    = calibration.phase_error_deg,
        mirror_power_before_db = mirror_before_dbc,
        mirror_power_after_db  = mirror_after_dbc,
        mirror_suppression_db  = suppression_db,
        iq_residual_flag       = iq_residual_flag,
        diagnostic_tone_used   = True,
    )


@dataclass
class NoiseFloorResult:
    noise_floor_db:      np.ndarray
    noise_floor_mean_db: float
    percentile_used:     float
    frames_used:         int
    fft_size:            int
    freq_axis_hz:        np.ndarray


def estimate_noise_floor(
    samples:              np.ndarray,
    sample_rate_hz:       int,
    centre_frequency_hz:  int,
    fft_size:             int   = FFT_SIZE_NOISE,
    percentile:           float = NOISE_PERCENTILE_DEFAULT,
    n_frames:             int   = NOISE_FRAME_COUNT_DEFAULT,
    window_type:          str   = "blackmanharris",
) -> NoiseFloorResult:
    if fft_size <= 0:
        raise ValueError("fft_size must be positive")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must lie in [0, 100]")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if len(samples) < fft_size:
        raise ValueError(
            "not enough samples for one noise-estimation frame; increase capture length or reduce fft_size"
        )

    logger.debug(
        "2c Noise-floor estimation: fft_size=%d  percentile=%.1f%%  K=%d",
        fft_size, percentile, n_frames,
    )

    window  = sig_windows.get_window(window_type, fft_size, fftbins=True)
    window /= np.sqrt(np.mean(window ** 2))

    available = len(samples) // fft_size
    k_actual  = min(n_frames, available)
    if k_actual < n_frames:
        logger.warning(
            "Only %d frames available (requested %d); reduce fft_size or increase num_samples.",
            k_actual, n_frames,
        )

    frames     = samples[: k_actual * fft_size].reshape(k_actual, fft_size)
    psd_frames = np.empty((k_actual, fft_size), dtype=np.float64)

    for m, frame in enumerate(frames):
        spectrum      = np.fft.fftshift(np.fft.fft(frame * window))
        psd_frames[m] = np.abs(spectrum) ** 2

    noise_linear = np.percentile(psd_frames, percentile, axis=0)
    noise_db     = 10.0 * np.log10(np.maximum(noise_linear, 1e-30))

    freq_axis = (
        np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
        + centre_frequency_hz
    )

    noise_mean_db = float(np.mean(noise_db))

    logger.debug(
        "Noise baseline: mean=%.1f dB  min=%.1f dB  max=%.1f dB  using %d/%d frames",
        noise_mean_db,
        float(np.min(noise_db)),
        float(np.max(noise_db)),
        k_actual,
        n_frames,
    )

    return NoiseFloorResult(
        noise_floor_db      = noise_db,
        noise_floor_mean_db = noise_mean_db,
        percentile_used     = percentile,
        frames_used         = k_actual,
        fft_size            = fft_size,
        freq_axis_hz        = freq_axis,
    )


@dataclass
class PreprocessingResult:
    samples:            np.ndarray
    dc_result:          DCSpikeRemovalResult
    iq_result:          IQCorrectionResult
    noise_result:       NoiseFloorResult
    acquisition_config: AcquisitionConfig
    timestamp_utc:      str

    @property
    def dc_guard_band_hz(self) -> float:
        return self.dc_result.dc_guard_band_hz

    @property
    def noise_floor_db(self) -> np.ndarray:
        return self.noise_result.noise_floor_db

    @property
    def noise_floor_mean_db(self) -> float:
        return self.noise_result.noise_floor_mean_db

    @property
    def iq_residual_flag(self) -> bool:
        return self.iq_result.iq_residual_flag


def run_stage2(
    acq_result:          AcquisitionResult,
    calibration:         Optional[IQCalibration] = None,
    fft_size:            int = FFT_SIZE_NOISE,
    guard_bins:          int = DC_GUARD_BINS_DEFAULT,
    noise_percentile:    float = NOISE_PERCENTILE_DEFAULT,
    noise_frames:        int = NOISE_FRAME_COUNT_DEFAULT,
    eval_tone_offset_hz: Optional[float] = None,
) -> PreprocessingResult:
    samples = acq_result.samples
    cfg     = acq_result.config

    if calibration is None:
        logger.warning("No IQCalibration provided; using identity calibration (g=1, φ=0).")
        calibration = IQCalibration()

    dc_result = remove_dc_spike(
        samples         = samples,
        sample_rate_hz  = cfg.sample_rate_hz,
        fft_size        = fft_size,
        guard_bins      = guard_bins,
    )

    iq_result = correct_iq_imbalance(
        samples             = dc_result.samples_corrected,
        calibration         = calibration,
        sample_rate_hz      = cfg.sample_rate_hz,
        fft_size            = fft_size,
        eval_tone_offset_hz = eval_tone_offset_hz,
    )

    noise_result = estimate_noise_floor(
        samples              = iq_result.samples_corrected,
        sample_rate_hz       = cfg.sample_rate_hz,
        centre_frequency_hz  = cfg.centre_frequency_hz,
        fft_size             = fft_size,
        percentile           = noise_percentile,
        n_frames             = noise_frames,
    )

    result = PreprocessingResult(
        samples            = iq_result.samples_corrected,
        dc_result          = dc_result,
        iq_result          = iq_result,
        noise_result       = noise_result,
        acquisition_config = cfg,
        timestamp_utc      = acq_result.timestamp_utc,
    )

    if result.iq_result.diagnostic_tone_used:
        logger.info(
            "Stage 2 complete: DC guard=%.1f kHz  IQ mirror=%.1f dBc→%.1f dBc (Δ%.1f dB)  "
            "Noise baseline mean=%.1f dB  IQ_RESIDUAL=%s",
            result.dc_guard_band_hz / 1e3,
            result.iq_result.mirror_power_before_db,
            result.iq_result.mirror_power_after_db,
            result.iq_result.mirror_suppression_db,
            result.noise_floor_mean_db,
            result.iq_residual_flag,
        )
    else:
        logger.info(
            "Stage 2 complete: DC guard=%.1f kHz  IQ correction applied (no diagnostic tone)  "
            "Noise baseline mean=%.1f dB",
            result.dc_guard_band_hz / 1e3,
            result.noise_floor_mean_db,
        )

    return result
