"""
Stage 5 — Channel-Centre Frequency Estimation
FM Broadcast Centre-Frequency Estimation Pipeline
"""

from __future__ import annotations

import logging
import time

import numpy as np

from dataclasses import dataclass
from typing import Optional

from stage3_spectral_estimation import SpectralEstimationResult, SpectralFrame
from stage4_peak_detection import PeakDetectionResult, PeakCandidate

logger = logging.getLogger(__name__)

INTERPOLATION_RESIDUAL_MIN_FRAC:  float = 0.05
INTERPOLATION_RESIDUAL_MAX_BINS:  float = 0.5

FRAME_CONSISTENCY_GOOD_HZ:        float = 200.0
FRAME_CONSISTENCY_FLAG_HZ:        float = 1000.0

CALIBRATION_RESIDUAL_DEFAULT_HZ:  float = 0.0

METHOD_EDGE_MIDPOINT = "edge_midpoint"
METHOD_CENTROID      = "power_centroid"
METHOD_PHASE_REFINED = "phase_refined"


@dataclass
class FrequencyCalibration:
    residual_hz:          float = CALIBRATION_RESIDUAL_DEFAULT_HZ
    uncertainty_hz:       float = 0.0
    reference_source:     str   = "uncalibrated"
    calibration_state_id: str   = "default"


@dataclass
class CarrierEstimationResult:
    estimated_frequency_hz:      float
    deviation_from_target_hz:    float
    estimation_residual_hz:      float

    region_prominence_db:        float
    occupied_bandwidth_hz:       float
    edge_bandwidth_hz:           float
    snr_db:                      float
    noise_floor_at_peak_dbm_hz:  float
    peak_power_dbm_hz:           float

    method_edge_midpoint_hz:     float
    method_power_centroid_hz:    float
    method_phase_refinement_hz:  Optional[float]
    primary_estimation_method:   str

    edge_centroid_diff_hz:       float
    edge_phase_diff_hz:          Optional[float]

    frame_frequency_estimates_hz: list[float]
    frame_std_hz:                 float
    frame_mean_hz:                float
    frame_inconsistency_flag:     bool
    dc_proximity_flag:            bool

    calibration_residual_hz:      float
    calibration_state_id:         str
    offset_correction_hz:         int
    target_frequency_hz:          int
    centre_frequency_hz:          int

    computation_time_s:           float

    def summary(self) -> str:
        method_tag = f"[{self.primary_estimation_method}]"
        flag = " FRAME_INCONSISTENT" if self.frame_inconsistency_flag else ""
        return (
            f"{self.estimated_frequency_hz / 1e6:.9f} MHz  "
            f"Δtarget={self.deviation_from_target_hz:+.1f} Hz  "
            f"σ_frame={self.frame_std_hz:.1f} Hz  "
            f"residual={self.estimation_residual_hz:.1f} Hz  "
            f"{method_tag}{flag}"
        )


def _interp_crossing(
    f0_hz: float,
    y0:    float,
    f1_hz: float,
    y1:    float,
) -> float:
    if abs(y1 - y0) < 1e-12:
        return 0.5 * (f0_hz + f1_hz)

    alpha = -y0 / (y1 - y0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return f0_hz + alpha * (f1_hz - f0_hz)


def estimate_channel_edge_midpoint(
    psd_db:         np.ndarray,
    threshold_db:   np.ndarray,
    freq_axis_hz:   np.ndarray,
    start_bin:      int,
    end_bin:        int,
) -> tuple[float, float]:
    if psd_db.shape != threshold_db.shape or psd_db.shape != freq_axis_hz.shape:
        raise ValueError("psd_db, threshold_db, and freq_axis_hz must have the same shape")

    assert np.isfinite(psd_db).all(), "psd_db contains non-finite values"
    assert np.isfinite(threshold_db).all(), "threshold_db contains non-finite values"
    assert np.isfinite(freq_axis_hz).all(), "freq_axis_hz contains non-finite values"

    df = np.diff(freq_axis_hz)
    assert np.all(df > 0), "freq_axis_hz must be strictly increasing"

    diff = psd_db - threshold_db
    n = len(diff)

    if not (0 <= start_bin <= end_bin < n):
        raise ValueError("invalid region bounds")

    if end_bin <= start_bin:
        raise ValueError("region must span at least two bins")

    # Clamp interpolation anchor indices to the valid array range.
    # A region touching bin 0 or the last bin is valid (e.g. a band-edge
    # carrier or a saturating adjacent signal that extends to the boundary);
    # we use the boundary bin rather than raising.
    li0 = max(start_bin - 1, 0)
    li1 = start_bin
    left_hz = _interp_crossing(
        float(freq_axis_hz[li0]), float(diff[li0]),
        float(freq_axis_hz[li1]), float(diff[li1]),
    )

    ri0 = end_bin
    ri1 = min(end_bin + 1, n - 1)
    right_hz = _interp_crossing(
        float(freq_axis_hz[ri0]), float(diff[ri0]),
        float(freq_axis_hz[ri1]), float(diff[ri1]),
    )

    centre_hz = 0.5 * (left_hz + right_hz)
    bandwidth_hz = right_hz - left_hz

    if bandwidth_hz <= 0.0:
        raise ValueError("interpolated edge bandwidth must be positive")

    return centre_hz, bandwidth_hz


def estimate_channel_power_centroid(
    psd_db:          np.ndarray,
    noise_floor_db:  np.ndarray,
    freq_axis_hz:    np.ndarray,
    start_bin:       int,
    end_bin:         int,
) -> float:
    if psd_db.shape != noise_floor_db.shape or psd_db.shape != freq_axis_hz.shape:
        raise ValueError("psd_db, noise_floor_db, and freq_axis_hz must have the same shape")

    assert np.isfinite(psd_db).all(), "psd_db contains non-finite values"
    assert np.isfinite(noise_floor_db).all(), "noise_floor_db contains non-finite values"
    assert np.isfinite(freq_axis_hz).all(), "freq_axis_hz contains non-finite values"

    df = np.diff(freq_axis_hz)
    assert np.all(df > 0), "freq_axis_hz must be strictly increasing"

    if not (0 <= start_bin <= end_bin < len(psd_db)):
        raise ValueError("invalid region bounds")

    if end_bin <= start_bin:
        raise ValueError("region must span at least two bins")

    sl = slice(start_bin, end_bin + 1)

    p_sig = np.maximum(
        np.power(10.0, psd_db[sl] / 10.0) - np.power(10.0, noise_floor_db[sl] / 10.0),
        0.0,
    )

    if float(np.sum(p_sig)) <= 0.0:
        return 0.5 * (float(freq_axis_hz[start_bin]) + float(freq_axis_hz[end_bin]))

    return float(np.sum(freq_axis_hz[sl] * p_sig) / np.sum(p_sig))


def per_frame_estimates(
    frames:             list[SpectralFrame],
    region_start_bin:   int,
    region_end_bin:     int,
    threshold_db:       np.ndarray,
    noise_floor_db:     np.ndarray,
    freq_axis_hz:       np.ndarray,
    fallback_hz:        float,
) -> list[float]:
    if threshold_db.shape != freq_axis_hz.shape:
        raise ValueError("threshold_db and freq_axis_hz must have the same shape")
    if noise_floor_db.shape != freq_axis_hz.shape:
        raise ValueError("noise_floor_db and freq_axis_hz must have the same shape")

    assert np.isfinite(threshold_db).all(), "threshold_db contains non-finite values"
    assert np.isfinite(noise_floor_db).all(), "noise_floor_db contains non-finite values"
    assert np.isfinite(freq_axis_hz).all(), "freq_axis_hz contains non-finite values"

    estimates: list[float] = []

    for frame in frames:
        psd_frame = frame.psd_db

        if psd_frame.shape != freq_axis_hz.shape:
            estimates.append(float(fallback_hz))
            continue

        if not np.isfinite(psd_frame).all():
            estimates.append(float(fallback_hz))
            continue

        try:
            f_edge, bw = estimate_channel_edge_midpoint(
                psd_db       = psd_frame,
                threshold_db = threshold_db,
                freq_axis_hz = freq_axis_hz,
                start_bin    = region_start_bin,
                end_bin      = region_end_bin,
            )

            if bw <= 0.0:
                raise ValueError("non-positive bandwidth")

            estimates.append(float(f_edge))
            continue
        except Exception:
            try:
                f_cent = estimate_channel_power_centroid(
                    psd_db         = psd_frame,
                    noise_floor_db = noise_floor_db,
                    freq_axis_hz   = freq_axis_hz,
                    start_bin      = region_start_bin,
                    end_bin        = region_end_bin,
                )
                estimates.append(float(f_cent))
            except Exception:
                estimates.append(float(fallback_hz))

    return estimates


def apply_frequency_correction(
    raw_estimate_hz: float,
    offset_hz:       int,
    cal:             FrequencyCalibration,
) -> float:
    corrected = raw_estimate_hz + cal.residual_hz
    logger.debug(
        "Frequency correction: raw=%.6f MHz  ε_cal=%+.1f Hz  -> %.6f MHz",
        raw_estimate_hz / 1e6,
        cal.residual_hz,
        corrected / 1e6,
    )
    _ = offset_hz
    return corrected


def run_stage5(
    peak_result:       PeakDetectionResult,
    spec_result:       SpectralEstimationResult,
    samples:           np.ndarray,
    freq_cal:          Optional[FrequencyCalibration] = None,
    enable_music:      bool = True,
    enable_pisarenko:  bool = False,
    candidate:         Optional[PeakCandidate] = None,
) -> CarrierEstimationResult:
    t0 = time.perf_counter()

    if freq_cal is None:
        freq_cal = FrequencyCalibration()
        logger.warning(
            "No FrequencyCalibration provided; ε_cal = 0 Hz. Absolute frequency accuracy is limited by oscillator stability."
        )

    peak = candidate or peak_result.primary
    if peak is None:
        raise ValueError("Stage 5 requires a valid Stage 4 candidate. No valid candidate was found.")

    # ------------------------------------------------------------------
    # Global estimator path: calibrated domain only
    #   psd_dbm_hz            : calibrated PSD [dBm/Hz]
    #   threshold_dbm_hz      : calibrated threshold [dBm/Hz]
    #   noise_floor_calibrated: calibrated noise floor [dBm/Hz]
    # ------------------------------------------------------------------
    psd_dbm_hz = spec_result.psd_dbm_hz
    freq_axis_hz = spec_result.freq_axis_hz

    assert peak_result.threshold_dbm_hz is not None, "peak_result.threshold_dbm_hz must be available"
    assert np.isfinite(psd_dbm_hz).all(), "spec_result.psd_dbm_hz contains non-finite values"
    assert np.isfinite(peak_result.threshold_dbm_hz).all(), "peak_result.threshold_dbm_hz contains non-finite values"
    assert np.isfinite(spec_result.noise_floor_calibrated).all(), "spec_result.noise_floor_calibrated contains non-finite values"
    assert np.isfinite(freq_axis_hz).all(), "spec_result.freq_axis_hz contains non-finite values"

    df = np.diff(freq_axis_hz)
    assert np.all(df > 0), "spec_result.freq_axis_hz must be strictly increasing"

    if psd_dbm_hz.shape != peak_result.threshold_dbm_hz.shape or psd_dbm_hz.shape != freq_axis_hz.shape:
        raise ValueError("Global calibrated arrays must have the same shape")

    if spec_result.noise_floor_calibrated.shape != psd_dbm_hz.shape:
        raise ValueError("spec_result.noise_floor_calibrated must match spec_result.psd_dbm_hz in shape")

    assert np.all(
        peak_result.threshold_dbm_hz >= spec_result.noise_floor_calibrated
    ), "threshold_dbm_hz must be >= noise_floor_calibrated binwise"

    n_bins = len(freq_axis_hz)
    if not (0 <= peak.region_start_bin <= peak.region_end_bin < n_bins):
        raise ValueError("Peak region bounds are outside the spectral grid")

    if peak.region_end_bin <= peak.region_start_bin:
        raise ValueError("Peak region must span at least two bins")

    if peak.region_start_bin == 0 or peak.region_end_bin >= n_bins - 1:
        logger.debug(
            "Peak region touches the spectral boundary (start_bin=%d end_bin=%d n=%d); "
            "edge interpolation will clamp to boundary bin — accuracy may be reduced.",
            peak.region_start_bin, peak.region_end_bin, n_bins,
        )

    # ------------------------------------------------------------------
    # Per-frame estimator path: relative domain only
    # ------------------------------------------------------------------
    threshold_rel_db_frame   = (spec_result.noise_floor_calibrated
                                 - spec_result.scalar_correction_db
                                + peak_result.detection_margin_db)
    noise_floor_rel_db_frame = (spec_result.noise_floor_calibrated
                                - spec_result.scalar_correction_db)

    assert np.isfinite(threshold_rel_db_frame).all(), "threshold_rel_db_frame contains non-finite values"
    assert np.isfinite(noise_floor_rel_db_frame).all(), "noise_floor_rel_db_frame contains non-finite values"

    if threshold_rel_db_frame.shape != freq_axis_hz.shape:
        raise ValueError("threshold_rel_db_frame must match freq_axis_hz in shape")
    if noise_floor_rel_db_frame.shape != freq_axis_hz.shape:
        raise ValueError("noise_floor_rel_db_frame must match freq_axis_hz in shape")

    assert np.all(
        threshold_rel_db_frame >= noise_floor_rel_db_frame
    ), "threshold_rel_db_frame must be >= noise_floor_rel_db_frame binwise"

    logger.info(
        "Stage 5: region=[%d, %d]  f_region≈%.6f MHz  SNR=%.1f dB  MUSIC(req=%s→off)  Pisarenko(req=%s→off)",
        peak.region_start_bin,
        peak.region_end_bin,
        peak.centre_frequency_hz / 1e6,
        peak.snr_db,
        enable_music,
        enable_pisarenko,
    )

    f_edge_midpoint, bw_edge = estimate_channel_edge_midpoint(
        psd_db       = psd_dbm_hz,
        threshold_db = peak_result.threshold_dbm_hz,
        freq_axis_hz = freq_axis_hz,
        start_bin    = peak.region_start_bin,
        end_bin      = peak.region_end_bin,
    )

    f_centroid = estimate_channel_power_centroid(
        psd_db         = psd_dbm_hz,
        noise_floor_db = spec_result.noise_floor_calibrated,
        freq_axis_hz   = freq_axis_hz,
        start_bin      = peak.region_start_bin,
        end_bin        = peak.region_end_bin,
    )

    f_phase = None

    residual_hz = max(
        abs(f_edge_midpoint - f_centroid),
        INTERPOLATION_RESIDUAL_MIN_FRAC * spec_result.bin_spacing_hz,
    )
    residual_hz = min(
        residual_hz,
        INTERPOLATION_RESIDUAL_MAX_BINS * spec_result.bin_spacing_hz,
    )

    primary_raw_hz = f_edge_midpoint
    primary_method = METHOD_EDGE_MIDPOINT

    if enable_music or enable_pisarenko:
        logger.debug("Phase-domain refinements requested but disabled in the FM broadcast path.")

    frame_ests: list[float] = []
    frame_std  = 0.0
    frame_mean = primary_raw_hz

    if spec_result.frames:
        frame_ests = per_frame_estimates(
            frames           = spec_result.frames,
            region_start_bin = peak.region_start_bin,
            region_end_bin   = peak.region_end_bin,
            threshold_db     = threshold_rel_db_frame,
            noise_floor_db   = noise_floor_rel_db_frame,
            freq_axis_hz     = freq_axis_hz,
            fallback_hz      = primary_raw_hz,
        )

        if len(frame_ests) >= 2:
            frame_std  = float(np.std(frame_ests))
            frame_mean = float(np.mean(frame_ests))
        elif len(frame_ests) == 1:
            frame_std  = 0.0
            frame_mean = float(frame_ests[0])

    inconsistent = frame_std > FRAME_CONSISTENCY_FLAG_HZ

    if inconsistent:
        logger.warning(
            "FRAME_INCONSISTENT: σ(f̂)=%.1f Hz > %.1f Hz threshold. This may indicate unstable channel boundaries, "
            "interference, or poor SNR.",
            frame_std,
            FRAME_CONSISTENCY_FLAG_HZ,
        )

    f_calibrated = apply_frequency_correction(
        raw_estimate_hz = primary_raw_hz,
        offset_hz       = spec_result.offset_correction_hz,
        cal             = freq_cal,
    )

    deviation_hz = f_calibrated - spec_result.target_frequency_hz

    edge_centroid_diff = abs(f_edge_midpoint - f_centroid)
    edge_phase_diff = None

    elapsed = time.perf_counter() - t0

    logger.info(
        "Stage 5 complete: f̂=%.9f MHz  Δtarget=%+.1f Hz  method=%s  edgeBW=%.1f kHz  "
        "σ_frame=%.1f Hz  ε_cal=%+.1f Hz  elapsed=%.1f ms",
        f_calibrated / 1e6,
        deviation_hz,
        primary_method,
        bw_edge / 1e3,
        frame_std,
        freq_cal.residual_hz,
        elapsed * 1e3,
    )

    return CarrierEstimationResult(
        estimated_frequency_hz       = f_calibrated,
        deviation_from_target_hz     = deviation_hz,
        estimation_residual_hz       = residual_hz,
        region_prominence_db         = peak.region_prominence_db,
        occupied_bandwidth_hz        = peak.occupied_bandwidth_hz,
        edge_bandwidth_hz            = bw_edge,
        snr_db                       = peak.snr_db,
        noise_floor_at_peak_dbm_hz   = peak.noise_floor_at_peak_dbm_hz,
        peak_power_dbm_hz            = peak.peak_power_dbm_hz,
        method_edge_midpoint_hz      = f_edge_midpoint,
        method_power_centroid_hz     = f_centroid,
        method_phase_refinement_hz   = f_phase,
        primary_estimation_method    = primary_method,
        edge_centroid_diff_hz        = edge_centroid_diff,
        edge_phase_diff_hz           = edge_phase_diff,
        frame_frequency_estimates_hz = frame_ests,
        frame_std_hz                 = frame_std,
        frame_mean_hz                = frame_mean,
        frame_inconsistency_flag     = inconsistent,
        dc_proximity_flag            = peak.dc_proximity_flag,
        calibration_residual_hz      = freq_cal.residual_hz,
        calibration_state_id         = freq_cal.calibration_state_id,
        offset_correction_hz         = spec_result.offset_correction_hz,
        target_frequency_hz          = spec_result.target_frequency_hz,
        centre_frequency_hz          = spec_result.centre_frequency_hz,
        computation_time_s           = elapsed,
    )
