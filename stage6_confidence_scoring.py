"""
Stage 6 — Confidence Scoring and Result Assembly
FM Broadcast Centre-Frequency Estimation Pipeline
"""

from __future__ import annotations

import logging
import time

import numpy as np
from dataclasses import dataclass, field

from stage1_acquisition import AcquisitionResult
from stage2_preprocessing import PreprocessingResult
from stage5_carrier_estimation import CarrierEstimationResult

logger = logging.getLogger(__name__)


class QualityFlag:
    CLEAR               = 0b00000000
    DC_PROXIMITY        = 0b00000001
    IQ_RESIDUAL         = 0b00000010
    LOW_SNR             = 0b00000100
    GAIN_SATURATION     = 0b00001000
    FRAME_INCONSISTENT  = 0b00010000

    @staticmethod
    def decode(flags: int) -> list[str]:
        names = {
            QualityFlag.DC_PROXIMITY:       "DC_PROXIMITY",
            QualityFlag.IQ_RESIDUAL:        "IQ_RESIDUAL",
            QualityFlag.LOW_SNR:            "LOW_SNR",
            QualityFlag.GAIN_SATURATION:    "GAIN_SATURATION",
            QualityFlag.FRAME_INCONSISTENT: "FRAME_INCONSISTENT",
        }
        return [label for bit, label in names.items() if flags & bit]

    @staticmethod
    def encode(*flag_names: str) -> int:
        mapping = {
            "DC_PROXIMITY":       QualityFlag.DC_PROXIMITY,
            "IQ_RESIDUAL":        QualityFlag.IQ_RESIDUAL,
            "LOW_SNR":            QualityFlag.LOW_SNR,
            "GAIN_SATURATION":    QualityFlag.GAIN_SATURATION,
            "FRAME_INCONSISTENT": QualityFlag.FRAME_INCONSISTENT,
        }
        mask = 0
        for name in flag_names:
            if name not in mapping:
                raise ValueError(f"Unknown flag name: '{name}'")
            mask |= mapping[name]
        return mask


SNR_SATURATION_DB:           float = 30.0
SNR_ZERO_DB:                 float = 15.0
SNR_LOW_FLAG_DB:             float = 15.0

PROMINENCE_SATURATION_DB:    float = 20.0
PROMINENCE_ZERO_DB:          float = 10.0

CONSISTENCY_SATURATION_HZ:   float = 200.0
CONSISTENCY_ZERO_HZ:         float = 1000.0

CAP_PER_FLAG:                float = 0.70
CAP_ADDITIONAL_FLAG:         float = 0.10

WEIGHT_SNR:                  float = 0.40
WEIGHT_PROMINENCE:           float = 0.35
WEIGHT_CONSISTENCY:          float = 0.25

assert abs(WEIGHT_SNR + WEIGHT_PROMINENCE + WEIGHT_CONSISTENCY - 1.0) < 1e-9


def _linear_score(
    value:            float,
    sat_value:        float,
    zero_value:       float,
    higher_is_better: bool = True,
) -> float:
    if higher_is_better:
        if value >= sat_value:
            return 1.0
        if value <= zero_value:
            return 0.0
        return (value - zero_value) / (sat_value - zero_value)

    if value <= sat_value:
        return 1.0
    if value >= zero_value:
        return 0.0
    return (zero_value - value) / (zero_value - sat_value)


@dataclass
class ConfidenceBreakdown:
    snr_db:              float
    prominence_db:       float
    frame_std_hz:        float

    c_snr:               float
    c_prominence:        float
    c_consistency:       float

    c_uncapped:          float

    quality_flags:       int
    n_flags_raised:      int
    confidence_cap:      float

    confidence:          float

    def summary(self) -> str:
        flag_str = ", ".join(QualityFlag.decode(self.quality_flags)) or "—"
        return (
            f"C={self.confidence:.3f}  "
            f"(SNR={self.c_snr:.3f}×0.40  "
            f"prom={self.c_prominence:.3f}×0.35  "
            f"σ={self.c_consistency:.3f}×0.25)  "
            f"cap={self.confidence_cap:.2f}  "
            f"flags=[{flag_str}]"
        )


def compute_snr_score(snr_db: float) -> float:
    score = _linear_score(snr_db, SNR_SATURATION_DB, SNR_ZERO_DB, higher_is_better=True)
    logger.debug("c_SNR: SNR=%.1f dB -> %.3f", snr_db, score)
    return score


def compute_prominence_score(prominence_db: float) -> float:
    score = _linear_score(prominence_db, PROMINENCE_SATURATION_DB, PROMINENCE_ZERO_DB, higher_is_better=True)
    logger.debug("c_Π: prominence=%.1f dB -> %.3f", prominence_db, score)
    return score


def compute_consistency_score(frame_std_hz: float) -> float:
    score = _linear_score(frame_std_hz, CONSISTENCY_SATURATION_HZ, CONSISTENCY_ZERO_HZ, higher_is_better=False)
    logger.debug("c_σ: σ=%.1f Hz -> %.3f", frame_std_hz, score)
    return score


def build_quality_flags(
    est: CarrierEstimationResult,
    pre: PreprocessingResult,
    acq: AcquisitionResult,
) -> int:
    flags = QualityFlag.CLEAR

    if est.snr_db < SNR_LOW_FLAG_DB:
        flags |= QualityFlag.LOW_SNR
        logger.warning("LOW_SNR flag: SNR %.1f dB < %.1f dB threshold.", est.snr_db, SNR_LOW_FLAG_DB)

    if getattr(pre.iq_result, "diagnostic_tone_used", False):
        if pre.iq_result.iq_residual_flag:
            flags |= QualityFlag.IQ_RESIDUAL
            logger.warning(
                "IQ_RESIDUAL flag: diagnostic-tone mirror %.1f dBc exceeded the allowed floor.",
                pre.iq_result.mirror_power_after_db,
            )
    else:
        logger.debug("IQ_RESIDUAL not evaluated in Stage 6 because no diagnostic tone was used in Stage 2.")

    if acq.clipping_detected:
        flags |= QualityFlag.GAIN_SATURATION
        logger.warning("GAIN_SATURATION flag: ADC clipping detected in Stage 1.")

    if est.frame_inconsistency_flag:
        flags |= QualityFlag.FRAME_INCONSISTENT
        logger.warning("FRAME_INCONSISTENT flag: σ(f̂)=%.1f Hz > %.1f Hz.", est.frame_std_hz, CONSISTENCY_ZERO_HZ)

    if est.dc_proximity_flag:
        flags |= QualityFlag.DC_PROXIMITY
        logger.warning("DC_PROXIMITY flag: propagated from CarrierEstimationResult.")

    active = QualityFlag.decode(flags)
    if active:
        logger.info("Quality flags raised: %s", ", ".join(active))
    else:
        logger.debug("Quality flags: all clear.")

    return flags


def compute_confidence_cap(n_flags: int) -> float:
    if n_flags == 0:
        return 1.0

    cap = CAP_PER_FLAG - (n_flags - 1) * CAP_ADDITIONAL_FLAG
    cap = max(cap, 0.10)
    logger.debug("Confidence cap: n_flags=%d -> %.2f", n_flags, cap)
    return cap


@dataclass
class CarrierFrequencyResult:
    """
    Final compliance record produced by the full estimation pipeline.

    Canonical schema is now region-centric rather than spike-centric:

      - region_prominence_db   replaces legacy spike_prominence_db
      - occupied_bandwidth_hz  replaces legacy spike_width_3db_hz
      - edge_bandwidth_hz      replaces legacy spike_width_6db_hz

    Rationale:
      The detector was refactored from local-peak/spike logic to occupied-region
      / channel-centric logic. The new names better describe the measured
      quantities.

    Backward compatibility:
      Legacy read-only alias properties are provided so older downstream code can
      still access:
          spike_prominence_db
          spike_width_3db_hz
          spike_width_6db_hz
    """
    estimated_frequency_hz:     float
    deviation_from_target_hz:   float
    region_prominence_db:       float
    occupied_bandwidth_hz:      float
    edge_bandwidth_hz:          float

    confidence:                 float
    snr_db:                     float
    noise_floor_dbm_hz:         float
    estimation_residual_hz:     float

    confidence_breakdown:       ConfidenceBreakdown

    dc_guard_band_hz:           float
    iq_amplitude_imbalance_db:  float
    iq_phase_imbalance_deg:     float
    mirror_suppression_db:      float

    quality_flags:              int
    quality_flag_names:         list[str] = field(default_factory=list)

    primary_estimation_method:  str = ""
    method_edge_midpoint_hz:    float = 0.0
    method_power_centroid_hz:   float = 0.0
    method_phase_refinement_hz: float | None = None
    frame_std_hz:               float = 0.0
    n_averaging_frames:         int = 0

    timestamp_utc:              str = ""
    observation_window_s:       float = 0.0
    fft_size:                   int = 0
    averaging_frames:           int = 0
    centre_frequency_hz:        int = 0
    target_frequency_hz:        int = 0
    offset_correction_hz:       int = 0
    calibration_state_id:       str = ""
    config_id:                  str = ""
    receiver_gain_db:           float = 0.0

    # ------------------------------------------------------------------
    # Legacy compatibility aliases (read-only)
    # ------------------------------------------------------------------
    @property
    def spike_prominence_db(self) -> float:
        return self.region_prominence_db

    @property
    def spike_width_3db_hz(self) -> float:
        return self.occupied_bandwidth_hz

    @property
    def spike_width_6db_hz(self) -> float:
        return self.edge_bandwidth_hz

    @property
    def passes_snr_gate(self) -> bool:
        return self.snr_db >= SNR_LOW_FLAG_DB

    @property
    def flag_count(self) -> int:
        return int(self.quality_flags).bit_count()

    def get_flag_count(self) -> int:
        return self.flag_count

    @property
    def requires_review(self) -> bool:
        return self.quality_flags != QualityFlag.CLEAR

    def report(self) -> str:
        sep = "─" * 68
        mirror_text = f"{self.mirror_suppression_db:.1f} dB" if np.isfinite(self.mirror_suppression_db) else "N/A"

        lines = [
            sep,
            "  CarrierFrequencyResult",
            sep,
            f"  Estimated frequency  : {self.estimated_frequency_hz / 1e6:.9f} MHz",
            f"  Deviation (target)   : {self.deviation_from_target_hz:+.2f} Hz"
            f"  ({self.deviation_from_target_hz / 1e3:+.4f} kHz)",
            f"  Region prominence    : {self.region_prominence_db:.1f} dB  (legacy: spike_prominence_db)",
            f"  Occupied bandwidth   : {self.occupied_bandwidth_hz / 1e3:.2f} kHz  (legacy: spike_width_3db_hz)",
            f"  Edge bandwidth       : {self.edge_bandwidth_hz / 1e3:.2f} kHz  (legacy: spike_width_6db_hz)",
            sep,
            f"  Confidence           : {self.confidence:.4f}",
            f"  {self.confidence_breakdown.summary()}",
            f"  SNR                  : {self.snr_db:.1f} dB",
            f"  Noise floor          : {self.noise_floor_dbm_hz:.1f} dBm/Hz",
            f"  Est. residual        : ±{self.estimation_residual_hz:.1f} Hz",
            sep,
            f"  DC guard band        : {self.dc_guard_band_hz / 1e3:.2f} kHz",
            f"  IQ amplitude imbal.  : {self.iq_amplitude_imbalance_db:+.2f} dB",
            f"  IQ phase imbalance   : {self.iq_phase_imbalance_deg:+.2f}°",
            f"  Mirror suppression   : {mirror_text}",
            sep,
            f"  Quality flags        : 0x{self.quality_flags:02X}  [{', '.join(self.quality_flag_names) or 'CLEAR'}]",
            f"  Requires review      : {self.requires_review}",
            sep,
            f"  Primary method       : {self.primary_estimation_method}",
            f"  Frame σ(f̂)          : {self.frame_std_hz:.2f} Hz  ({self.n_averaging_frames} frames)",
            sep,
            f"  Target               : {self.target_frequency_hz / 1e6:.4f} MHz",
            f"  LO (offset-tuned)    : {self.centre_frequency_hz / 1e6:.4f} MHz",
            f"  Timestamp            : {self.timestamp_utc}",
            f"  Calibration ID       : {self.calibration_state_id}",
            f"  FFT size / frames    : {self.fft_size} / {self.averaging_frames}",
            f"  Receiver gain        : {self.receiver_gain_db:.0f} dB",
            sep,
        ]
        return "\n".join(lines)


def run_stage6(
    est_result:          CarrierEstimationResult,
    pre_result:          PreprocessingResult,
    acq_result:          AcquisitionResult,
    spec_fft_size:       int = 0,
    spec_n_frames:       int = 0,
    config_id:           str = "default",
) -> CarrierFrequencyResult:
    t0 = time.perf_counter()
    cfg = acq_result.config

    logger.info(
        "Stage 6: SNR=%.1f dB  prominence=%.1f dB  σ_frame=%.1f Hz",
        est_result.snr_db,
        est_result.region_prominence_db,
        est_result.frame_std_hz,
    )

    c_snr  = compute_snr_score(est_result.snr_db)
    c_prom = compute_prominence_score(est_result.region_prominence_db)
    c_cons = compute_consistency_score(est_result.frame_std_hz)

    c_uncapped = WEIGHT_SNR * c_snr + WEIGHT_PROMINENCE * c_prom + WEIGHT_CONSISTENCY * c_cons

    flags = build_quality_flags(
        est = est_result,
        pre = pre_result,
        acq = acq_result,
    )
    flag_names = QualityFlag.decode(flags)
    n_flags = bin(flags).count("1")
    cap = compute_confidence_cap(n_flags)
    confidence = float(np.clip(min(c_uncapped, cap), 0.0, 1.0))

    breakdown = ConfidenceBreakdown(
        snr_db         = est_result.snr_db,
        prominence_db  = est_result.region_prominence_db,
        frame_std_hz   = est_result.frame_std_hz,
        c_snr          = c_snr,
        c_prominence   = c_prom,
        c_consistency  = c_cons,
        c_uncapped     = c_uncapped,
        quality_flags  = flags,
        n_flags_raised = n_flags,
        confidence_cap = cap,
        confidence     = confidence,
    )

    if cfg.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive in AcquisitionConfig")

    if spec_fft_size > 0 and spec_n_frames > 0:
        n_samples_used = spec_fft_size * spec_n_frames
    else:
        n_samples_used = len(acq_result.samples)

    obs_window_s = n_samples_used / cfg.sample_rate_hz
    elapsed = time.perf_counter() - t0

    logger.info(
        "Stage 6 complete: C=%.4f (uncapped=%.4f  cap=%.2f)  flags=0x%02X [%s]  elapsed=%.2f ms",
        confidence,
        c_uncapped,
        cap,
        flags,
        ", ".join(flag_names) or "CLEAR",
        elapsed * 1e3,
    )

    return CarrierFrequencyResult(
        estimated_frequency_hz      = est_result.estimated_frequency_hz,
        deviation_from_target_hz    = est_result.deviation_from_target_hz,
        region_prominence_db        = est_result.region_prominence_db,
        occupied_bandwidth_hz       = est_result.occupied_bandwidth_hz,
        edge_bandwidth_hz           = est_result.edge_bandwidth_hz,

        confidence                  = confidence,
        snr_db                      = est_result.snr_db,
        noise_floor_dbm_hz          = est_result.noise_floor_at_peak_dbm_hz,
        estimation_residual_hz      = est_result.estimation_residual_hz,

        confidence_breakdown        = breakdown,

        dc_guard_band_hz            = pre_result.dc_result.dc_guard_band_hz,
        iq_amplitude_imbalance_db   = pre_result.iq_result.amplitude_imbalance_db,
        iq_phase_imbalance_deg      = pre_result.iq_result.phase_imbalance_deg,
        mirror_suppression_db       = pre_result.iq_result.mirror_suppression_db,

        quality_flags               = flags,
        quality_flag_names          = flag_names,

        primary_estimation_method   = est_result.primary_estimation_method,
        method_edge_midpoint_hz     = est_result.method_edge_midpoint_hz,
        method_power_centroid_hz    = est_result.method_power_centroid_hz,
        method_phase_refinement_hz  = est_result.method_phase_refinement_hz,
        frame_std_hz                = est_result.frame_std_hz,
        n_averaging_frames          = len(est_result.frame_frequency_estimates_hz),

        timestamp_utc               = acq_result.timestamp_utc,
        observation_window_s        = obs_window_s,
        fft_size                    = spec_fft_size,
        averaging_frames            = spec_n_frames,
        centre_frequency_hz         = cfg.centre_frequency_hz,
        target_frequency_hz         = cfg.target_frequency_hz,
        offset_correction_hz        = cfg.offset_frequency_hz,
        calibration_state_id        = est_result.calibration_state_id,
        config_id                   = config_id,
        receiver_gain_db            = float(cfg.total_gain_db),
    )
