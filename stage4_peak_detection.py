"""
Stage 4 — Channel Candidate Detection
FM Broadcast Centre-Frequency Estimation Pipeline

Occupied-region detection replaces local-max peak detection so the stage
tracks FM channel occupancy rather than a single spectral spike.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from dataclasses import dataclass
from typing import Optional

from stage3_spectral_estimation import SpectralEstimationResult

logger = logging.getLogger(__name__)

# 4a — threshold
DETECTION_MARGIN_DB_DEFAULT:   float = 8.0
DETECTION_MARGIN_DB_MIN:       float = 6.0
DETECTION_MARGIN_DB_MAX:       float = 20.0

# 4b/4c — occupied-region modelling
PROMINENCE_MIN_DB:             float = 10.0
PROMINENCE_FLAG_DB:            float = 20.0
SADDLE_SEARCH_HALF_WIDTH_HZ:   float = 400_000.0
REGION_GAP_MAX_HZ:             float = 15_000.0

# 4d — occupied bandwidth plausibility
WIDTH_MIN_HZ:                  float = 40_000.0
WIDTH_MAX_HZ:                  float = 400_000.0
IDEAL_WIDTH_MIN_HZ:            float = 100_000.0
IDEAL_WIDTH_MAX_HZ:            float = 250_000.0

# 4e — artefact rejection
DC_PROXIMITY_GUARD_FACTOR:     float = 2.0
MIRROR_FREQ_TOLERANCE_HZ:      float = 5_000.0
MIRROR_POWER_MARGIN_DB:        float = 6.0
MIRROR_WIDTH_RATIO_MIN:        float = 0.5

MAX_HARMONIC_ORDER:            int   = 5
MAX_CANDIDATES_RETURNED:       int   = 8


@dataclass
class PeakCandidate:
    """
    One occupied-channel candidate.

    This object is region-centric:
      - centre_frequency_hz is the coarse channel-centre estimate
      - region_prominence_db measures the region peak above the outer shoulders
      - occupied_bandwidth_hz is the raw above-threshold span of the region
    """
    bin_index:                int
    centre_frequency_hz:      float
    deviation_from_target_hz: float

    peak_power_dbm_hz:        float
    noise_floor_at_peak_dbm_hz: float
    snr_db:                   float

    region_prominence_db:     float
    shoulder_power_dbm_hz:    float
    occupied_bandwidth_hz:    float

    region_start_bin:         int = 0
    region_end_bin:           int = 0
    edge_midpoint_hz:         float = 0.0
    power_centroid_hz:        float = 0.0
    ranking_score:            float = 0.0

    dc_proximity_flag:        bool = False
    mirror_flag:              bool = False
    harmonic_flag:            bool = False
    width_out_of_range_flag:  bool = False

    mirror_of_bin:            Optional[int] = None
    harmonic_order:           Optional[int] = None

    @property
    def is_artefact(self) -> bool:
        return self.mirror_flag or self.harmonic_flag

    @property
    def is_valid_carrier(self) -> bool:
        return (
            not self.is_artefact
            and not self.dc_proximity_flag
            and not self.width_out_of_range_flag
            and self.region_prominence_db >= PROMINENCE_MIN_DB
        )

    def summary(self) -> str:
        flags = []
        if self.dc_proximity_flag:
            flags.append("DC_PROXIMITY")
        if self.mirror_flag:
            flags.append(f"MIRROR(bin={self.mirror_of_bin})")
        if self.harmonic_flag:
            flags.append(f"HARMONIC(n={self.harmonic_order})")
        if self.width_out_of_range_flag:
            flags.append("WIDTH_OOR")
        flag_str = " ".join(flags) if flags else "—"
        return (
            f"{self.centre_frequency_hz/1e6:11.6f} MHz  "
            f"SNR {self.snr_db:5.1f} dB  "
            f"prom {self.region_prominence_db:5.1f} dB  "
            f"BW {self.occupied_bandwidth_hz/1e3:6.1f} kHz  "
            f"score {self.ranking_score:4.2f}  "
            f"flags [{flag_str}]  "
            f"valid={self.is_valid_carrier}"
        )


@dataclass
class PeakDetectionResult:
    candidates:           list[PeakCandidate]

    n_raw_regions:        int
    n_artefacts:          int
    n_dc_proximity:       int
    n_width_rejected:     int
    n_valid:              int

    threshold_dbm_hz:     np.ndarray
    detection_margin_db:  float

    target_frequency_hz:  int
    computation_time_s:   float

    @property
    def primary(self) -> Optional[PeakCandidate]:
        valid = [c for c in self.candidates if c.is_valid_carrier]
        return valid[0] if valid else None

    def report(self) -> str:
        lines = [
            f"Stage 4 — {len(self.candidates)} returned candidates "
            f"({self.n_valid} valid, {self.n_artefacts} artefacts, "
            f"{self.n_dc_proximity} DC-adj, {self.n_width_rejected} width-OOR)",
            f"  Detection margin : {self.detection_margin_db:.1f} dB",
            "  " + "─" * 78,
        ]
        for i, c in enumerate(self.candidates):
            tag = "►" if (c is self.primary) else " "
            lines.append(f"  {tag} [{i:02d}] {c.summary()}")
        return "\n".join(lines)


def build_threshold(
    noise_floor_calibrated: np.ndarray,
    detection_margin_db:    float = DETECTION_MARGIN_DB_DEFAULT,
) -> np.ndarray:
    if noise_floor_calibrated.ndim != 1:
        raise ValueError("noise_floor_calibrated must be a 1D array")

    if not (DETECTION_MARGIN_DB_MIN <= detection_margin_db <= DETECTION_MARGIN_DB_MAX):
        raise ValueError(
            f"detection_margin_db={detection_margin_db:.1f} outside "
            f"[{DETECTION_MARGIN_DB_MIN}, {DETECTION_MARGIN_DB_MAX}] dB."
        )

    threshold = noise_floor_calibrated + detection_margin_db
    logger.debug(
        "Threshold: margin=%.1f dB  mean=%.1f dBm/Hz  range=[%.1f, %.1f]",
        detection_margin_db,
        float(np.mean(threshold)),
        float(np.min(threshold)),
        float(np.max(threshold)),
    )
    return threshold.astype(np.float64)


def _close_small_gaps(mask: np.ndarray, max_gap_bins: int) -> np.ndarray:
    mask = mask.astype(bool).copy()
    if max_gap_bins <= 0:
        return mask

    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue

        j = i
        while j < n and not mask[j]:
            j += 1

        gap_len = j - i
        left_true = (i > 0 and mask[i - 1])
        right_true = (j < n and mask[j])

        if left_true and right_true and gap_len <= max_gap_bins:
            mask[i:j] = True

        i = j

    return mask


def find_active_regions(
    psd_dbm_hz:      np.ndarray,
    threshold:       np.ndarray,
    *,
    max_gap_bins:    int,
    min_region_bins: int,
) -> list[tuple[int, int]]:
    if psd_dbm_hz.shape != threshold.shape:
        raise ValueError("psd_dbm_hz and threshold must have the same shape")

    mask = psd_dbm_hz > threshold
    mask = _close_small_gaps(mask, max_gap_bins=max_gap_bins)

    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends   = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1

    regions: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_region_bins:
            regions.append((int(s), int(e)))

    logger.debug(
        "Occupied-region detection: %d regions above threshold "
        "(gap_close=%d bins, min_region=%d bins)",
        len(regions), max_gap_bins, min_region_bins,
    )
    return regions


def _db_to_power(x_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, x_db / 10.0)


def assemble_channel_candidates(
    spec:                 SpectralEstimationResult,
    regions:              list[tuple[int, int]],
    shoulder_search_bins: int,
) -> list[PeakCandidate]:
    psd   = spec.psd_dbm_hz
    freq  = spec.freq_axis_hz
    noise = spec.noise_floor_calibrated

    candidates: list[PeakCandidate] = []

    for start_bin, end_bin in regions:
        sl = slice(start_bin, end_bin + 1)

        peak_bin = start_bin + int(np.argmax(psd[sl]))
        peak_power_dbm_hz = float(psd[peak_bin])

        noise_floor_at_peak_dbm_hz = float(np.median(noise[sl]))
        snr_db = peak_power_dbm_hz - noise_floor_at_peak_dbm_hz

        left_edge_hz  = float(freq[start_bin])
        right_edge_hz = float(freq[end_bin])
        occupied_bw_hz = right_edge_hz - left_edge_hz
        edge_midpoint_hz = 0.5 * (left_edge_hz + right_edge_hz)

        p_sig = np.maximum(_db_to_power(psd[sl]) - _db_to_power(noise[sl]), 0.0)
        if float(np.sum(p_sig)) > 0.0:
            power_centroid_hz = float(np.sum(freq[sl] * p_sig) / np.sum(p_sig))
        else:
            power_centroid_hz = edge_midpoint_hz

        left_ref_start = max(0, start_bin - shoulder_search_bins)
        left_ref_end   = start_bin
        right_ref_start = end_bin + 1
        right_ref_end   = min(len(psd), end_bin + 1 + shoulder_search_bins)

        left_ref  = float(np.max(psd[left_ref_start:left_ref_end])) if left_ref_end > left_ref_start else float(psd[start_bin])
        right_ref = float(np.max(psd[right_ref_start:right_ref_end])) if right_ref_end > right_ref_start else float(psd[end_bin])

        shoulder_power_dbm_hz = max(left_ref, right_ref)
        region_prominence_db = peak_power_dbm_hz - shoulder_power_dbm_hz

        width_out_of_range = not (WIDTH_MIN_HZ <= occupied_bw_hz <= WIDTH_MAX_HZ)

        candidate = PeakCandidate(
            bin_index                  = peak_bin,
            centre_frequency_hz        = edge_midpoint_hz,
            deviation_from_target_hz   = edge_midpoint_hz - spec.target_frequency_hz,
            peak_power_dbm_hz          = peak_power_dbm_hz,
            noise_floor_at_peak_dbm_hz = noise_floor_at_peak_dbm_hz,
            snr_db                     = snr_db,
            region_prominence_db       = region_prominence_db,
            shoulder_power_dbm_hz      = shoulder_power_dbm_hz,
            occupied_bandwidth_hz      = occupied_bw_hz,
            region_start_bin           = start_bin,
            region_end_bin             = end_bin,
            edge_midpoint_hz           = edge_midpoint_hz,
            power_centroid_hz          = power_centroid_hz,
            width_out_of_range_flag    = width_out_of_range,
        )

        candidates.append(candidate)

    return candidates


def flag_dc_proximity(
    freq_hz:          float,
    dc_frequency_hz:  float,
    dc_guard_band_hz: float,
    guard_factor:     float = DC_PROXIMITY_GUARD_FACTOR,
) -> bool:
    distance = abs(freq_hz - dc_frequency_hz)
    return distance < guard_factor * dc_guard_band_hz


def flag_mirror_images(
    candidates:      list[PeakCandidate],
    reflect_axis_hz: float,
    tolerance_hz:    float = MIRROR_FREQ_TOLERANCE_HZ,
    min_power_margin_db: float = MIRROR_POWER_MARGIN_DB,
) -> None:
    for i, a in enumerate(candidates):
        if a.mirror_flag:
            continue

        f_mirror = 2.0 * reflect_axis_hz - a.centre_frequency_hz

        for j, b in enumerate(candidates):
            if i == j or b.mirror_flag:
                continue

            if abs(b.centre_frequency_hz - f_mirror) > tolerance_hz:
                continue

            power_diff = a.peak_power_dbm_hz - b.peak_power_dbm_hz
            if abs(power_diff) < min_power_margin_db:
                continue

            bw_small = min(a.occupied_bandwidth_hz, b.occupied_bandwidth_hz)
            bw_large = max(a.occupied_bandwidth_hz, b.occupied_bandwidth_hz, 1e-12)
            if (bw_small / bw_large) < MIRROR_WIDTH_RATIO_MIN:
                continue

            stronger, weaker = (a, b) if power_diff > 0 else (b, a)
            weaker.mirror_flag = True
            weaker.mirror_of_bin = stronger.bin_index

            logger.debug(
                "Mirror flag: %.6f MHz marked as mirror of %.6f MHz",
                weaker.centre_frequency_hz / 1e6,
                stronger.centre_frequency_hz / 1e6,
            )


def flag_harmonics(candidates: list[PeakCandidate]) -> None:
    _ = candidates
    return


def score_channel_candidate(
    candidate: PeakCandidate,
    target_frequency_hz: int,
) -> float:
    proximity_score = float(np.exp(-abs(candidate.centre_frequency_hz - target_frequency_hz) / 150_000.0))
    snr_score = float(np.clip(candidate.snr_db / 25.0, 0.0, 1.0))
    width_score = 1.0 if IDEAL_WIDTH_MIN_HZ <= candidate.occupied_bandwidth_hz <= IDEAL_WIDTH_MAX_HZ else 0.0
    prominence_score = float(np.clip(candidate.region_prominence_db / PROMINENCE_FLAG_DB, 0.0, 1.0))

    return (
        0.45 * proximity_score
        + 0.25 * snr_score
        + 0.20 * width_score
        + 0.10 * prominence_score
    )


def deduplicate_nearby_candidates(
    candidates:         list[PeakCandidate],
    min_separation_hz:  float,
) -> list[PeakCandidate]:
    if min_separation_hz <= 0:
        return sorted(candidates, key=lambda c: c.ranking_score, reverse=True)

    kept: list[PeakCandidate] = []
    for cand in sorted(candidates, key=lambda c: c.ranking_score, reverse=True):
        if all(abs(cand.centre_frequency_hz - k.centre_frequency_hz) > min_separation_hz for k in kept):
            kept.append(cand)

    return kept


def run_stage4(
    spec_result:         SpectralEstimationResult,
    detection_margin_db: float = DETECTION_MARGIN_DB_DEFAULT,
    min_distance_hz:     float = 100_000.0,
    saddle_search_hz:    float = SADDLE_SEARCH_HALF_WIDTH_HZ,
    max_candidates:      int   = MAX_CANDIDATES_RETURNED,
    dc_guard_band_hz:    float = 30_000.0,
) -> PeakDetectionResult:
    t0 = time.perf_counter()
    psd = spec_result.psd_dbm_hz
    bin_spacing = spec_result.bin_spacing_hz

    logger.info(
        "Stage 4: detection_margin=%.1f dB  dedup_sep=%.0f kHz  "
        "shoulder_window=±%.0f kHz  max_candidates=%d",
        detection_margin_db,
        min_distance_hz / 1e3,
        saddle_search_hz / 1e3,
        max_candidates,
    )

    threshold = build_threshold(
        spec_result.noise_floor_calibrated,
        detection_margin_db,
    )

    max_gap_bins = max(1, int(round(REGION_GAP_MAX_HZ / bin_spacing)))
    min_region_bins = max(3, int(round(WIDTH_MIN_HZ / bin_spacing)))

    regions = find_active_regions(
        psd_dbm_hz      = psd,
        threshold       = threshold,
        max_gap_bins    = max_gap_bins,
        min_region_bins = min_region_bins,
    )

    n_raw = len(regions)

    if n_raw == 0:
        logger.warning("No occupied regions detected above threshold. Returning empty result.")
        return PeakDetectionResult(
            candidates          = [],
            n_raw_regions       = 0,
            n_artefacts         = 0,
            n_dc_proximity      = 0,
            n_width_rejected    = 0,
            n_valid             = 0,
            threshold_dbm_hz    = threshold,
            detection_margin_db = detection_margin_db,
            target_frequency_hz = spec_result.target_frequency_hz,
            computation_time_s  = time.perf_counter() - t0,
        )

    shoulder_bins = max(3, int(round(saddle_search_hz / bin_spacing)))

    candidates = assemble_channel_candidates(
        spec                 = spec_result,
        regions              = regions,
        shoulder_search_bins = shoulder_bins,
    )

    for c in candidates:
        c.dc_proximity_flag = flag_dc_proximity(
            freq_hz          = c.centre_frequency_hz,
            dc_frequency_hz  = float(spec_result.target_frequency_hz),
            dc_guard_band_hz = dc_guard_band_hz,
        )

    flag_mirror_images(
        candidates      = candidates,
        reflect_axis_hz = float(spec_result.target_frequency_hz),
    )

    flag_harmonics(candidates)

    for c in candidates:
        c.ranking_score = score_channel_candidate(
            candidate           = c,
            target_frequency_hz = spec_result.target_frequency_hz,
        )

    candidates = deduplicate_nearby_candidates(
        candidates        = candidates,
        min_separation_hz = min_distance_hz,
    )

    valid_cands = sorted(
        [c for c in candidates if c.is_valid_carrier],
        key=lambda c: c.ranking_score,
        reverse=True,
    )
    invalid_cands = sorted(
        [c for c in candidates if not c.is_valid_carrier],
        key=lambda c: c.ranking_score,
        reverse=True,
    )
    ranked = (valid_cands + invalid_cands)[:max_candidates]

    n_mirror   = sum(1 for c in candidates if c.mirror_flag)
    n_harmonic = sum(1 for c in candidates if c.harmonic_flag)
    n_artefact = n_mirror + n_harmonic
    n_dc       = sum(1 for c in candidates if c.dc_proximity_flag)
    n_width    = sum(1 for c in candidates if c.width_out_of_range_flag)
    n_valid    = sum(1 for c in candidates if c.is_valid_carrier)

    elapsed = time.perf_counter() - t0

    logger.info(
        "Stage 4 complete: raw_regions=%d  artefacts=%d (mirror=%d, harm=%d)  "
        "DC-adj=%d  width-OOR=%d  valid=%d  elapsed=%.1f ms",
        n_raw, n_artefact, n_mirror, n_harmonic,
        n_dc, n_width, n_valid, elapsed * 1e3,
    )

    return PeakDetectionResult(
        candidates          = ranked,
        n_raw_regions       = n_raw,
        n_artefacts         = n_artefact,
        n_dc_proximity      = n_dc,
        n_width_rejected    = n_width,
        n_valid             = n_valid,
        threshold_dbm_hz    = threshold,
        detection_margin_db = detection_margin_db,
        target_frequency_hz = spec_result.target_frequency_hz,
        computation_time_s  = elapsed,
    )
