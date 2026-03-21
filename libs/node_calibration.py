"""Node-calibration integration for the SDRSystemCalibration fork.

This module keeps the calibration runtime fully self-contained inside the SDR
fork. The architectural boundary is explicit:

- network access remains in :mod:`libs.data_request`;
- optional campaign materialization remains local to this module;
- the vendored ``measurement_calibration`` package owns artifact loading and
  deployment-time calibration mathematics.

The public facade exposed here intentionally covers only deployment-oriented
workflows:

1. load and calibrate campaign data returned by ``DataRequest``;
2. calibrate one in-memory real-time PSD observation; and
3. materialize SDR campaign frames into the CSV schema expected by the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import ast
import csv
import importlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd


FloatArray = np.ndarray

_DEFAULT_MODEL_DIRNAME = Path("models") / "stable"
_DEFAULT_CAMPAIGNS_DIRNAME = Path("data") / "campaigns"
_CALIBRATION_MODEL_ENV = "CALIBRATION_MODEL_DIR"
_ALLOW_FREQUENCY_EXTRAPOLATION_ENV = "CALIBRATION_ALLOW_FREQUENCY_EXTRAPOLATION"
_ALLOW_CONFIGURATION_OOD_ENV = "CALIBRATION_ALLOW_CONFIGURATION_OOD"


class CalibrationIntegrationError(RuntimeError):
    """Raised when the SDR fork cannot bridge into the calibration model."""


class CampaignParamsLike(Protocol):
    """Minimal campaign-parameter contract expected from ``DataRequest``."""

    name: str
    schedule: Any
    config: Any


class DataRequestLike(Protocol):
    """Minimal ``DataRequest`` surface required by the integration facade."""

    def get_campaign_params(
        self,
        campaign_id: int,
    ) -> CampaignParamsLike:
        """Return the parsed campaign parameters for one campaign id."""

    def load_campaigns_and_nodes(
        self,
        campaigns: Mapping[str, int],
        node_ids: Sequence[int],
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Load campaign/node measurement frames from the remote API."""


@dataclass(frozen=True)
class CalibrationServiceConfig:
    """Configuration for the SDR-to-model integration boundary.

    Parameters
    ----------
    model_dir:
        Directory that contains the calibration artifact bundle used for
        deployment-time evaluation. When ``None``, the service first checks
        ``CALIBRATION_MODEL_DIR`` and then defaults to ``models/stable``
        inside the SDR fork.
    allow_frequency_extrapolation:
        Whether deployment frequencies outside the training support should be
        evaluated with explicit boundary clamping.
    allow_configuration_ood:
        Whether configuration out-of-distribution evaluations are allowed.
    skip_unknown_sensors:
        Whether sensors absent from the artifact registry should be skipped
        instead of raising an exception.
    calibrated_column_name:
        Name used for the calibrated dB-domain PSD column added to returned
        data frames.
    variance_column_name:
        Name used for the propagated variance column added to returned data
        frames.
    """

    model_dir: Path | None = None
    allow_frequency_extrapolation: bool = False
    allow_configuration_ood: bool = True
    skip_unknown_sensors: bool = True
    calibrated_column_name: str = "calibrated_pxx"
    variance_column_name: str = "calibrated_variance_power2"

    @classmethod
    def from_environment(cls) -> CalibrationServiceConfig:
        """Build one integration config from environment variables."""

        model_dir = _path_from_env(_CALIBRATION_MODEL_ENV)
        return cls(
            model_dir=model_dir,
            allow_frequency_extrapolation=_bool_from_env(
                _ALLOW_FREQUENCY_EXTRAPOLATION_ENV,
                default=False,
            ),
            allow_configuration_ood=_bool_from_env(
                _ALLOW_CONFIGURATION_OOD_ENV,
                default=True,
            ),
        )

    def resolved_model_dir(self) -> Path:
        """Return the artifact directory used for deployment inference."""

        if self.model_dir is not None:
            return Path(self.model_dir).resolve()
        return _resolve_sdr_repo_root() / _DEFAULT_MODEL_DIRNAME


@dataclass(frozen=True)
class CampaignConfigurationSnapshot:
    """Serializable deployment configuration used by the calibration model."""

    central_frequency_hz: float
    span_hz: float
    resolution_bandwidth_hz: float
    lna_gain_db: float
    vga_gain_db: float
    acquisition_interval_s: float
    antenna_amplifier_enabled: bool

    def to_mapping(self) -> dict[str, float | bool]:
        """Return the raw configuration features as a plain mapping."""

        return {
            "central_frequency_hz": self.central_frequency_hz,
            "span_hz": self.span_hz,
            "resolution_bandwidth_hz": self.resolution_bandwidth_hz,
            "lna_gain_db": self.lna_gain_db,
            "vga_gain_db": self.vga_gain_db,
            "acquisition_interval_s": self.acquisition_interval_s,
            "antenna_amplifier_enabled": self.antenna_amplifier_enabled,
        }


@dataclass(frozen=True)
class CalibrationTrustSummary:
    """Deployment trust diagnostics normalized for the SDR fork."""

    frequency_support_hz: tuple[float, float]
    requested_frequency_hz: tuple[float, float]
    frequency_extrapolation_detected: bool
    configuration_support_available: bool
    configuration_out_of_distribution: bool
    configuration_geometry_support_available: bool
    configuration_geometric_out_of_distribution: bool
    out_of_range_feature_names: tuple[str, ...]
    max_abs_standardized_feature: float
    overall_out_of_distribution: bool
    configuration_mahalanobis_distance: float | None = None
    configuration_mahalanobis_threshold: float | None = None
    configuration_mahalanobis_tail_probability: float | None = None


@dataclass(frozen=True)
class CalibratedSensorResult:
    """Calibration output for one sensor frame collection.

    Parameters
    ----------
    sensor_id:
        Sensor label such as ``"Node1"``.
    frequency_hz:
        Frequency grid shared by every row in ``calibrated_frame`` [Hz].
    configuration:
        Deployment configuration used for the evaluation.
    calibrated_frame:
        Copy of the source frame with calibrated PSD and propagated variance
        columns appended.
    raw_power_db:
        Parsed dB-domain PSD tensor with shape ``(n_records, n_frequencies)``.
    calibrated_power_db:
        Calibrated dB-domain PSD tensor with the same shape as
        ``raw_power_db``.
    propagated_variance_power2:
        First-order propagated observation-noise variance with the same shape
        as ``raw_power_db`` in squared linear power units.
    trust_summary:
        Normalized deployment diagnostics emitted by the imported model.
    """

    sensor_id: str
    frequency_hz: FloatArray
    configuration: CampaignConfigurationSnapshot
    calibrated_frame: pd.DataFrame
    raw_power_db: FloatArray
    calibrated_power_db: FloatArray
    propagated_variance_power2: FloatArray
    trust_summary: CalibrationTrustSummary


@dataclass(frozen=True)
class CalibratedCampaignResult:
    """Calibration output for one campaign loaded through ``DataRequest``."""

    campaign_label: str
    configuration: CampaignConfigurationSnapshot
    available_sensor_ids: tuple[str, ...]
    calibrated_sensors: dict[str, CalibratedSensorResult]
    skipped_sensors: dict[str, str]
    model_dir: Path


@dataclass(frozen=True)
class MaterializedCampaignResult:
    """Filesystem payload written from SDR campaign frames for training."""

    campaign_label: str
    campaign_id: int
    output_dir: Path
    metadata_csv_path: Path
    saved_sensor_csv_paths: dict[str, Path]


@dataclass(frozen=True)
class _MeasurementCalibrationBindings:
    """Imported inference-only API surface from the vendored model package."""

    campaign_configuration_type: Any
    calibrate_sensor_observations: Any
    load_two_level_calibration_artifact: Any
    power_db_to_linear: Any
    power_linear_to_db: Any


def configuration_from_campaign_params(
    campaign_params: CampaignParamsLike,
) -> CampaignConfigurationSnapshot:
    """Translate ``DataRequest.get_campaign_params()`` into model features."""

    schedule = getattr(campaign_params, "schedule", None)
    config = getattr(campaign_params, "config", None)
    if schedule is None or config is None:
        raise ValueError(
            "campaign_params must expose .schedule and .config attributes"
        )

    central_frequency_hz = _first_finite_float(
        getattr(config, "center_freq_hz", None),
        getattr(config, "centerFrequency", None),
        field_name="center_freq_hz",
    )
    span_hz = _first_finite_float(
        getattr(config, "span", None),
        getattr(config, "sample_rate_hz", None),
        field_name="span",
    )
    rbw_hz = _parse_positive_float(
        getattr(config, "rbw", None),
        field_name="rbw",
    )
    acquisition_interval_s = _parse_positive_float(
        getattr(schedule, "interval_seconds", None),
        field_name="interval_seconds",
    )
    return CampaignConfigurationSnapshot(
        central_frequency_hz=central_frequency_hz,
        span_hz=span_hz,
        resolution_bandwidth_hz=rbw_hz,
        lna_gain_db=_parse_float(getattr(config, "lna_gain", None), "lna_gain"),
        vga_gain_db=_parse_float(getattr(config, "vga_gain", None), "vga_gain"),
        acquisition_interval_s=acquisition_interval_s,
        antenna_amplifier_enabled=bool(getattr(config, "antenna_amp", False)),
    )


class NodeCalibrationService:
    """Facade that adapts ``DataRequest`` payloads to the vendored model.

    Side Effects
    ------------
    - Imports the vendored ``measurement_calibration`` package from this fork.
    - Reads the selected calibration artifact from disk.
    - Optional materialization helpers write campaign data to disk.
    """

    def __init__(
        self,
        config: CalibrationServiceConfig | None = None,
    ) -> None:
        """Initialize the integration service and validate the import path."""

        self.config = (
            CalibrationServiceConfig.from_environment()
            if config is None
            else config
        )
        self._bindings = _load_measurement_calibration_bindings()
        self._artifact = None

    @property
    def artifact(self) -> Any:
        """Return the lazily loaded calibration artifact."""

        if self._artifact is None:
            model_dir = self.config.resolved_model_dir()
            self._artifact = self._bindings.load_two_level_calibration_artifact(model_dir)
        return self._artifact

    def calibrate_campaign_frames(
        self,
        campaign_label: str,
        node_frames: Mapping[str, pd.DataFrame],
        campaign_params: CampaignParamsLike,
    ) -> CalibratedCampaignResult:
        """Calibrate one already loaded campaign using the imported artifact."""

        configuration = configuration_from_campaign_params(campaign_params)
        model_configuration = self._build_model_configuration(configuration)
        artifact_result = self.artifact.result

        calibrated_sensors: dict[str, CalibratedSensorResult] = {}
        skipped_sensors: dict[str, str] = {}
        available_sensor_ids = tuple(sorted(str(sensor_id) for sensor_id in node_frames))
        known_sensor_ids = set(str(sensor_id) for sensor_id in artifact_result.sensor_ids)

        # Calibrate each sensor independently because the deployment model is
        # single-sensor at inference time.
        for sensor_id in available_sensor_ids:
            if sensor_id not in known_sensor_ids:
                message = (
                    f"Sensor {sensor_id!r} is absent from the artifact registry "
                    f"{tuple(artifact_result.sensor_ids)}"
                )
                if self.config.skip_unknown_sensors:
                    skipped_sensors[sensor_id] = message
                    continue
                raise ValueError(message)

            frame = node_frames[sensor_id]
            if frame.empty:
                skipped_sensors[sensor_id] = "Sensor frame is empty"
                continue

            raw_power_db, frequency_hz = _extract_sensor_power_db(frame)
            observations_power = np.asarray(
                self._bindings.power_db_to_linear(raw_power_db),
                dtype=np.float64,
            )
            deployment = self._bindings.calibrate_sensor_observations(
                result=artifact_result,
                sensor_id=sensor_id,
                configuration=model_configuration,
                frequency_hz=frequency_hz,
                observations_power=observations_power,
                allow_frequency_extrapolation=(
                    self.config.allow_frequency_extrapolation
                ),
                allow_configuration_ood=self.config.allow_configuration_ood,
            )
            calibrated_power_db = np.asarray(
                self._bindings.power_linear_to_db(deployment.calibrated_power),
                dtype=np.float64,
            )
            variance_power2 = np.asarray(
                deployment.propagated_variance_power2,
                dtype=np.float64,
            )
            trust_summary = _trust_summary_from_model(
                deployment.curves.trust_diagnostics
            )
            calibrated_frame = _build_calibrated_frame(
                frame=frame,
                calibrated_power_db=calibrated_power_db,
                propagated_variance_power2=variance_power2,
                calibrated_column_name=self.config.calibrated_column_name,
                variance_column_name=self.config.variance_column_name,
            )
            calibrated_sensors[sensor_id] = CalibratedSensorResult(
                sensor_id=sensor_id,
                frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
                configuration=configuration,
                calibrated_frame=calibrated_frame,
                raw_power_db=np.asarray(raw_power_db, dtype=np.float64),
                calibrated_power_db=calibrated_power_db,
                propagated_variance_power2=variance_power2,
                trust_summary=trust_summary,
            )

        return CalibratedCampaignResult(
            campaign_label=str(campaign_label),
            configuration=configuration,
            available_sensor_ids=available_sensor_ids,
            calibrated_sensors=calibrated_sensors,
            skipped_sensors=skipped_sensors,
            model_dir=self.config.resolved_model_dir(),
        )

    def calibrate_loaded_campaigns(
        self,
        data_request: DataRequestLike,
        campaigns: Mapping[str, int],
        node_ids: Sequence[int],
    ) -> dict[str, CalibratedCampaignResult]:
        """Load campaign frames through ``DataRequest`` and calibrate them."""

        loaded_frames = data_request.load_campaigns_and_nodes(campaigns, node_ids)
        calibrated_results: dict[str, CalibratedCampaignResult] = {}
        for campaign_label, campaign_id in campaigns.items():
            campaign_params = data_request.get_campaign_params(int(campaign_id))
            calibrated_results[campaign_label] = self.calibrate_campaign_frames(
                campaign_label=campaign_label,
                node_frames=loaded_frames.get(campaign_label, {}),
                campaign_params=campaign_params,
            )
        return calibrated_results

    def calibrate_realtime_observation(
        self,
        sensor_id: str,
        pxx: Any,
        start_freq_hz: float,
        end_freq_hz: float,
        campaign_params: CampaignParamsLike,
    ) -> CalibratedSensorResult:
        """Calibrate one in-memory PSD observation for a single sensor.

        This helper exists for the SDR fork's real-time notebook path, which
        already exposes one PSD vector plus frequency bounds from
        ``DataRequest.get_realtime_signal()``. The caller must still provide
        campaign-like configuration parameters because the imported model uses
        configuration-conditional inference.
        """

        realtime_frame = pd.DataFrame(
            [
                {
                    "pxx": pxx,
                    "start_freq_hz": float(start_freq_hz),
                    "end_freq_hz": float(end_freq_hz),
                }
            ]
        )
        campaign_result = self.calibrate_campaign_frames(
            campaign_label="realtime",
            node_frames={str(sensor_id): realtime_frame},
            campaign_params=campaign_params,
        )
        if sensor_id in campaign_result.calibrated_sensors:
            return campaign_result.calibrated_sensors[str(sensor_id)]
        skipped_reason = campaign_result.skipped_sensors.get(
            str(sensor_id),
            "Realtime observation could not be calibrated",
        )
        raise ValueError(skipped_reason)

    def materialize_campaign(
        self,
        campaign_label: str,
        campaign_id: int,
        node_frames: Mapping[str, pd.DataFrame],
        campaign_params: CampaignParamsLike,
        output_root: Path | None = None,
    ) -> MaterializedCampaignResult:
        """Persist one SDR campaign in the schema expected by the model repo."""

        output_root = (
            _resolve_sdr_repo_root() / _DEFAULT_CAMPAIGNS_DIRNAME
            if output_root is None
            else Path(output_root)
        )
        campaign_dir = Path(output_root) / str(campaign_label)
        campaign_dir.mkdir(parents=True, exist_ok=True)

        metadata_row = _build_metadata_row(
            campaign_label=campaign_label,
            campaign_id=int(campaign_id),
            campaign_params=campaign_params,
        )
        metadata_csv_path = campaign_dir / "metadata.csv"
        with metadata_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=tuple(metadata_row))
            writer.writeheader()
            writer.writerow(metadata_row)

        saved_sensor_csv_paths: dict[str, Path] = {}

        # Each per-node file preserves the original SDR frame columns while
        # forcing the PSD column to a stable JSON representation.
        for sensor_id, frame in sorted(node_frames.items()):
            if frame.empty:
                continue
            serializable_frame = frame.copy()
            serializable_frame["pxx"] = [
                json.dumps(_parse_pxx_cell(value).tolist())
                for value in serializable_frame["pxx"]
            ]
            output_path = campaign_dir / f"{sensor_id}.csv"
            serializable_frame.to_csv(output_path, index=False)
            saved_sensor_csv_paths[sensor_id] = output_path

        return MaterializedCampaignResult(
            campaign_label=str(campaign_label),
            campaign_id=int(campaign_id),
            output_dir=campaign_dir,
            metadata_csv_path=metadata_csv_path,
            saved_sensor_csv_paths=saved_sensor_csv_paths,
        )

    def materialize_loaded_campaigns(
        self,
        data_request: DataRequestLike,
        campaigns: Mapping[str, int],
        node_ids: Sequence[int],
        output_root: Path | None = None,
    ) -> dict[str, MaterializedCampaignResult]:
        """Fetch SDR campaign frames through ``DataRequest`` and persist them."""

        loaded_frames = data_request.load_campaigns_and_nodes(campaigns, node_ids)
        materialized: dict[str, MaterializedCampaignResult] = {}
        for campaign_label, campaign_id in campaigns.items():
            campaign_params = data_request.get_campaign_params(int(campaign_id))
            materialized[campaign_label] = self.materialize_campaign(
                campaign_label=campaign_label,
                campaign_id=int(campaign_id),
                node_frames=loaded_frames.get(campaign_label, {}),
                campaign_params=campaign_params,
                output_root=output_root,
            )
        return materialized

    def _build_model_configuration(
        self,
        configuration: CampaignConfigurationSnapshot,
    ) -> Any:
        """Instantiate the imported model configuration type."""

        return self._bindings.campaign_configuration_type(**configuration.to_mapping())


def _path_from_env(variable_name: str) -> Path | None:
    """Return a resolved path from one environment variable when set."""

    raw_value = os.getenv(variable_name)
    if raw_value is None or not raw_value.strip():
        return None
    return Path(raw_value).expanduser().resolve()


def _bool_from_env(variable_name: str, default: bool) -> bool:
    """Parse one boolean environment variable with a conservative fallback."""

    raw_value = os.getenv(variable_name)
    if raw_value is None:
        return bool(default)
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable {variable_name} must be boolean-like, got {raw_value!r}"
    )


def _resolve_sdr_repo_root() -> Path:
    """Return the SDR fork root from this module location."""

    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def _load_measurement_calibration_bindings(
) -> _MeasurementCalibrationBindings:
    """Import the vendored inference modules from the SDR fork only."""

    try:
        spectral_module = importlib.import_module(
            "measurement_calibration.spectral_calibration"
        )
        artifacts_module = importlib.import_module("measurement_calibration.artifacts")
    except ModuleNotFoundError as error:
        raise CalibrationIntegrationError(
            "Could not import the vendored measurement_calibration package from "
            "the SDR fork. Keep src/measurement_calibration available inside "
            "SDRSystemCalibration."
        ) from error

    return _MeasurementCalibrationBindings(
        campaign_configuration_type=getattr(
            spectral_module,
            "CampaignConfiguration",
        ),
        load_two_level_calibration_artifact=getattr(
            artifacts_module,
            "load_two_level_calibration_artifact",
        ),
        calibrate_sensor_observations=getattr(
            spectral_module,
            "calibrate_sensor_observations",
        ),
        power_db_to_linear=getattr(spectral_module, "power_db_to_linear"),
        power_linear_to_db=getattr(spectral_module, "power_linear_to_db"),
    )


def _first_finite_float(*values: Any, field_name: str) -> float:
    """Return the first finite float across multiple candidate values."""

    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed > 0.0:
            return parsed
    raise ValueError(f"Could not resolve a positive float for {field_name}")


def _parse_float(value: Any, field_name: str) -> float:
    """Parse one finite float from a field that may be numeric or string."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {parsed!r}")
    return parsed


def _parse_positive_float(value: Any, field_name: str) -> float:
    """Parse one strictly positive finite float."""

    parsed = _parse_float(value, field_name)
    if parsed <= 0.0:
        raise ValueError(f"{field_name} must be strictly positive, got {parsed!r}")
    return parsed


def _parse_pxx_cell(pxx_raw: Any) -> FloatArray:
    """Normalize one SDR ``pxx`` cell to a one-dimensional float array."""

    if isinstance(pxx_raw, np.ndarray):
        values = np.asarray(pxx_raw, dtype=np.float64).ravel()
    elif isinstance(pxx_raw, pd.Series):
        values = np.asarray(pxx_raw.to_numpy(dtype=np.float64), dtype=np.float64).ravel()
    elif isinstance(pxx_raw, (list, tuple)):
        values = np.asarray(pxx_raw, dtype=np.float64).ravel()
    else:
        raw_text = str(pxx_raw).strip()
        if not raw_text:
            raise ValueError("pxx cell is empty")
        if raw_text.startswith("[") and raw_text.endswith("]"):
            try:
                values = np.asarray(json.loads(raw_text), dtype=np.float64).ravel()
            except json.JSONDecodeError:
                values = np.asarray(ast.literal_eval(raw_text), dtype=np.float64).ravel()
        else:
            values = np.asarray(ast.literal_eval(raw_text), dtype=np.float64).ravel()

    if values.ndim != 1 or values.size < 1:
        raise ValueError("pxx cell must decode to a one-dimensional non-empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError("pxx cell contains non-finite values")
    return values


def _extract_sensor_power_db(frame: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    """Parse one sensor frame into a dB tensor and its frequency grid."""

    required_columns = {"pxx", "start_freq_hz", "end_freq_hz"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            "Sensor frame is missing required columns: "
            f"{missing_columns}"
        )

    parsed_rows: list[FloatArray] = []
    start_frequency_hz: float | None = None
    end_frequency_hz: float | None = None

    # The imported model requires a stable frequency axis for every row.
    for row_index, row in frame.reset_index(drop=True).iterrows():
        current_values = _parse_pxx_cell(row["pxx"])
        current_start_hz = _parse_float(row["start_freq_hz"], "start_freq_hz")
        current_end_hz = _parse_float(row["end_freq_hz"], "end_freq_hz")
        if current_end_hz <= current_start_hz:
            raise ValueError(
                "end_freq_hz must be strictly greater than start_freq_hz for "
                f"row {row_index}"
            )

        if start_frequency_hz is None:
            start_frequency_hz = current_start_hz
            end_frequency_hz = current_end_hz
        elif (
            not np.isclose(current_start_hz, start_frequency_hz)
            or not np.isclose(current_end_hz, end_frequency_hz)
        ):
            raise ValueError(
                "All rows in one sensor frame must share the same frequency bounds"
            )

        if parsed_rows and current_values.shape != parsed_rows[0].shape:
            raise ValueError(
                "All rows in one sensor frame must share the same pxx length"
            )
        parsed_rows.append(current_values)

    if start_frequency_hz is None or end_frequency_hz is None:
        raise ValueError("Sensor frame does not contain any usable rows")

    raw_power_db = np.stack(parsed_rows, axis=0).astype(np.float64)
    frequency_hz = np.linspace(
        start_frequency_hz,
        end_frequency_hz,
        raw_power_db.shape[1],
        dtype=np.float64,
    )
    return raw_power_db, frequency_hz


def _build_calibrated_frame(
    frame: pd.DataFrame,
    calibrated_power_db: FloatArray,
    propagated_variance_power2: FloatArray,
    calibrated_column_name: str,
    variance_column_name: str,
) -> pd.DataFrame:
    """Return a copy of one source frame with calibration columns appended."""

    calibrated_frame = frame.copy(deep=True).reset_index(drop=True)
    calibrated_frame[calibrated_column_name] = [
        row.tolist() for row in np.asarray(calibrated_power_db, dtype=np.float64)
    ]
    calibrated_frame[variance_column_name] = [
        row.tolist() for row in np.asarray(propagated_variance_power2, dtype=np.float64)
    ]
    return calibrated_frame


def _trust_summary_from_model(model_trust: Any) -> CalibrationTrustSummary:
    """Translate the imported model trust object into a local dataclass."""

    return CalibrationTrustSummary(
        frequency_support_hz=tuple(model_trust.frequency_support_hz),
        requested_frequency_hz=tuple(model_trust.requested_frequency_hz),
        frequency_extrapolation_detected=bool(
            model_trust.frequency_extrapolation_detected
        ),
        configuration_support_available=bool(
            model_trust.configuration_support_available
        ),
        configuration_out_of_distribution=bool(
            model_trust.configuration_out_of_distribution
        ),
        configuration_geometry_support_available=bool(
            model_trust.configuration_geometry_support_available
        ),
        configuration_geometric_out_of_distribution=bool(
            model_trust.configuration_geometric_out_of_distribution
        ),
        out_of_range_feature_names=tuple(model_trust.out_of_range_feature_names),
        max_abs_standardized_feature=float(model_trust.max_abs_standardized_feature),
        overall_out_of_distribution=bool(model_trust.overall_out_of_distribution),
        configuration_mahalanobis_distance=(
            None
            if model_trust.configuration_mahalanobis_distance is None
            else float(model_trust.configuration_mahalanobis_distance)
        ),
        configuration_mahalanobis_threshold=(
            None
            if model_trust.configuration_mahalanobis_threshold is None
            else float(model_trust.configuration_mahalanobis_threshold)
        ),
        configuration_mahalanobis_tail_probability=(
            None
            if model_trust.configuration_mahalanobis_tail_probability is None
            else float(model_trust.configuration_mahalanobis_tail_probability)
        ),
    )


def _build_metadata_row(
    campaign_label: str,
    campaign_id: int,
    campaign_params: CampaignParamsLike,
) -> dict[str, str | int | float | bool]:
    """Build the compatibility ``metadata.csv`` row for one SDR campaign."""

    configuration = configuration_from_campaign_params(campaign_params)
    schedule = campaign_params.schedule
    config = campaign_params.config
    acquisition_freq_minutes = configuration.acquisition_interval_s / 60.0
    return {
        "campaign_label": str(campaign_label),
        "campaign_id": int(campaign_id),
        "start_date": str(getattr(schedule, "start_date", "")),
        "stop_date": str(getattr(schedule, "end_date", "")),
        "start_time": str(getattr(schedule, "start_time", "")),
        "stop_time": str(getattr(schedule, "end_time", "")),
        "acquisition_freq_minutes": acquisition_freq_minutes,
        "central_freq_MHz": configuration.central_frequency_hz / 1.0e6,
        "span_MHz": configuration.span_hz / 1.0e6,
        "sample_rate_hz": _parse_float(
            getattr(config, "sample_rate_hz", configuration.span_hz),
            "sample_rate_hz",
        ),
        "lna_gain_dB": configuration.lna_gain_db,
        "vga_gain_dB": configuration.vga_gain_db,
        "rbw_kHz": configuration.resolution_bandwidth_hz / 1.0e3,
        "antenna_amp": configuration.antenna_amplifier_enabled,
    }


__all__ = [
    "CalibratedCampaignResult",
    "CalibratedSensorResult",
    "CalibrationIntegrationError",
    "CalibrationServiceConfig",
    "CalibrationTrustSummary",
    "CampaignConfigurationSnapshot",
    "MaterializedCampaignResult",
    "NodeCalibrationService",
    "configuration_from_campaign_params",
]
