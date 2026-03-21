"""Compatibilidad CSV para resultados de ``DataRequest``.

Este módulo convierte la estructura retornada por
``DataRequest.load_campaigns_and_nodes`` al mismo formato de CSV usado en
``DataBase-RF-FM-88MHz-108MHz-Bogota-Funza/``.

La estructura de entrada esperada es:

``{campaign_label: {"Node1": pd.DataFrame, "Node2": pd.DataFrame, ...}}``

Cada CSV de salida respeta estas 16 columnas (en este orden):

``id, mac, campaign_id, pxx, start_freq_hz, end_freq_hz, timestamp, lat, lng,
excursion_peak_to_peak_hz, excursion_peak_deviation_hz,
excursion_rms_deviation_hz, depth_peak_to_peak, depth_peak_deviation,
depth_rms_deviation, created_at``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


TARGET_COLUMNS: tuple[str, ...] = (
    "id",
    "mac",
    "campaign_id",
    "pxx",
    "start_freq_hz",
    "end_freq_hz",
    "timestamp",
    "lat",
    "lng",
    "excursion_peak_to_peak_hz",
    "excursion_peak_deviation_hz",
    "excursion_rms_deviation_hz",
    "depth_peak_to_peak",
    "depth_peak_deviation",
    "depth_rms_deviation",
    "created_at",
)


DEFAULT_NODE_FILE_SUFFIX: dict[int, str] = {
    1: "Bogota",
    2: "Bogota",
    3: "Bogota",
    4: "Bogota",
    5: "Bogota",
    6: "Bogota",
    7: "Bogota",
    8: "Bogota",
    9: "Funza",
    10: "Bogota",
}


def _parse_node_id(node_key: str) -> int | None:
    match = re.search(r"(\d+)", str(node_key))
    if not match:
        return None
    return int(match.group(1))


def _to_json_array_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float64).ravel()
        return json.dumps(array.tolist(), separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=np.float64).ravel()
        return json.dumps(array.tolist(), separators=(",", ":"))
    if pd.isna(value):
        return "[]"

    text = str(value).strip()
    if not text:
        return "[]"

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            array = np.asarray(parsed, dtype=np.float64).ravel()
            return json.dumps(array.tolist(), separators=(",", ":"))
        except Exception:
            return text

    return text


def _normalize_frame(frame: pd.DataFrame, campaign_id: int | None = None) -> pd.DataFrame:
    normalized = frame.copy()

    for column in TARGET_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""

    if campaign_id is not None and "campaign_id" in normalized.columns:
        normalized["campaign_id"] = normalized["campaign_id"].replace("", campaign_id)
        normalized["campaign_id"] = normalized["campaign_id"].fillna(campaign_id)

    normalized["pxx"] = normalized["pxx"].apply(_to_json_array_text)
    normalized = normalized.loc[:, list(TARGET_COLUMNS)]
    return normalized


def save_datarequest_structure_to_csv(
    loaded_data: Mapping[str, Mapping[str, pd.DataFrame]],
    output_path: str | Path,
    *,
    campaign_id_map: Mapping[str, int] | None = None,
    node_file_suffix: Mapping[int, str] | None = None,
    split_by_campaign: bool = True,
) -> dict[str, list[Path]]:
    """Guardar la estructura de ``DataRequest`` como CSV compatibles.

    Args:
        loaded_data: Resultado de ``load_campaigns_and_nodes``.
        output_path: Directorio base de salida.
        campaign_id_map: Mapa ``{campaign_label: campaign_id}`` opcional.
        node_file_suffix: Mapa opcional para nombre de archivo por nodo.
            Por defecto usa el convenio histórico (Node9-Funza, resto Bogota).
        split_by_campaign: Si ``True`` crea subcarpetas por campaña.

    Returns:
        Diccionario ``{campaign_label: [paths_csv_guardados...]}``.
    """
    base_output = Path(output_path)
    base_output.mkdir(parents=True, exist_ok=True)

    suffix_map = dict(DEFAULT_NODE_FILE_SUFFIX)
    if node_file_suffix:
        suffix_map.update(node_file_suffix)

    saved_paths: dict[str, list[Path]] = {}

    for campaign_label, nodes_dict in loaded_data.items():
        campaign_dir = base_output / str(campaign_label) if split_by_campaign else base_output
        campaign_dir.mkdir(parents=True, exist_ok=True)

        campaign_paths: list[Path] = []
        campaign_id = campaign_id_map.get(campaign_label) if campaign_id_map else None

        for node_key, frame in sorted(nodes_dict.items()):
            if frame is None or frame.empty:
                continue

            node_id = _parse_node_id(node_key)
            suffix = suffix_map.get(node_id, "Node") if node_id is not None else "Node"

            if node_id is not None:
                filename = f"Node{node_id}-{suffix}.csv"
            else:
                filename = f"{node_key}.csv"

            normalized = _normalize_frame(frame=frame, campaign_id=campaign_id)
            output_file = campaign_dir / filename
            normalized.to_csv(output_file, index=False)
            campaign_paths.append(output_file)

        saved_paths[str(campaign_label)] = campaign_paths

    return saved_paths


class DataRequestCsvCompat:
    """Capa de compatibilidad para descargar + exportar CSV desde ``DataRequest``."""

    def __init__(self, data_request: Any):
        self.data_request = data_request

    def download_to_path(
        self,
        campaigns: Mapping[str, int],
        node_ids: list[int],
        output_path: str | Path,
        *,
        split_by_campaign: bool = True,
    ) -> dict[str, list[Path]]:
        """Descargar vía ``DataRequest`` y guardar CSV en ``output_path``."""
        loaded_data = self.data_request.load_campaigns_and_nodes(campaigns, node_ids)
        return save_datarequest_structure_to_csv(
            loaded_data=loaded_data,
            output_path=output_path,
            campaign_id_map=campaigns,
            split_by_campaign=split_by_campaign,
        )
