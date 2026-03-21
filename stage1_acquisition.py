"""
Stage 1 - IQ acquisition from SigMF database (GitHub).
This replaces direct HackRF hardware capture for notebook workflows.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

import diskcache as dc
import numpy as np
import requests

logger = logging.getLogger(__name__)

DB_BASE_URL = "https://raw.githubusercontent.com/dramirezbe/DataBase-IQ-FM-88MHz-108MHz/main/"
DB_TREE_URL = (
    "https://api.github.com/repos/dramirezbe/"
    "DataBase-IQ-FM-88MHz-108MHz/git/trees/main?recursive=1"
)

DEFAULT_CACHE_DIR = "dataset_cache"
DEFAULT_MEASURES = list(range(1, 31))


@dataclass
class IQParams:
    center_freq: int
    sample_rate: int
    lna: int
    vga: int
    amp_enabled: bool


@dataclass
class AcquisitionConfig:
    target_frequency_hz: int
    offset_frequency_hz: int
    sample_rate_hz: int
    num_samples: int
    if_gain_db: int
    baseband_gain_db: int
    amp_enable: bool
    record_number: int
    centre_frequency_hz: int = field(init=False)
    total_gain_db: int = field(init=False)

    @property
    def lna_gain_db(self) -> int:
        return self.if_gain_db

    @property
    def vga_gain_db(self) -> int:
        return self.baseband_gain_db

    def __post_init__(self) -> None:
        self.centre_frequency_hz = self.target_frequency_hz + self.offset_frequency_hz
        self.total_gain_db = self.if_gain_db + self.baseband_gain_db + (14 if self.amp_enable else 0)


@dataclass
class AcquisitionResult:
    samples: np.ndarray
    config: AcquisitionConfig
    timestamp_utc: str
    clipping_detected: bool = False


class SigMFRepo:
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        self.base_url = DB_BASE_URL
        self.cache = dc.Cache(cache_dir)
        self.tree = self._load_tree()

    def _load_tree(self) -> list[dict]:
        cache_key = "repo_tree"
        if cache_key in self.cache:
            return self.cache[cache_key]
        tree = requests.get(DB_TREE_URL, timeout=30).json().get("tree", [])
        self.cache[cache_key] = tree
        return tree

    def _find_pair(self, number: int) -> tuple[str, str]:
        prefix = f"{number:02d}"
        all_files = [item["path"] for item in self.tree if "path" in item]
        meta_matches = [
            p for p in all_files
            if p.startswith(prefix) and p.endswith(".sigmf-meta")
        ]
        if not meta_matches:
            raise FileNotFoundError(f"No SigMF metadata for measure {number} (prefix {prefix})")
        meta_file = meta_matches[0]
        data_file = meta_file.replace(".sigmf-meta", ".sigmf-data")
        return meta_file, data_file

    def get_measure(self, number: int) -> tuple[np.ndarray, IQParams]:
        meta_key = f"meta_{number:02d}"
        iq_key = f"iq_{number:02d}"

        if meta_key in self.cache and iq_key in self.cache:
            meta = self.cache[meta_key]
            raw_int = self.cache[iq_key]
        else:
            meta_file, data_file = self._find_pair(number)
            meta = requests.get(self.base_url + meta_file, timeout=30).json()
            raw_data = requests.get(self.base_url + data_file, timeout=60).content
            raw_int = np.frombuffer(raw_data, dtype=np.int8)
            self.cache[meta_key] = meta
            self.cache[iq_key] = raw_int

        params = IQParams(
            center_freq=int(meta["captures"][0]["core:frequency"]),
            sample_rate=int(meta["global"]["core:sample_rate"]),
            lna=int(meta["global"].get("hackrf:lna_gain_db", 0)),
            vga=int(meta["global"].get("hackrf:vga_gain_db", 0)),
            amp_enabled=bool(meta["global"].get("hackrf:amp_enabled", False)),
        )

        return raw_int, params


def make_config(
    target_hz: int,
    sample_rate_hz: int,
    if_gain_db: int,
    baseband_gain_db: int,
    amp_enable: bool,
    record_number: int,
    num_samples: int,
    offset_hz: Optional[int] = None,
) -> AcquisitionConfig:
    if offset_hz is None:
        offset_hz = sample_rate_hz // 4
    return AcquisitionConfig(
        target_frequency_hz=target_hz,
        offset_frequency_hz=offset_hz,
        sample_rate_hz=sample_rate_hz,
        num_samples=num_samples,
        if_gain_db=if_gain_db,
        baseband_gain_db=baseband_gain_db,
        amp_enable=amp_enable,
        record_number=record_number,
    )


def _to_complex_iq(raw_iq_int8: np.ndarray) -> np.ndarray:
    i_raw = raw_iq_int8[0::2].astype(np.float32)
    q_raw = raw_iq_int8[1::2].astype(np.float32)
    return ((i_raw + 1j * q_raw) * (1.0 / 128.0)).astype(np.complex64)


def run_stage1(
    target_frequency_hz: Optional[int] = None,
    sample_rate_hz: Optional[int] = None,
    if_gain_db: Optional[int] = None,
    baseband_gain_db: Optional[int] = None,
    amp_enable: Optional[bool] = None,
    device_index: int = 0,
    lna_gain_db: Optional[int] = None,
    vga_gain_db: Optional[int] = None,
    record_number: int = 1,
    num_samples: Optional[int] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> AcquisitionResult:
    del device_index  # Kept for backward-compatible signature.

    repo = SigMFRepo(cache_dir=cache_dir)
    raw_int, params = repo.get_measure(record_number)

    if lna_gain_db is not None:
        if_gain_db = lna_gain_db
    if vga_gain_db is not None:
        baseband_gain_db = vga_gain_db

    target = int(params.center_freq if target_frequency_hz is None else target_frequency_hz)
    sr = int(params.sample_rate if sample_rate_hz is None else sample_rate_hz)
    ifg = int(params.lna if if_gain_db is None else if_gain_db)
    bbg = int(params.vga if baseband_gain_db is None else baseband_gain_db)
    amp = bool(params.amp_enabled if amp_enable is None else amp_enable)

    iq = _to_complex_iq(raw_int)
    if num_samples is None:
        num_samples = int(iq.size)
    iq = iq[:num_samples]

    cfg = make_config(
        target_hz=target,
        sample_rate_hz=sr,
        if_gain_db=ifg,
        baseband_gain_db=bbg,
        amp_enable=amp,
        record_number=record_number,
        num_samples=len(iq),
    )

    clipping = bool(np.any((raw_int == 127) | (raw_int == -128)))
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

    logger.info(
        "Loaded IQ measure %02d from DB: n=%d, f0=%.3f MHz, fs=%.3f Msps",
        record_number,
        len(iq),
        cfg.target_frequency_hz / 1e6,
        cfg.sample_rate_hz / 1e6,
    )

    return AcquisitionResult(
        samples=iq,
        config=cfg,
        timestamp_utc=timestamp,
        clipping_detected=clipping,
    )


def load_iq_records(
    numbers: Optional[list[int]] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_samples: Optional[int] = None,
) -> list[AcquisitionResult]:
    if numbers is None:
        numbers = DEFAULT_MEASURES
    records: list[AcquisitionResult] = []
    for n in numbers:
        records.append(run_stage1(record_number=int(n), cache_dir=cache_dir, num_samples=num_samples))
    return records
