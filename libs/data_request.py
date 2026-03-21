"""Data acquisition helpers for ANE RSM campaigns and realtime signals.

This module defines strongly-typed containers (using :class:`dataclasses.dataclass`)
for campaign metadata and realtime signal payloads, plus the
:class:`DataRequest` client used to consume ANE's REST API endpoints.

The typical workflow is:

1. Instantiate :class:`DataRequest` with an optional logger and the API base
   URL.
2. Use :meth:`get_campaign_params` to inspect a campaign's configuration and
   derived ``pxx_len``.  This is useful for setting up processing loops or
   just understanding the measurement settings.
3. Call :meth:`load_campaigns_and_nodes` to bulk-download/paginate signal
   measurements for a list of campaigns and sensor nodes.  Results are
   returned as a nested dictionary of :class:`pandas.DataFrame` instances and
   are cached locally in ``.cache`` for faster reloads.
4. For quick, single-node realtime snapshots use :meth:`get_realtime_signal`.

Example (see module ``__main__`` block for a runnable demo)::

    from libs.data_request import DataRequest
    import cfg

    log = cfg.set_logger()
    dr = DataRequest(log=log, base_url=cfg.API_URL)

    # inspect campaign metadata
    cinfo = dr.get_campaign_params(176)
    print(cinfo)

    # load all nodes from a campaign
    campaigns = {'FM original': 176}
    nodes = list(range(1, 11))
    df_dict = dr.load_campaigns_and_nodes(campaigns, nodes)

    # realtime snapshot for node 1
    realtime = dr.get_realtime_signal(1)
    print(realtime)
"""

import requests
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import diskcache as dc
from urllib3.exceptions import InsecureRequestWarning
import urllib3

urllib3.disable_warnings(InsecureRequestWarning)

@dataclass
class ScheduleParams:
    """Campaign schedule definition returned by the API.

    Attributes:
        start_date (str): Inclusive campaign start date (YYYY-MM-DD).
        end_date (str): Inclusive campaign end date (YYYY-MM-DD).
        start_time (str): Daily acquisition start time.
        end_time (str): Daily acquisition end time.
        interval_seconds (int): Sampling interval in seconds.
    """

    start_date: str
    end_date: str
    start_time: str
    end_time: str
    interval_seconds: int

@dataclass
class ConfigParams:
    """SDR configuration parameters associated with a campaign.

    Attributes:
        rbw (str): Resolution bandwidth value as provided by the API.
        span (int): Frequency span in hertz.
        antenna (str): Antenna identifier or mode.
        lna_gain (int): LNA gain level.
        vga_gain (int): VGA gain level.
        antenna_amp (bool): Whether the antenna amplifier is enabled.
        center_freq_hz (int): Center frequency in hertz.
        sample_rate_hz (int): Sample rate in hertz.
        centerFrequency (int): Alternate center frequency key from API payload.
    """

    rbw: str
    span: int
    antenna: str
    lna_gain: int
    vga_gain: int
    antenna_amp: bool
    center_freq_hz: int
    sample_rate_hz: int
    centerFrequency: int

@dataclass
class RealtimeSignal:
    """Realtime signal payload for a single sensor node.

    Attributes:
        mac (str): Sensor MAC address.
        active_configuration (str): Active SDR configuration identifier.
        pxx (numpy.ndarray): Power spectral density vector.
        start_freq_hz (int): Start frequency in hertz for the PSD vector.
        end_freq_hz (int): End frequency in hertz for the PSD vector.
    """

    mac: str
    active_configuration: str
    pxx: np.ndarray
    start_freq_hz: int
    end_freq_hz: int

@dataclass
class CampaignParams:
    """Campaign-level metadata and derived processing parameters.

    Attributes:
        name (str): Campaign display name.
        schedule (ScheduleParams): Schedule information.
        config (ConfigParams): SDR configuration for the campaign.
        pxx_len (int): Derived FFT length based on sample rate and RBW.
    """

    name: str
    schedule: ScheduleParams
    config: ConfigParams
    pxx_len: int

class DataRequest:
    """Client for ANE RSM API endpoints used in this project.

    The client supports:

    * realtime signal retrieval via :meth:`get_realtime_signal`
    * campaign parameter discovery via :meth:`get_campaign_params`
    * paginated download of historical signals via :meth:`get_api_signals`
    * high‑level bulk loading with automatic disk caching via
      :meth:`load_campaigns_and_nodes`

    ``DataRequest`` is intentionally lightweight and depends only on
    ``requests`` for HTTP, ``pandas``/``numpy`` for data wrangling, and
    ``diskcache`` to persist downloaded DataFrames.  If you are analyzing
    campaign data or writing a notebook, instantiate this once and reuse it
    across cells.
    """

    def __init__(self, log=None, base_url="https://rsm.ane.gov.co:12443/api"):
        """Initialize a request client and its local cache.

        Args:
            log (logging.Logger, optional): External logger instance. If not
                provided, a default logger named ``DataRequest`` is created.
            base_url (str): API base URL.
        """
        self.base_url = base_url

        # Initialize diskcache (creates a folder named '.cache' in your directory)
        self.cache = dc.Cache('.cache')

        if not log:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
            self._log = logging.getLogger("DataRequest")
        else:
            self._log = log

        self._log.info(f"Initialized DataRequest class with base_url: {self.base_url}")
        
        self.node_macs = {
            1: 'd8:3a:dd:f7:1d:f2', 2: 'd8:3a:dd:f4:4e:26', 3: 'd8:3a:dd:f7:22:87',
            4: 'd8:3a:dd:f6:fc:be', 5: 'd8:3a:dd:f7:21:52', 6: 'd8:3a:dd:f7:1a:cc',
            7: 'd8:3a:dd:f7:1d:b6', 8: 'd8:3a:dd:f7:1b:20', 9: 'd8:3a:dd:f4:4e:d1',
            10: 'd8:3a:dd:f7:1d:90'
        }

    def get_realtime_signal(self, node_id):
        """Fetch a realtime power spectrum for a single sensor node.

        The ANE API offers a lightweight ``/realtime`` endpoint that returns a
        one‑sample PSD vector for the requested node.  This method translates
        the node index into its MAC address (see :attr:`node_macs`) and
        performs the HTTP GET call, returning a :class:`RealtimeSignal` object
        that is convenient to inspect or plot.

        Parameters
        ----------
        node_id : int
            Integer identifier for the node (1 through 10).  The mapping to a
            MAC address is defined in the constructor; unknown IDs will return
            ``None`` values in the fields of the returned object.

        Returns
        -------
        RealtimeSignal
            Container with ``mac``, ``active_configuration`` and the PSD
            vector as a ``numpy.ndarray``.

        Raises
        ------
        requests.RequestException
            If there is a network error or non‑200 HTTP response.
        ValueError
            If the response is not valid JSON.
        """
        mac = self.node_macs.get(node_id)
        self._log.info(f"Fetching realtime signal for Node {node_id} (MAC: {mac})")
        url = f"{self.base_url}/campaigns/sensor/{mac}/realtime"
        data = requests.get(url, verify=False).json()
        
        sig_entry = data['signal'][0] if data.get('signal') else {}
        
        return RealtimeSignal(
            mac=data.get("mac"),
            active_configuration=data.get("active_configuration"),
            pxx=np.array(sig_entry.get("pxx", [])),
            start_freq_hz=sig_entry.get("start_freq_hz", 0),
            end_freq_hz=sig_entry.get("end_freq_hz", 0)
        )

    def get_campaign_params(self, campaign_id):
        """Retrieve metadata for a named campaign and compute derived values.

        The endpoint ``/campaigns/{id}/parameters`` returns JSON describing a
        campaign's name, schedule, and SDR configuration.  This method parses
        the response into :class:`ScheduleParams` and :class:`ConfigParams`
        dataclasses, then calculates ``pxx_len`` – the FFT length that should be
        used when converting raw ``pxx`` vectors to frequency axes.  The
        algorithm mirrors the logic used elsewhere in the project and handles
        malformed or missing values gracefully.

        Parameters
        ----------
        campaign_id : int
            Numeric ID of the campaign to query.

        Returns
        -------
        CampaignParams
            Container with ``name``, ``schedule``, ``config`` and ``pxx_len``.

        Raises
        ------
        requests.RequestException
            For networking errors or non‑200 status codes.
        ValueError
            If JSON parsing fails.
        """
        self._log.info(f"Fetching parameters for campaign ID: {campaign_id}")
        url = f"{self.base_url}/campaigns/{campaign_id}/parameters"
        data = requests.get(url, verify=False).json()
        
        config_data = data.get("config", {})
        
        # --- Calculate pxx_len ---
        # 1. Safely parse RBW (it comes in as a string according to your logs)
        try:
            rbw_val = float(config_data.get("rbw", 1000.0))
            safe_rbw = rbw_val if rbw_val > 0 else 1000.0
        except (ValueError, TypeError):
            safe_rbw = 1000.0
            
        # 2. Get sample rate
        sample_rate = float(config_data.get("sample_rate_hz", 0.0))
        
        # 3. Apply the C-formula logic
        if sample_rate > 0:
            enbw_factor = 1.363 # HAMMING_TYPE
            required_nperseg_val = enbw_factor * sample_rate / safe_rbw
            exponent = int(np.ceil(np.log2(required_nperseg_val)))
            pxx_len = int(2 ** exponent)
        else:
            pxx_len = 0 # Fallback if no sample rate is provided
        # -------------------------

        # Unpack dictionaries directly into the new dataclasses
        return CampaignParams(
            name=data.get("name"),
            schedule=ScheduleParams(**data.get("schedule", {})),
            config=ConfigParams(**config_data),
            pxx_len=pxx_len
        )

    def get_api_signals(self, mac, camp_id, node_desc):
        """Download all signal measurements for a single node / campaign pair.

        The /signals endpoint returns paginated results.  This helper iterates
        through pages automatically, updating a single ``tqdm`` progress bar so
        that notebooks and scripts show live progress.  If the request times
        out or an unexpected error occurs the loop breaks and whatever has been
        downloaded so far is returned.

        Parameters
        ----------
        mac : str
            Sensor MAC address, e.g. ``"d8:3a:dd:f7:1d:f2"``.
        camp_id : int
            Campaign ID to use as a filter parameter in the API call.
        node_desc : str
            Human‑readable label for logging/progress bar purposes (e.g. "Node
            3").

        Returns
        -------
        list
            Flattened list of measurement dictionaries as returned by the API.
        """
        url = f"{self.base_url}/campaigns/sensor/{mac}/signals"
        params = {"campaign_id": camp_id, "page": 1, "page_size": 1000}
        signals = []
        
        # Single, flat progress bar that works everywhere
        with tqdm(desc=f"  ↳ {node_desc}", unit="page", leave=True) as pbar:
            while True:
                try:
                    response = requests.get(url, params=params, verify=False, timeout=20)
                    data = response.json()
                    
                    measurements = data.get('measurements', [])
                    if not measurements:
                        break
                        
                    signals.extend(measurements)
                    pbar.update(1)
                    
                    if not data.get('pagination', {}).get('has_next'):
                        break
                    
                    params['page'] += 1
                    
                except Exception as e:
                    print(f"\n  [!] Timeout/Error on {node_desc} (MAC {mac}): {e}")
                    break 
            
        return signals

    def load_campaigns_and_nodes(self, campaigns, node_ids):
        """Batch-load signals for multiple campaigns and nodes with caching.

        Parameters
        ----------
        campaigns : dict[str, int]
            Mapping from campaign descriptive label (e.g. ``"FM original"``) to
            its numeric ID.  Labels are used as the top‑level keys in the
            returned dictionary.
        node_ids : list[int]
            List of integer node identifiers to request; they are translated to
            MAC addresses using :attr:`node_macs`.

        Returns
        -------
        dict[str, dict[str, pandas.DataFrame]]
            Structure ``{campaign_label: {"Node1": df1, "Node2": df2, ...}}``.
            DataFrames contain raw API measurement dictionaries with any rows
            missing ``pxx`` dropped.  Each DataFrame is also stored in the local
            ``.cache`` directory under a key like
            ``"campaign_176_node_3"`` so repeated calls are instantaneous.
        """
        self._log.info(f"Loading data for campaigns: {list(campaigns.keys())} and nodes: {node_ids}")
        df_full = {camp: {} for camp in campaigns}
        
        for camp, camp_id in campaigns.items():
            print(f"\n🚀 Starting Campaign: {camp}")
            for node_id in node_ids:
                mac = self.node_macs.get(node_id)
                if not mac:
                    continue

                # Define a unique key for the cache
                cache_key = f"campaign_{camp_id}_node_{node_id}"
                
                # Check if it exists in the cache
                if cache_key in self.cache:
                    print(f"  ↳ Node {node_id} loaded instantly from local cache")
                    df_full[camp][f"Node{node_id}"] = self.cache[cache_key]
                else:
                    datos = self.get_api_signals(mac, camp_id, node_desc=f"Node {node_id}")
                    if datos:
                        df = pd.DataFrame(datos).dropna(subset=['pxx'])
                        df_full[camp][f"Node{node_id}"] = df
                        # Save the DataFrame directly to the cache
                        self.cache[cache_key] = df
                        
        self._log.info("Finished loading data for all campaigns and nodes.")
        return df_full


# ---------------------------------------------------------------------------
# Example usage when run as a script / tutorial
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # This block can also serve as an executable tutorial in a notebook cell
    import cfg

    log = cfg.set_logger()
    dr = DataRequest(log=log, base_url=cfg.API_URL)

    # 1. inspect a campaign
    print("\n== Campaign metadata example ==")
    campaign_id = 176
    info = dr.get_campaign_params(campaign_id)
    log.info(f"Campaign name: {info.name}")
    log.info(f"Computed pxx_len: {info.pxx_len}")

    # 2. load signals for all nodes in a campaign (uses caching)
    print("\n== Bulk download example ==")
    camps = {"FM original": campaign_id}
    nodes = list(range(1, 11))
    df_data = dr.load_campaigns_and_nodes(camps, nodes)
    for camp_label, nodes_dict in df_data.items():
        log.info(f"Campaign {camp_label} has data for nodes: {list(nodes_dict.keys())}")

    # 3. realtime snapshot for node 1
    print("\n== Realtime signal example ==")
    realtime = dr.get_realtime_signal(1)
    log.info(f"Realtime PSD length: {len(realtime.pxx)}")
