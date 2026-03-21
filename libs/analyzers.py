import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SignalStabilityAnalyzer:
    """Analyse the spectral stability of a set of SDR nodes across a measurement
    campaign.

    The analyser operates on per-node :class:`pandas.DataFrame` objects whose
    ``pxx`` column stores Power Spectral Density (PSD) vectors in dB.  It
    builds probability density histograms, computes Pearson cross-correlation
    among the node spectra, and estimates pairwise Mutual Information (MI) to
    produce a ranked list of nodes ordered by spectral similarity.

    Key attributes set after calling :meth:`parse_data`
    ---------------------------------------------------
    node_densities : dict[str, numpy.ndarray]
        Per-node probability density vector :math:`\\hat{f}_i`.
    global_density : numpy.ndarray
        Density vector for the pooled dataset.
    means : numpy.ndarray
        Per-node sample mean :math:`\\mu_i` in dB.
    stds : numpy.ndarray
        Per-node sample standard deviation :math:`\\sigma_i` in dB.

    Key attributes set after calling :meth:`execute`
    -------------------------------------------------
    df_product : pandas.DataFrame
        Columns ``correlation``, ``MI``, and ``product`` — the combined
        ranking score
        :math:`\\text{score}_i = \\alpha |s_i^{\\text{corr}}| + (1-\\alpha) s_i^{\\text{MI}}`.
    """

    def __init__(self, datos_nodos, global_range=(-70, -20), alpha_node=0.72):
        """Initialize the analyzer with node data and histogram configuration.

        The number of histogram bins is derived from the PSD vector length
        :math:`N` using the adaptive formula:

        .. math::

            n_{\\text{bins}} = \\left\\lceil
            \\frac{\\log_2 N + \\sqrt{N}}{2} \\right\\rceil

        Bin edges are evenly spaced over ``global_range`` and bin centres are
        the midpoints :math:`b_k = \\tfrac{1}{2}(e_k + e_{k+1})`, where
        :math:`e_k` are the edges produced by :func:`numpy.linspace`.

        Parameters
        ----------
        datos_nodos : dict[str, pandas.DataFrame]
            Mapping of node labels to DataFrames that must contain a ``pxx``
            column holding PSD vectors (arrays, lists, or serialised strings).
        global_range : tuple[float, float], optional
            The shared :math:`[x_{\\min},\\, x_{\\max}]` dB range for all
            histograms.  Defaults to ``(-70, -20)``.
        alpha_node : float, optional
            Per-node curve opacity used in single-campaign plots.
            Defaults to ``0.72``.
        """
        self.datos_nodos = datos_nodos
        self.labels = list(datos_nodos.keys())
        self.n_nodes = len(self.labels)
        self.global_range = global_range
        self.alpha_node = alpha_node
        
        # Determine PXX length for bin calculation
        pxx_len = 1024
        for df in datos_nodos.values():
            if not df.empty:
                val = df['pxx'].iloc[0]
                pxx_len = len(ast.literal_eval(val) if isinstance(val, str) else val)
                break

        # Math for bins
        self.n_bins = int(np.ceil((np.log2(pxx_len) + np.sqrt(pxx_len)) / 2))
        self.bin_edges = np.linspace(global_range[0], global_range[1], self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        
        # Containers
        self.global_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.node_counts = {}
        self.node_densities = {}
        self.means = np.zeros(self.n_nodes)
        self.stds = np.zeros(self.n_nodes)
        self.total_values = 0

    @staticmethod
    def parse_pxx_cell(pxx_raw):
        """Parse a single PSD cell into a flat float64 NumPy array.

        Accepts the following representations of a PSD row:

        * :class:`numpy.ndarray` — cast to ``float64`` and flattened.
        * :class:`list` / :class:`tuple` — converted via :func:`numpy.array`.
        * :class:`pandas.Series` — cast via ``to_numpy``.
        * Serialised bracket string ``"[v1, v2, ...]"`` — parsed inline
          without :func:`ast.literal_eval` when the bracket form is detected;
          otherwise delegated to :func:`ast.literal_eval`.

        Parameters
        ----------
        pxx_raw : array-like or str
            Raw PSD value from a DataFrame cell.

        Returns
        -------
        numpy.ndarray
            One-dimensional array of PSD values in dB.
        """
        if isinstance(pxx_raw, np.ndarray): return pxx_raw.astype(float).ravel()
        if isinstance(pxx_raw, (list, tuple)): return np.array(pxx_raw, dtype=float).ravel()
        if isinstance(pxx_raw, pd.Series): return pxx_raw.to_numpy(dtype=float).ravel()
        s = str(pxx_raw).strip()
        if s.startswith("[") and s.endswith("]"):
            return np.fromiter(map(float, s[1:-1].replace(",", " ").split()), dtype=float)
        return np.asarray(ast.literal_eval(s), dtype=float).ravel()

    def parse_data(self):
        """Parse all node DataFrames and compute per-node probability densities.

        For each node :math:`i` the method concatenates every PSD row into a
        flat sample vector :math:`\\mathbf{x}_i`, then fills a histogram with
        :math:`n_{\\text{bins}}` bins over ``global_range``.

        The **probability density** for bin :math:`k` is estimated as:

        .. math::

            \\hat{f}_i(b_k) = \\frac{c_{i,k}}{N_i \\cdot \\Delta b}

        where :math:`c_{i,k}` is the raw count in bin :math:`k`,
        :math:`N_i = \\sum_k c_{i,k}` is the total sample count for node
        :math:`i`, and :math:`\\Delta b` is the uniform bin width.

        Node statistics are stored as:

        .. math::

            \\mu_i = \\frac{1}{N_i} \\sum_{j=1}^{N_i} x_{i,j}, \\qquad
            \\sigma_i = \\sqrt{\\frac{1}{N_i} \\sum_{j=1}^{N_i}
            \\left(x_{i,j} - \\mu_i\\right)^2}

        Side-effects
        ------------
        Populates :attr:`node_densities`, :attr:`global_density`,
        :attr:`node_counts`, :attr:`global_counts`, :attr:`means`,
        :attr:`stds`, and :attr:`total_values`.
        """
        for i, label in enumerate(self.labels):
            df = self.datos_nodos[label]
            flat = np.concatenate([self.parse_pxx_cell(val) for val in df['pxx']])
            self.total_values += flat.size
            
            counts, _ = np.histogram(flat, bins=self.bin_edges, range=self.global_range)
            self.node_counts[label] = counts
            self.global_counts += counts
            
            self.means[i] = np.mean(flat)
            self.stds[i] = np.std(flat)

        c_sum_global = self.global_counts.sum()
        self.global_density = self.global_counts / (c_sum_global * self.bin_width) if c_sum_global else self.global_counts
        
        for lbl, c in self.node_counts.items():
            c_sum = c.sum()
            self.node_densities[lbl] = c / (c_sum * self.bin_width) if c_sum > 0 else np.zeros_like(c, dtype=float)

    def get_correlation_scores(self):
        """Compute per-node spectral correlation scores from the density matrix.

        Let :math:`d_i \\in \\mathbb{R}^{n_{\\text{bins}}}` be the probability
        density vector of node :math:`i`.  The Pearson correlation between
        nodes :math:`i` and :math:`j` is:

        .. math::

            \\rho_{ij} = \\frac{\\displaystyle\\sum_k
            \\bigl(d_i(k) - \\bar{d}_i\\bigr)
            \\bigl(d_j(k) - \\bar{d}_j\\bigr)}{\\sqrt{\\displaystyle
            \\sum_k \\bigl(d_i(k) - \\bar{d}_i\\bigr)^2 \\;
            \\sum_k \\bigl(d_j(k) - \\bar{d}_j\\bigr)^2}}

        The diagonal is forced to zero (:math:`\\rho_{ii} = 0`) and the
        scalar score for node :math:`i` is the mean absolute off-diagonal
        correlation:

        .. math::

            s_i^{\\text{corr}} = \\frac{1}{K - 1}
            \\sum_{j \\neq i} |\\rho_{ij}|

        where :math:`K` is the total number of nodes.

        Returns
        -------
        pandas.Series
            Correlation score :math:`s_i^{\\text{corr}}` indexed by node label.
        """
        density_matrix = np.array([self.node_densities[lbl] for lbl in self.labels])
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(density_matrix)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 0.0) 
        return pd.Series(np.sum(np.abs(corr_matrix), axis=1) / (self.n_nodes - 1), index=self.labels)

    def get_single_row_vectors(self):
        """Extract one representative PSD spectrum per node.

        Iterates over the first 105 rows of each node's DataFrame and returns
        the first non-empty PSD vector found.  This single-row vector
        :math:`\\mathbf{v}_i \\in \\mathbb{R}^{N}` serves as a spectral
        *fingerprint* for node :math:`i` when computing pairwise mutual
        information in :meth:`MI_matrix`.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from node label to its representative PSD vector.
            Nodes that have no valid row within the first 105 entries are
            excluded from the returned dictionary.
        """
        vectors = {}
        for lbl, df in self.datos_nodos.items():
            for row in range(min(105, len(df))):
                try:
                    candidate = self.parse_pxx_cell(df["pxx"].iloc[row])
                    if candidate.size > 0:
                        vectors[lbl] = candidate
                        break
                except Exception: continue
        return vectors

    @staticmethod
    def mutual_information_hist(x, y, bins=64, edges=None):
        """Estimate the mutual information between two signals via a 2-D histogram.

        The joint distribution :math:`P(x, y)` is approximated by normalising
        a two-dimensional histogram.  Marginal distributions are obtained by
        summing along each axis:

        .. math::

            P(x) = \\sum_y P(x, y), \\qquad P(y) = \\sum_x P(x, y)

        Mutual information is then estimated as:

        .. math::

            I(X;\\, Y) = \\sum_{x,\\, y} P(x, y)
            \\log_2 \\frac{P(x, y)}{P(x)\\, P(y)}

        Only cells where :math:`P(x, y) > 0` and :math:`P(x)P(y) > 0`
        contribute to the sum, avoiding :math:`\\log(0)` singularities.

        Parameters
        ----------
        x : array-like
            First signal sample vector.
        y : array-like
            Second signal sample vector.  Truncated to ``len(x)`` if longer.
        bins : int, optional
            Number of bins per axis when ``edges`` is ``None``.
            Defaults to ``64``.
        edges : numpy.ndarray or None, optional
            Pre-computed 1-D bin edges shared by both axes.  When supplied,
            ``bins`` is ignored.

        Returns
        -------
        float
            Mutual information :math:`I(X;\\,Y)` in **bits**.
        """
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        H, _, _ = np.histogram2d(x, y, bins=[edges, edges] if edges is not None else bins)
        total_H = np.sum(H)
        if total_H == 0: return 0.0
        Pxy = H / total_H
        Px_Py = np.sum(Pxy, axis=1)[:, None] * np.sum(Pxy, axis=0)[None, :]
        nz = (Pxy > 0) & (Px_Py > 0)
        return np.sum(Pxy[nz] * np.log(Pxy[nz] / Px_Py[nz])) / np.log(2.0)

    def MI_matrix(self, bins=64):
        """Build the pairwise MI matrix and return normalised per-node scores.

        Let :math:`\\mathbf{v}_i` be the representative PSD vector of node
        :math:`i` (from :meth:`get_single_row_vectors`).  The symmetric MI
        matrix is defined as:

        .. math::

            M_{ij} = I(\\mathbf{v}_i;\\, \\mathbf{v}_j), \\quad M_{ii} = 0

        Bin edges are derived from the 1st–99th percentile of the pooled data
        to reduce sensitivity to outliers.

        The raw row-sum score :math:`\\tilde{s}_i = \\sum_j M_{ij}` is
        normalised to the unit interval:

        .. math::

            s_i^{\\text{MI}} =
            \\frac{\\tilde{s}_i}{\\displaystyle\\max_k \\tilde{s}_k}

        Parameters
        ----------
        bins : int, optional
            Number of histogram bins per axis.  Defaults to ``64``.

        Returns
        -------
        pandas.Series
            Normalised MI score :math:`s_i^{\\text{MI}} \\in [0, 1]` indexed
            by node label.  Nodes without a valid PSD vector receive ``0``.
        """
        vectors = self.get_single_row_vectors()
        if not vectors: return pd.Series(0, index=self.labels)
        all_vals = np.concatenate(list(vectors.values()))
        lo, hi = np.percentile(all_vals, (1.0, 99.0))
        edges = np.linspace(lo, hi, bins + 1)
        valid_labels = list(vectors.keys())
        M = np.zeros((len(valid_labels), len(valid_labels)))
        for i, ni in enumerate(valid_labels):
            for j, nj in enumerate(valid_labels):
                if j < i: continue
                M[i, j] = M[j, i] = 0.0 if i == j else self.mutual_information_hist(vectors[ni], vectors[nj], edges=edges)
        scores = pd.Series(M.sum(axis=1), index=valid_labels)
        if scores.max() > 0: scores = scores / scores.max()
        return scores.reindex(self.labels, fill_value=0.0)

    def execute(self, alpha=0.5):
        """Run the full analysis pipeline and build the node-ranking DataFrame.

        Calls :meth:`parse_data` (if not already run), then computes the
        correlation score :math:`s_i^{\\text{corr}}` via
        :meth:`get_correlation_scores` and the MI score
        :math:`s_i^{\\text{MI}}` via :meth:`MI_matrix`.

        The combined ranking metric is a convex combination:

        .. math::

            \\text{score}_i = \\alpha\\,\\bigl|s_i^{\\text{corr}}\\bigr|
            + (1 - \\alpha)\\, s_i^{\\text{MI}}, \\quad \\alpha \\in [0, 1]

        Results are stored in :attr:`df_product`, a
        :class:`pandas.DataFrame` with columns ``correlation``, ``MI``, and
        ``product``, sorted by ``MI`` in descending order.

        Parameters
        ----------
        alpha : float, optional
            Mixing coefficient :math:`\\alpha` weighting correlation vs.
            mutual information.  Defaults to ``0.5``.
        """
        if self.total_values == 0: self.parse_data()
        corr_scores = self.get_correlation_scores()
        mi_scores = self.MI_matrix(bins=self.n_bins)
        self.df_product = pd.DataFrame({'correlation': corr_scores, 'MI': mi_scores}).fillna(0)
        self.df_product['product'] = alpha * self.df_product['correlation'].abs() + (1 - alpha) * mi_scores
        self.df_product.index = [f"Node{l}" if str(l).isdigit() else str(l) for l in self.labels]
        self.df_product = self.df_product.sort_values(by='MI', ascending=False)

    # --- Plot Helpers (Multi-Campaign) ---
    def plot_hist(self, ax, label=None, color=None, **kwargs):
        """Overlay the global probability density on *ax* with a filled-mountain style.

        Renders the global density estimate :math:`\\hat{f}(b_k)` as a line
        plot with a semi-transparent filled area beneath it, suitable for
        overlaying multiple campaigns on the same axes without visual clutter.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes object.
        label : str or None, optional
            Legend label for this campaign's density curve.
        color : color-spec or None, optional
            Matplotlib colour for both the line and fill.  When ``None``,
            the next colour in the axes cycle is used automatically.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.plot`.
        """
        
        # 1. Draw the line (the 'mountain' peak)
        line, = ax.plot(self.bin_centers, self.global_density, 
                        label=label, linewidth=2, color=color, **kwargs)
        
        # 2. Get the color of the line to match the fill
        line_color = line.get_color()
        
        # 3. Add the fill (the mountain body)
        # alpha=0.2 makes it transparent so you can see overlapping campaigns
        ax.fill_between(self.bin_centers, self.global_density, 
                        color=line_color, alpha=0.2)
        
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_ylabel('Probability Density')

    def plot_ranking(self, ax1, label=None, color='b', alpha=0.8):
        """Overlay MI and combined-product ranking curves on a logarithmic axis.

        Two curves are drawn on a semi-log scale:

        * **MI score** :math:`s_i^{\\text{MI}}` — solid circle markers on
          the primary *ax1* (left :math:`y`-axis).
        * **Combined product** :math:`\\text{score}_i` — dashed square
          markers on a twin right-hand axis.

        Both axes share the same :math:`x`-ticks, labelled with the node
        names from :attr:`df_product` after sorting.

        Parameters
        ----------
        ax1 : matplotlib.axes.Axes
            Primary (left) axes for the MI curve.
        label : str or None, optional
            Prefix used in legend entries, e.g. ``"Campaign A"``.
        color : color-spec, optional
            Colour applied to both curves.  Defaults to ``'b'``.
        alpha : float, optional
            Base opacity for the MI curve; the product curve uses
            ``alpha * 0.7``.  Defaults to ``0.8``.

        Returns
        -------
        list[matplotlib.lines.Line2D]
            Concatenated list of both line artists for legend construction.
        """
        x_pos = np.arange(len(self.df_product))
        ln1 = ax1.semilogy(x_pos, self.df_product['MI'], marker='o', color=color, 
                           label=f'{label} MI', linewidth=2, markersize=7, alpha=alpha)
        
        ax2 = next((a for a in ax1.get_shared_x_axes().get_siblings(ax1) if a is not ax1), None)
        if ax2 is None: ax2 = ax1.twinx()
            
        ln2 = ax2.semilogy(x_pos, self.df_product['product'], marker='s', color=color, 
                           linestyle='--', label=f'{label} Prod', linewidth=1.5, markersize=6, alpha=alpha*0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.df_product.index, rotation=45, ha='right')
        return ln1 + ln2

    # --- Full Single-Campaign Graphic ---
    def execute_one_histogram(self, campaign_name="Campaign"):
        """Render the full single-campaign dual-panel diagnostic figure.

        Produces a stacked figure with two panels:

        **Panel A — PSD Distribution**
            Plots the global density :math:`\\hat{f}(b_k)` and all per-node
            densities :math:`\\hat{f}_i(b_k)` on the same axes.  Node legend
            entries include the sample mean :math:`\\mu_i`.

        **Panel B — Mean ± 1σ Bar Chart**
            For each node :math:`i`, draws a bar of height
            :math:`2\\sigma_i` centred on :math:`\\mu_i`, representing the
            symmetric uncertainty interval
            :math:`[\\mu_i - \\sigma_i,\\; \\mu_i + \\sigma_i]`.
            A horizontal reference line marks the global mean:

            .. math::

                \\bar{\\mu} = \\frac{1}{K} \\sum_{i=1}^{K} \\mu_i

        Parameters
        ----------
        campaign_name : str, optional
            Title label embedded in the figure heading.
            Defaults to ``"Campaign"``.
        """
        if self.total_values == 0: self.parse_data()
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, self.n_nodes))

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.4})

        # Panel A: PDF
        ax1.fill_between(self.bin_centers, self.global_density, alpha=0.1, color='steelblue')
        ax1.plot(self.bin_centers, self.global_density, color='steelblue', lw=3, label=f'All nodes (N={self.total_values:,})')
        for i, lbl in enumerate(self.labels):
            ax1.plot(self.bin_centers, self.node_densities[lbl], lw=1.2, alpha=self.alpha_node, 
                     color=colors[i], label=f"{lbl} μ={self.means[i]:.1f} dB")
        ax1.set_title(f"PSD Distribution — {campaign_name}", fontsize=14)
        ax1.set_ylabel('Probability Density')
        ax1.legend(ncol=3, loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.2)

        # Panel B: Mean/Sigma
        xs = np.arange(self.n_nodes)
        ax2.bar(xs, 2 * self.stds, bottom=self.means - self.stds, alpha=0.4, color=colors, width=0.6, label='μ ± 1σ')
        ax2.scatter(xs, self.means, color=colors, zorder=5, s=80, edgecolors='black')
        g_mean = np.mean(self.means)
        ax2.axhline(g_mean, color='steelblue', lw=1.5, ls='--', label=f'Global mean = {g_mean:.1f} dB')
        ax2.set_xticks(xs)
        ax2.set_xticklabels(self.labels, rotation=25, ha='right')
        ax2.set_ylabel('PSD (dB)')
        ax2.legend(loc='best')
        ax2.grid(True, axis='y', alpha=0.3)
        plt.show()