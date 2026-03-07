import glob
import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
FILE_GLOB = "Node*-Bogota.csv"
PXX_COL = "pxx"
EPS = 1e-12

# Histograma / PDF (estimada desde histograma)
N_BINS = 50
DENSITY_SMOOTH = True
SMOOTH_WINDOW = 7  # impar recomendado

# =========================
# Helpers
# =========================
def parse_vector(cell):
    """Convierte la celda 'pxx' a np.ndarray."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.asarray(cell, dtype=float)
    if pd.isna(cell):
        return None

    s = str(cell).strip()

    # Caso típico: "array([...])" / "np.array([...])" / "tensor([...])"
    for prefix in ("array(", "np.array(", "tensor("):
        if s.startswith(prefix) and s.endswith(")"):
            s = s[len(prefix):-1].strip()
            break

    try:
        v = ast.literal_eval(s)
        return np.asarray(v, dtype=float)
    except Exception:
        pass

    s2 = s.replace("\n", " ").replace("  ", " ")
    try:
        v = ast.literal_eval(s2)
        return np.asarray(v, dtype=float)
    except Exception as e:
        raise ValueError(
            f"No pude parsear pxx. Ejemplo: {str(cell)[:120]}..."
        ) from e


def moving_average(y, w):
    """Suavizado simple por media móvil (para la 'pdf' del histograma)."""
    if w <= 1:
        return y
    w = int(w)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


def robust_mad(x):
    """MAD = median(|x - median(x)|)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


# =========================
# Main
# =========================
files = sorted(glob.glob(FILE_GLOB))
if not files:
    raise FileNotFoundError(f"No encontré archivos con patrón: {FILE_GLOB} en {os.getcwd()}")

N_sensors = len(files)

# Prepara subplots: 3 columnas, filas = sensores
fig, axes = plt.subplots(
    nrows=N_sensors,
    ncols=3,
    figsize=(18, 4 * N_sensors),
    constrained_layout=True
)

# Si solo hay 1 sensor, axes sale 1D; lo convertimos a 2D
if N_sensors == 1:
    axes = axes.reshape(1, 3)

all_results = []

for i, fp in enumerate(files):
    sensor = os.path.splitext(os.path.basename(fp))[0]
    df = pd.read_csv(fp)

    if PXX_COL not in df.columns:
        raise KeyError(f"En {fp} no existe la columna '{PXX_COL}'. Columnas: {list(df.columns)}")

    # =========================
    # Extraer ruido por adquisición
    # ruido_i = median_k PSD_i[k]
    # =========================
    noise_vals = []
    bad_rows = 0

    for cell in df[PXX_COL].values:
        v = parse_vector(cell)
        if v is None or len(v) == 0:
            bad_rows += 1
            continue
        noise_vals.append(float(np.median(v)))

    noise_vals = np.asarray(noise_vals, dtype=float)

    if len(noise_vals) < 5:
        print(f"WARNING: {sensor}: muy pocas filas válidas ({len(noise_vals)}). No se graficará.")
        continue

    # =========================
    # Estadísticos solicitados
    # =========================
    mean_n = float(np.mean(noise_vals))          # valor esperado (media)
    var_n  = float(np.var(noise_vals, ddof=0))   # varianza (poblacional)
    std_n  = float(np.std(noise_vals, ddof=0))   # desviación estándar (poblacional)
    med_n  = float(np.median(noise_vals))        # mediana (robusto)

    mad_n = robust_mad(noise_vals)               # MAD (robusto)
    robust_std = float(1.4826 * mad_n)           # "std robusta" aprox si fuera gaussiana

    all_results.append({
        "sensor": sensor,
        "n_samples": int(len(noise_vals)),
        "bad_rows": int(bad_rows),
        "mean": mean_n,
        "median": med_n,
        "var": var_n,
        "std": std_n,
        "mad": mad_n,
        "robust_std(1.4826*MAD)": robust_std
    })

    # =========================
    # SUBPLOT (fila i, col 0): serie temporal del ruido
    # =========================
    ax = axes[i, 0]
    ax.plot(noise_vals, linewidth=1)
    ax.axhline(mean_n, linestyle="--", linewidth=2, label=f"Mean={mean_n:.3g}")
    ax.axhline(med_n,  linestyle="--", linewidth=2, label=f"Median={med_n:.3g}")
    ax.set_title(f"{sensor} — Ruido por adquisición (median(PSD))")
    ax.set_xlabel("Índice de adquisición (fila)")
    ax.set_ylabel("Ruido (unidades PSD)")
    ax.grid(True)
    ax.legend(fontsize=8)

    # =========================
    # SUBPLOT (fila i, col 1): histograma + PDF estimada
    # =========================
    ax = axes[i, 1]
    counts, bin_edges = np.histogram(noise_vals, bins=N_BINS, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    pdf_est = counts.copy()
    if DENSITY_SMOOTH:
        pdf_est = moving_average(pdf_est, SMOOTH_WINDOW)

    ax.hist(noise_vals, bins=N_BINS, density=True, alpha=0.4, label="Hist (densidad)")
    ax.plot(bin_centers, pdf_est, linewidth=2, label="PDF est. (hist)")
    ax.axvline(mean_n, linestyle="--", linewidth=2, label=f"Mean={mean_n:.3g}")
    ax.axvline(med_n,  linestyle="--", linewidth=2, label=f"Median={med_n:.3g}")

    # Texto corto con stats (sin saturar)
    ax.text(
        0.02, 0.98,
        f"std={std_n:.3g}\nvar={var_n:.3g}\nrob_std={robust_std:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    ax.set_title(f"{sensor} — Distribución del ruido")
    ax.set_xlabel("Ruido (median(PSD))")
    ax.set_ylabel("Densidad")
    ax.grid(True)
    ax.legend(fontsize=8)

    # =========================
    # SUBPLOT (fila i, col 2): Z-score robusto
    # =========================
    ax = axes[i, 2]
    z_robust = (noise_vals - med_n) / (robust_std + EPS)
    ax.plot(z_robust, linewidth=1)
    ax.axhline(3.0,  linestyle="--", linewidth=2, label="±3 robust std")
    ax.axhline(-3.0, linestyle="--", linewidth=2)
    ax.set_title(f"{sensor} — Z-score robusto del ruido")
    ax.set_xlabel("Índice de adquisición (fila)")
    ax.set_ylabel("Z robusto")
    ax.grid(True)
    ax.legend(fontsize=8)

# Mostrar el “panel” completo (filas=sensores, cols=3)
plt.show()

# =========================
# Resumen por sensor (tabla + 2 barras)
# =========================
summary = pd.DataFrame(all_results)
if len(summary) == 0:
    raise RuntimeError("No se generó resumen: ¿todas las filas fueron inválidas o faltó la columna pxx?")

summary = summary.sort_values("robust_std(1.4826*MAD)", ascending=False).reset_index(drop=True)

print("\n=== Resumen por sensor (ruido por adquisición = median(PSD)) ===")
print(summary[[
    "sensor", "n_samples", "bad_rows",
    "mean", "median", "var", "std",
    "mad", "robust_std(1.4826*MAD)"
]])

# Plot comparativo: std por sensor
plt.figure(figsize=(12, 4))
plt.bar(summary["sensor"], summary["std"])
plt.title("Comparación entre sensores — Desviación estándar del ruido (por adquisiciones)")
plt.xlabel("Sensor")
plt.ylabel("Std del ruido")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# Plot comparativo: std robusta por sensor
plt.figure(figsize=(12, 4))
plt.bar(summary["sensor"], summary["robust_std(1.4826*MAD)"])
plt.title("Comparación entre sensores — Std robusta del ruido (1.4826*MAD)")
plt.xlabel("Sensor")
plt.ylabel("Std robusta del ruido")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()