
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import norm
import h5py


TARGET_COLS   = ['planet_temp', 'log_H2O', 'log_CO2', 'log_CH4', 'log_CO', 'log_NH3']
UNITS         = ['K', 'log VMR', 'log VMR', 'log VMR', 'log VMR', 'log VMR']
N_TARGETS     = len(TARGET_COLS)
_NON_PARAM_COLS = {"public_key", "planet_ID"}
TRAINING_MEAN = [1193.84508481,   -6.61742274,   -6.74424125,   -6.48690315, -5.42609959,   -7.24521743]
TRAINING_STD = [677.75622031,   2.26626616,   1.76707911,   2.24564161, 1.63712806,   1.98380339]


def _score_split(y_true: np.ndarray, mu: np.ndarray, std: np.ndarray) -> dict:
    """
    Core CRPS skill score computation for a single (N, P) split.
    Internal helper -- call compute_leaderboard_score from outside.
    """
    # --- Gaussian CRPS per sample per parameter ---
    # CRPS(N(mu, sigma), y) = sigma * [z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]
    # where z = (y - mu) / sigma
    z = (y_true - mu) / std
    crps = std * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )  # shape (N, P)

    # --- Baseline: predict training mean (0) with unit spread (1) ---
    # Since targets are normalised, this is the trivial "know-nothing" baseline.
    z_ref    = y_true                  # equiv. (y_true - 0) / 1
    crps_ref = 1.0 * (
        z_ref * (2 * norm.cdf(z_ref) - 1)
        + 2 * norm.pdf(z_ref)
        - 1.0 / np.sqrt(np.pi)
    )  # shape (N, P)

    # --- Aggregate ---
    crps_per_param     = crps.mean(axis=0)      # (P,)
    crps_ref_per_param = crps_ref.mean(axis=0)  # (P,)

    skill_per_param = 1.0 - crps_per_param / crps_ref_per_param  # (P,)
    skill_score     = float(skill_per_param.mean())               # scalar

    mean_spread = float(std.mean())

    return {
        "score":         skill_score,
        "mean_crps":           float(crps.mean()),
        "crps_per_param":      crps_per_param,
        "score_per_param":     skill_per_param,
    }

def compute_participant_score(y_true, mu, std) -> dict:
    """
    Compute the CRPS skill score for a participant submission.
    No public/private split — returns a single score dict.

    Parameters
    ----------
    y_true : pd.DataFrame
        Ground truth table. Must contain "planet_ID" and the physical
        parameter columns. No "public_key" column needed.
    mu : pd.DataFrame
        Participant predicted means. Must contain "planet_ID" and the same
        physical parameter columns as y_true (in the same order).
    std : pd.DataFrame
        Participant predicted standard deviations. Same structure as mu.

    Returns
    -------
    dict — score dict (see _score_split for full key descriptions).
    """
    param_cols = [c for c in y_true.columns if c not in _NON_PARAM_COLS]

    y_params   = y_true[param_cols].to_numpy(dtype=float)
    mu_params  = mu[param_cols].to_numpy(dtype=float)
    std_params = std[param_cols].to_numpy(dtype=float)

    # --- Normalise to zero mean, unit variance using training statistics ---
    y_params   = (y_params   - TRAINING_MEAN) / TRAINING_STD
    mu_params  = (mu_params  - TRAINING_MEAN) / TRAINING_STD  # shift + scale
    std_params =  std_params                  / TRAINING_STD  # scale only

    if np.any(std_params <= 0):
        raise ValueError(
            "All std values must be strictly positive. "
            f"Found {np.sum(std_params <= 0)} non-positive entries."
        )
    if not (y_params.shape == mu_params.shape == std_params.shape):
        raise ValueError(
            f"Shape mismatch: y_params={y_params.shape}, "
            f"mu={mu_params.shape}, std={std_params.shape}. All must be identical."
        )

    return _score_split(y_params, mu_params, std_params)


def array_to_submission(arr: np.ndarray, planet_ids=None) -> "pd.DataFrame":
    """
    Convert a numpy array of predictions into the submission DataFrame format.
 
    Parameters
    ----------
    arr : np.ndarray, shape (N, 6)
        Each row is one planet; columns are planet_temp, log_H2O, log_CO2,
        log_CH4, log_CO, log_NH3 — in that order.
    planet_ids : array-like of length N, optional
        Planet IDs to assign. If None, uses 0-based integer indices.
 
    Returns
    -------
    pd.DataFrame with columns ["planet_ID", "planet_temp", "log_H2O",
        "log_CO2", "log_CH4", "log_CO", "log_NH3"].
    """
    import pandas as pd
 
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(TARGET_COLS):
        raise ValueError(
            f"Expected shape (N, {len(TARGET_COLS)}), got {arr.shape}. "
            f"Columns must be: {TARGET_COLS}"
        )
 
    if planet_ids is None:
        planet_ids = np.arange(len(arr))
 
    df = pd.DataFrame(arr, columns=TARGET_COLS)
    df.insert(0, "planet_ID", planet_ids)
    return df

def load_spectral_data(spectral_data_path):
    with h5py.File(spectral_data_path, "r") as h5f:
        n_planets = len(h5f.keys())
        noise_stack    = np.zeros((n_planets, 52))
        spectrum_stack = np.zeros((n_planets, 52))

        for i, planet in enumerate(range(n_planets)):
            spectrum_stack[i] = h5f[f'Planet_{planet}']['instrument_spectrum'][:]
            noise_stack[i]    = h5f[f'Planet_{planet}']['instrument_noise'][:]

        first = next(iter(h5f.keys()))
        wl_grid = h5f[first]['instrument_wlgrid'][:]
        width   = h5f[first]['instrument_width'][:]

    return spectrum_stack, noise_stack, wl_grid, width

def style_ax(ax):
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')
    ax.grid(True, linestyle='-', linewidth=0.4, color='#21262d', alpha=0.8)


def plot_predicted_vs_true(y_true_arr, y_pred_mean):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Predicted vs True', color='#e6edf3', fontsize=12, y=1.01)

    for i, ax in enumerate(axes.flat):
        style_ax(ax)
        yt, yp = y_true_arr[:, i], y_pred_mean[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2     = 1 - ss_res / ss_tot

        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], color='#f0883e',
                linewidth=1.0, linestyle='--', label='1:1', zorder=3)
        ax.scatter(yt, yp, s=2, alpha=0.3, color='#58a6ff', linewidths=0)
        ax.set_xlabel(f'True  ({UNITS[i]})', color='#8b949e', fontsize=9)
        ax.set_ylabel(f'Predicted  ({UNITS[i]})', color='#8b949e', fontsize=9)
        ax.set_title(f'{TARGET_COLS[i]}   R² = {r2:.3f}',
                     color='#c9d1d9', fontsize=9, pad=6)

    plt.tight_layout()
    plt.show()


def plot_residuals(y_true_arr, y_pred_mean):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Residual distributions  (predicted − true)',
                 color='#e6edf3', fontsize=12, y=1.01)

    for i, ax in enumerate(axes.flat):
        style_ax(ax)
        residuals = y_pred_mean[:, i] - y_true_arr[:, i]
        ax.hist(residuals, bins=60, color='#58a6ff',
                alpha=0.7, edgecolor='none')
        ax.axvline(0,            color='#f0883e', linewidth=1.0, linestyle='--', label='zero')
        ax.axvline(residuals.mean(), color='#3fb950', linewidth=1.0,
                   linestyle=':', label=f'mean = {residuals.mean():.3f}')
        ax.set_xlabel(f'Residual  ({UNITS[i]})', color='#8b949e', fontsize=9)
        ax.set_ylabel('Count', color='#8b949e', fontsize=9)
        ax.set_title(TARGET_COLS[i], color='#c9d1d9', fontsize=9, pad=6)
        ax.legend(fontsize=7.5, framealpha=0.15, facecolor='#161b22',
                  edgecolor='#21262d', labelcolor='#c9d1d9')

    plt.tight_layout()
    plt.show()


def plot_calibration(y_true_arr, y_pred_mean, y_pred_std, n_bins=15):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('Uncertainty calibration  (mean |error| vs mean σ per bin)',
                 color='#e6edf3', fontsize=12, y=1.01)

    for i, ax in enumerate(axes.flat):
        style_ax(ax)
        abs_err = np.abs(y_pred_mean[:, i] - y_true_arr[:, i])
        std     = y_pred_std[:, i]
        bins    = np.percentile(std, np.linspace(0, 100, n_bins + 1))
        bin_idx = np.digitize(std, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        mean_std, mean_err = [], []
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 5:
                mean_std.append(std[mask].mean())
                mean_err.append(abs_err[mask].mean())

        lo = min(min(mean_std), min(mean_err))
        hi = max(max(mean_std), max(mean_err))
        ax.plot([lo, hi], [lo, hi], color='#f0883e',
                linewidth=1.0, linestyle='--', label='perfect calibration')
        ax.plot(mean_std, mean_err, 'o-', color='#58a6ff',
                markersize=4, linewidth=1.2, label='model')
        ax.set_xlabel('Mean σ (uncertainty)', color='#8b949e', fontsize=9)
        ax.set_ylabel('Mean |error|', color='#8b949e', fontsize=9)
        ax.set_title(TARGET_COLS[i], color='#c9d1d9', fontsize=9, pad=6)
        ax.legend(fontsize=7.5, framealpha=0.15, facecolor='#161b22',
                  edgecolor='#21262d', labelcolor='#c9d1d9')

    plt.tight_layout()
    plt.show()


def plot_error_vs_uncertainty(y_true_arr, y_pred_mean, y_pred_std):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('|Error| vs uncertainty', color='#e6edf3', fontsize=12, y=1.01)

    for i, ax in enumerate(axes.flat):
        style_ax(ax)
        abs_err = np.abs(y_pred_mean[:, i] - y_true_arr[:, i])
        std     = y_pred_std[:, i]
        corr    = np.corrcoef(std, abs_err)[0, 1]

        ax.scatter(std, abs_err, s=2, alpha=0.3, color='#58a6ff', linewidths=0)
        ax.set_xlabel('σ (uncertainty)', color='#8b949e', fontsize=9)
        ax.set_ylabel('|Error|', color='#8b949e', fontsize=9)
        ax.set_title(f'{TARGET_COLS[i]}   ρ = {corr:.3f}',
                     color='#c9d1d9', fontsize=9, pad=6)

    plt.tight_layout()
    plt.show()


def plot_spectrum(planet_id, spectrum_stack, noise_stack, wl_grid, y_true=None):
    """
    Visualise the transmission spectrum for a single planet.

    Parameters
    ----------
    planet_id : int or str
        Either the integer index (0-based) or the planet_ID string (e.g. '21988').
    spectrum_stack : np.ndarray, shape (n_planets, n_channels)
    noise_stack    : np.ndarray, shape (n_planets, n_channels)
    wl_grid        : np.ndarray, shape (n_channels,)
    y_true         : pd.DataFrame, optional
        Training targets with columns:
        planet_ID, planet_temp, log_H2O, log_CO2, log_CH4, log_CO, log_NH3
    """

    TARGET_COLS = ['planet_temp', 'log_H2O', 'log_CO2', 'log_CH4', 'log_CO', 'log_NH3']
    UNITS       = ['K', '', '', '', '', '']
    LABELS      = ['Temperature', 'log H₂O', 'log CO₂', 'log CH₄', 'log CO', 'log NH₃']

    # --- resolve planet_id to a row index ---------------------------------
    if isinstance(planet_id, str):
        if y_true is None:
            raise ValueError("y_true must be provided to look up by planet_ID string.")
        match = y_true.index[y_true['planet_ID'].astype(str) == planet_id].tolist()
        if not match:
            raise ValueError(f"planet_ID '{planet_id}' not found in y_true.")
        idx = match[0]
    elif isinstance(planet_id, (int, np.integer)):
        idx = int(planet_id)
    else:
        raise TypeError("planet_id must be an int (row index) or str (planet_ID label).")

    display_name = str(planet_id) if isinstance(planet_id, str) else \
                   (str(y_true.iloc[idx]['planet_ID']) if y_true is not None else str(idx))

    spec = spectrum_stack[idx]
    err  = noise_stack[idx]
    snr  = spec / err

    # --- layout -----------------------------------------------------------
    has_targets = y_true is not None
    fig = plt.figure(figsize=(14, 7 if has_targets else 6))
    fig.patch.set_facecolor('#0d1117')

    if has_targets:
        gs = gridspec.GridSpec(
            2, 2,
            height_ratios=[4, 1],
            width_ratios=[3, 1],
            hspace=0.08, wspace=0.28,
            left=0.08, right=0.97, top=0.88, bottom=0.12
        )
        ax_spec = fig.add_subplot(gs[0, 0])
        ax_snr  = fig.add_subplot(gs[1, 0], sharex=ax_spec)
        ax_tgt  = fig.add_subplot(gs[:, 1])
    else:
        gs = gridspec.GridSpec(
            2, 1, height_ratios=[4, 1], hspace=0.08,
            left=0.09, right=0.97, top=0.88, bottom=0.12
        )
        ax_spec = fig.add_subplot(gs[0])
        ax_snr  = fig.add_subplot(gs[1], sharex=ax_spec)

    # --- shared axis styling ----------------------------------------------
    def style_ax(ax):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#21262d')
        ax.grid(True, which='major', linestyle='-',  linewidth=0.4, color='#21262d', alpha=0.8)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color='#21262d', alpha=0.4)

    style_ax(ax_spec)
    style_ax(ax_snr)

    # --- spectrum panel ---------------------------------------------------
    ax_spec.fill_between(wl_grid, spec - 2*err, spec + 2*err,
                         alpha=0.10, color='#388bfd', linewidth=0, label='2σ')
    ax_spec.fill_between(wl_grid, spec - err, spec + err,
                         alpha=0.25, color='#388bfd', linewidth=0, label='1σ')
    ax_spec.plot(wl_grid, spec, color='#58a6ff', linewidth=1.0, alpha=0.6)
    ax_spec.errorbar(wl_grid, spec, yerr=err,
                     fmt='o', markersize=3.5, color='#58a6ff',
                     ecolor='#388bfd', elinewidth=0.8,
                     capsize=2.5, capthick=0.8,
                     label='Observed spectrum', zorder=5)

    ax_spec.set_ylabel('Transit Depth (Rp/Rs)²', color='#8b949e', fontsize=10)
    ax_spec.legend(fontsize=8.5, framealpha=0.15, facecolor='#161b22',
                   edgecolor='#21262d', labelcolor='#c9d1d9', loc='upper right')
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    # --- SNR panel --------------------------------------------------------
    ax_snr.fill_between(wl_grid, 0, snr, alpha=0.4, color='#3fb950', linewidth=0)
    ax_snr.plot(wl_grid, snr, color='#3fb950', linewidth=1.0)
    ax_snr.axhline(y=5, color='#3fb950', linewidth=0.6,
                   linestyle='--', alpha=0.5, label='SNR = 5')
    ax_snr.set_ylabel('SNR', color='#8b949e', fontsize=8)
    ax_snr.set_xlabel('Wavelength (μm)', color='#8b949e', fontsize=10)
    ax_snr.set_ylim(bottom=0)

    for ax in [ax_spec, ax_snr]:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:.1f}' if x < 10 else f'{int(x)}'))
        ax.xaxis.set_minor_formatter(NullFormatter())

    # --- target panel -----------------------------------------------------
    if has_targets:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        row = y_true.iloc[idx]

        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[:, 1],
            height_ratios=[1, 5], hspace=0.35
        )
        ax_temp = fig.add_subplot(gs_inner[0])
        ax_abun = fig.add_subplot(gs_inner[1])

        for ax in [ax_temp, ax_abun]:
            style_ax(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # -- temperature (single bar) --
        temp_val = row['planet_temp']
        ax_temp.barh([0], [temp_val], color='yellow', alpha=0.75,
                     height=0.5, zorder=3)
        ax_temp.set_yticks([0])
        ax_temp.set_yticklabels(['Temperature'], fontsize=9, color='#c9d1d9')
        ax_temp.set_xlabel('K', color='#8b949e', fontsize=8)
        ax_temp.tick_params(axis='x', colors='#8b949e', labelsize=8)
        ax_temp.text(temp_val + ax_temp.get_xlim()[1] * 0.02, 0,
                     f'{temp_val:.0f} K',
                     va='center', ha='left', fontsize=8, color='#c9d1d9')
        ax_temp.set_title('Ground truth', color='#8b949e',
                           fontsize=9, pad=6, loc='left')

        # -- log abundances --
        ABUN_COLS   = ['log_H2O', 'log_CO2', 'log_CH4', 'log_CO', 'log_NH3']
        ABUN_LABELS = ['log H₂O', 'log CO₂', 'log CH₄', 'log CO', 'log NH₃']
        abun_vals   = [row[c] for c in ABUN_COLS]
        y_pos       = np.arange(len(ABUN_LABELS))

        ax_abun.barh(y_pos, abun_vals, color='#58a6ff', alpha=0.75,
                     height=0.55, zorder=3)
        ax_abun.set_yticks(y_pos)
        ax_abun.set_yticklabels(ABUN_LABELS, fontsize=9, color='#c9d1d9')
        ax_abun.set_xlabel('log VMR', color='#8b949e', fontsize=8)
        ax_abun.tick_params(axis='x', colors='#8b949e', labelsize=8)
        ax_abun.axvline(x=0, color='#21262d', linewidth=0.8)

        x_range = max(abun_vals) - min(abun_vals)
        for val, y in zip(abun_vals, y_pos):
            ax_abun.text(val + x_range * 0.03, y,
                         f'{val:.2f}',
                         va='center', ha='left', fontsize=8, color='#c9d1d9')

    # --- title and metadata -----------------------------------------------
    fig.suptitle(f'Ariel Transmission Spectrum  —  Planet {display_name}',
                 color='#e6edf3', fontsize=12, fontweight='normal', y=0.95)
    mean_snr = np.nanmean(snr)
    peak_wl  = wl_grid[np.argmax(snr)]
    fig.text(0.97 if not has_targets else 0.60, 0.93,
             f'mean SNR {mean_snr:.1f}  |  peak at {peak_wl:.2f} μm',
             ha='right', va='top', fontsize=8, color='#8b949e', style='italic')

    plt.show()

def plot_population_overview(spectrum_stack, noise_stack, wl_grid,
                              n_planets=1000):
    """
    Population-level overview of the first n_planets spectra and,
    optionally, the distribution of their atmospheric parameters.

    Parameters
    ----------
    spectrum_stack : np.ndarray, shape (n_planets, n_channels)
    noise_stack    : np.ndarray, shape (n_planets, n_channels)
    wl_grid        : np.ndarray, shape (n_channels,)
    n_planets      : int, default 1000
    """

    TARGET_COLS  = ['planet_temp', 'log_H2O', 'log_CO2', 'log_CH4', 'log_CO', 'log_NH3']
    ABUN_LABELS  = ['Temperature (K)', 'log H₂O', 'log CO₂', 'log CH₄', 'log CO', 'log NH₃']
    COLORS       = ['#e3b341', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff', '#58a6ff']

    n          = min(n_planets, spectrum_stack.shape[0])
    specs      = spectrum_stack[:n]
    errs       = noise_stack[:n]
    mean_noise = errs.mean(axis=0)

    # --- layout -----------------------------------------------------------

    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(
        2, 1, height_ratios=[4, 1], hspace=0.08,
        left=0.08, right=0.97, top=0.93, bottom=0.10
    )
    ax_spec  = fig.add_subplot(gs[0])
    ax_noise = fig.add_subplot(gs[1], sharex=ax_spec)

    def style_ax(ax):
        ax.set_facecolor('#0d1117')
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#21262d')
        ax.grid(True, which='major', linestyle='-',
                linewidth=0.4, color='#21262d', alpha=0.8)

    def set_log_xaxis(ax):
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f'{x:.1f}' if x < 10 else f'{int(x)}'))
        ax.xaxis.set_minor_formatter(NullFormatter())

    # --- spectrum panel ---------------------------------------------------
    style_ax(ax_spec)

    for i in range(min(n, 100)):
        ax_spec.plot(wl_grid, specs[i], color='#58a6ff',
                     linewidth=0.4, alpha=0.2)

    set_log_xaxis(ax_spec)
    ax_spec.set_ylabel('Transit Depth (Rp/Rs)²', color='#8b949e', fontsize=10)
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    # --- noise panel ------------------------------------------------------
    style_ax(ax_noise)
    noise_p16, noise_p84 = np.percentile(errs, 16, axis=0), np.percentile(errs, 84, axis=0)
    ax_noise.fill_between(wl_grid, noise_p16, noise_p84,
                          color='#3fb950', alpha=0.25, linewidth=0)
    ax_noise.plot(wl_grid, mean_noise, color='#3fb950',
                  linewidth=1.0, label='Mean noise')
    set_log_xaxis(ax_noise)
    ax_noise.set_ylabel('Noise', color='#8b949e', fontsize=8)
    ax_noise.set_xlabel('Wavelength (μm)', color='#8b949e', fontsize=10)


    fig.suptitle(f'Population overview  —  first {n} planets',
                 color='#e6edf3', fontsize=12, y=0.97)
    plt.show()