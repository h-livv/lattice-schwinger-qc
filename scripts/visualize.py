"""Load exact-diagonalization datasets from ``data/`` and write figures to ``figures/``.

Run from anywhere: ``python scripts/visualize.py``. Paths are relative to the
repository root, not the working directory.

No Hamiltonian, diagonalization, or time-evolution code lives here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito, one color per quench field in the order stored in the npz files.
_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")
_MARKERS = ("o", "s", "^", "D", "v", "P")


def _style():
    """Shared matplotlib defaults for every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.8,
        }
    )


def _savefig(fig, out_path):
    """Write ``fig`` to ``out_path`` and close it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def _param_line(N, a, m, g):
    """Single-line parameter caption."""
    return rf"$N={N}$, $a={a:g}$, $m={m:g}$, $g={g:g}$"


def plot_condensate_spectrum(data_path, out_path):
    """Fig. 1: chiral condensate and the lowest two energies versus external field."""
    _style()
    with np.load(data_path) as data:
        eps = np.array(data["eps"])
        a = float(data["a"])
        gamma = np.array(data["gamma"]) * a
        E0 = np.array(data["E0"]) * a
        E1 = np.array(data["E1"]) * a
        caption = _param_line(int(data["N"]), a, float(data["m"]), float(data["g"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    axes[0].plot(eps, gamma, color=_COLORS[0])
    axes[0].set_xlim(-0.04, 3.03)
    axes[0].set_ylim(-0.459, -0.211)
    axes[0].xaxis.set_major_locator(plt.MultipleLocator(1))
    axes[0].yaxis.set_major_locator(plt.MultipleLocator(0.05))
    axes[0].set_xlabel(r"$\varepsilon$")
    axes[0].set_ylabel(r"$\langle\bar{\psi}\psi\rangle\cdot a$")
    axes[0].set_title("(a)")
    axes[1].plot(eps, E0, color=_COLORS[0], label=r"$E_0$")
    axes[1].plot(eps, E1, color=_COLORS[1], label=r"$E_1$")
    axes[1].set_xlim(-0.04, 3.03)
    axes[1].set_ylim(-5.32, 9.05)
    axes[1].xaxis.set_major_locator(plt.MultipleLocator(1))
    axes[1].yaxis.set_major_locator(plt.MultipleLocator(2.5))
    axes[1].set_xlabel(r"$\varepsilon$")
    axes[1].set_ylabel(r"$E\cdot a$")
    axes[1].set_title("(b)")
    axes[1].legend()
    fig.suptitle(f"Exact diagonalization, {caption}")
    _savefig(fig, out_path)


def plot_critical_scaling(data_path, out_path):
    """Fig. 2: critical field versus 1/N with a degree-1 extrapolation to infinite volume."""
    _style()
    with np.load(data_path) as data:
        N = np.array(data["N"], dtype=float)
        eps_c = np.array(data["eps_c"], dtype=float)
        a = float(data["a"])
        m = float(data["m"])
        g = float(data["g"])
    inv_n = 1.0 / N
    slope, intercept = np.polyfit(inv_n, eps_c, 1)
    print(f"eps_c(N->inf) = {intercept:.6f}  (slope d(eps_c)/d(1/N) = {slope:.6f})")
    x_line = np.array([0.0, inv_n.max()])
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.plot(x_line, slope * x_line + intercept, "--", color="0.35", label="linear fit")
    ax.plot(inv_n, eps_c, "o", color=_COLORS[0], markersize=7, label="ED data")
    ax.scatter(
        [0.0],
        [intercept],
        marker="x",
        s=60,
        color=_COLORS[3],
        zorder=3,
        label=rf"$\varepsilon_c(\infty)\approx {intercept:.3f}$",
    )
    for n_val, x, y in zip(N, inv_n, eps_c):
        ax.annotate(
            f"N={int(n_val)}",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )
    ax.set_xlabel(r"$1/N$")
    ax.set_ylabel(r"$\varepsilon_c(N)$")
    ax.set_title(rf"Finite-size critical field ($a={a:g}$, $m={m:g}$, $g={g:g}$)")
    ax.legend()
    _savefig(fig, out_path)


def plot_conserved_quantities(data_path, out_path):
    """Fig. 3 sanity check: total charge and energy drift, both conserved by exact evolution."""
    _style()
    with np.load(data_path) as data:
        times = np.array(data["times"])
        a = float(data["a"])
        t_over_a = times / a
        eps = np.array(data["eps"])
        Q_N = np.array(data["Q_N"])
        delta_E = np.array(data["delta_E"]) * a
        caption = _param_line(int(data["N"]), a, float(data["m"]), float(data["g"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharex=True, constrained_layout=True)
    for k, eps_k in enumerate(eps):
        style = dict(
            color=_COLORS[k % len(_COLORS)],
            marker=_MARKERS[k % len(_MARKERS)],
            markevery=15,
            markersize=4,
            label=rf"$\varepsilon={eps_k:g}$",
        )
        axes[0].plot(t_over_a, Q_N[k], **style)
        axes[1].plot(t_over_a, delta_E[k], **style)
    q_max = np.max(np.abs(Q_N))
    e_max = np.max(np.abs(delta_E))
    axes[0].set_title("(a) Total charge")
    axes[0].set_ylabel(r"$Q_N(t)$")
    axes[0].text(
        0.03,
        0.97,
        rf"$\max|Q_N|={q_max:.1e}$",
        transform=axes[0].transAxes,
        va="top",
    )
    axes[1].set_title("(b) Energy drift")
    axes[1].set_ylabel(r"$\Delta E(t)\cdot a$")
    axes[1].text(
        0.03,
        0.97,
        rf"$\max|\Delta E\cdot a|={e_max:.1e}$",
        transform=axes[1].transAxes,
        va="top",
    )
    for ax in axes:
        ax.set_xlim(0.0, 12.0)
        ax.set_ylim(-0.05, 0.05)
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
        ax.set_xlabel(r"$t/a$")
        ax.axhline(0.0, color="0.5", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(eps), frameon=False)
    fig.suptitle(
        "Numerical sanity check (exact evolution), not a physics result\n"
        f"{caption}. Both quantities are conserved and should stay at machine precision."
    )
    _savefig(fig, out_path)


def _plot_charge_grid(
    times, eps, curves, site_labels, ylabel, title, out_path, ylim, ystep, xstep, label_all_x=False
):
    """Subplot grid of charge-versus-time, one panel per spatial point or site."""
    _style()
    n_sites = curves.shape[-1]
    ncols = 2 if n_sites <= 4 else 4
    nrows = int(np.ceil(n_sites / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.1 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for i, ax in enumerate(axes.ravel()):
        if i >= n_sites:
            ax.axis("off")
            continue
        for k, eps_k in enumerate(eps):
            ax.plot(
                times,
                curves[k, :, i],
                color=_COLORS[k % len(_COLORS)],
                marker=_MARKERS[k % len(_MARKERS)],
                markevery=15,
                markersize=3.5,
                label=rf"$\varepsilon={eps_k:g}$",
            )
        ax.set_xlim(0.0, 12.0)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(plt.MultipleLocator(xstep))
        ax.yaxis.set_major_locator(plt.MultipleLocator(ystep))
        ax.set_title(site_labels[i])
        ax.set_ylabel(ylabel)
        on_bottom = i >= (nrows - 1) * ncols
        if label_all_x or on_bottom:
            ax.set_xlabel(r"$t/a$")
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(eps), frameon=False)
    fig.suptitle(title)
    _savefig(fig, out_path)


def plot_spatial_charge(data_path, out_path):
    """Fig. 4: spatial-point charge Q_i(t), one panel per spatial point."""
    with np.load(data_path) as data:
        times = np.array(data["times"]) / float(data["a"])
        eps = np.array(data["eps"])
        Q_i = np.array(data["Q_i"])
        N = int(data["N"])
        caption = _param_line(N, float(data["a"]), float(data["m"]), float(data["g"]))
    labels = [rf"$i={i + 1}$" for i in range(Q_i.shape[-1])]
    _plot_charge_grid(
        times,
        eps,
        Q_i,
        labels,
        r"$Q_i(t)$",
        f"Spatial-point charge, {caption}",
        out_path,
        ylim=(-0.606, 0.606),
        ystep=0.25,
        xstep=2,
        label_all_x=True,
    )


def plot_local_charge(data_path, out_path):
    """Fig. 8: site-resolved charge q_n(t), one panel per lattice site."""
    with np.load(data_path) as data:
        times = np.array(data["times"]) / float(data["a"])
        eps = np.array(data["eps"])
        q_n = np.array(data["q_n"])
        caption = _param_line(int(data["N"]), float(data["a"]), float(data["m"]), float(data["g"]))
    labels = [rf"$n={n + 1}$" for n in range(q_n.shape[-1])]
    _plot_charge_grid(
        times,
        eps,
        q_n,
        labels,
        r"$q_n(t)$",
        f"Site-resolved charge, {caption}",
        out_path,
        ylim=(-0.63, 0.63),
        ystep=0.5,
        xstep=5,
    )


def plot_field_energy(data_path, out_path):
    """Fig. 5: electric-field energy versus time, one panel per quench field."""
    _style()
    with np.load(data_path) as data:
        a = float(data["a"])
        times = np.array(data["times"]) / a
        eps = np.array(data["eps"])
        H_E = np.array(data["H_E"]) * a
        caption = _param_line(int(data["N"]), a, float(data["m"]), float(data["g"]))
    # Paper Fig. 5 windows, opened downward when the exact curve would otherwise leave the frame.
    he_windows = {
        0.5: ((0.930, 1.004), 0.02),
        1.0: ((3.327, 3.659), 0.1),
        1.5: ((6.840, 8.091), 0.25),
        2.0: ((11.398, 14.314), 0.5),
    }
    n_eps = eps.size
    ncols = 2
    nrows = int(np.ceil(n_eps / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8.6, 2.6 * nrows),
        sharex=False,
        squeeze=False,
        constrained_layout=True,
    )
    for k, ax in enumerate(axes.ravel()):
        if k >= n_eps:
            ax.axis("off")
            continue
        ax.plot(times, H_E[k], color=_COLORS[k % len(_COLORS)])
        lo, hi = he_windows[float(eps[k])][0]
        step = he_windows[float(eps[k])][1]
        lo = min(lo, np.floor(H_E[k].min() / step) * step)
        hi = max(hi, np.ceil(H_E[k].max() / step) * step)
        ax.set_xlim(0.0, 12.0)
        ax.set_ylim(lo, hi)
        ax.xaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(step))
        ax.set_title(rf"$\varepsilon={eps[k]:g}$")
        ax.set_xlabel(r"$t/a$")
        ax.set_ylabel(r"$H_E(t)\cdot a$")
    fig.suptitle(f"Electric-field energy, {caption}")
    _savefig(fig, out_path)


def plot_vacuum_fidelity(data_path, out_path):
    """Fig. 6: vacuum fidelity versus time, one curve per quench field."""
    _style()
    with np.load(data_path) as data:
        a = float(data["a"])
        times = np.array(data["times"]) / a
        eps = np.array(data["eps"])
        P_vac = np.array(data["P_vac"])
        caption = _param_line(int(data["N"]), a, float(data["m"]), float(data["g"]))
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    for k, eps_k in enumerate(eps):
        ax.plot(
            times,
            P_vac[k],
            color=_COLORS[k % len(_COLORS)],
            marker=_MARKERS[k % len(_MARKERS)],
            markevery=12,
            markersize=4,
            label=rf"$\varepsilon={eps_k:g}$",
        )
    ax.axvspan(0.0, 1.0, color="0.5", alpha=0.12, label=r"Fig. 7 fit window")
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(-0.006, 1.019)
    ax.xaxis.set_major_locator(plt.MultipleLocator(2))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r"$t/a$")
    ax.set_ylabel(r"$P_{\mathrm{vac}}(t)$")
    ax.set_title(f"Vacuum fidelity, {caption}")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    _savefig(fig, out_path)


def plot_decay_rate(data_path, out_path):
    """Fig. 7: early-time effective decay rate versus quench-field strength."""
    _style()
    with np.load(data_path) as data:
        eps = np.array(data["eps"])
        a = float(data["a"])
        gamma_eff = np.array(data["gamma_eff"]) * a
        N = int(data["N"])
        t_min = float(data["t_over_a_min"])
        t_max = float(data["t_over_a_max"])
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.plot(
        eps,
        gamma_eff,
        "o-",
        color=_COLORS[0],
        markersize=7,
    )
    ax.set_xlim(0.375, 3.125)
    ax.set_ylim(0.0, max(1.0, float(np.ceil(gamma_eff.max() / 0.2) * 0.2)))
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\gamma_{\mathrm{eff}}\cdot a$")
    ax.set_title(
        rf"Effective decay rate ($N={N}$, $a={a:g}$)"
        "\n"
        rf"fit $-\ln P_{{\mathrm{{vac}}}}=\gamma_{{\mathrm{{eff}}}} t+c$ "
        rf"on $t/a\in[{t_min:g},{t_max:g}]$"
    )
    _savefig(fig, out_path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    figures = root / "figures"
    plot_condensate_spectrum(
        data / "condensate_spectrum_n8.npz",
        figures / "fig1_condensate_spectrum.png",
    )
    plot_critical_scaling(
        data / "critical_field_scaling.npz",
        figures / "fig2_critical_scaling.png",
    )
    plot_conserved_quantities(
        data / "quench_conserved_n8.npz",
        figures / "fig3_conserved_sanity.png",
    )
    plot_spatial_charge(
        data / "charge_dynamics_n8.npz",
        figures / "fig4_spatial_charge.png",
    )
    plot_local_charge(
        data / "charge_dynamics_n8.npz",
        figures / "fig8_local_charge.png",
    )
    plot_field_energy(
        data / "field_energy_n8.npz",
        figures / "fig5_field_energy.png",
    )
    plot_vacuum_fidelity(
        data / "vacuum_fidelity_n8.npz",
        figures / "fig6_vacuum_fidelity.png",
    )
    plot_decay_rate(
        data / "decay_rate_n8.npz",
        figures / "fig7_decay_rate.png",
    )
