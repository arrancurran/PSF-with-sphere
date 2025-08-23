#!/usr/bin/env python3
"""
Focused Gaussian pair (2D TM, Ez) with MST force on a cylinder.

- Two continuous-wave focused sources (±y) shaped by a spherical phase mask.
- Frequency-domain |Ez|^2 snapshot.
- Time-averaged MST force (Fx,Fy) per unit length on an infinite cylinder.
- Optional animation sweeping the cylinder center along x.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# Matplotlib backend: allow interactive windows with --show, otherwise use Agg
import matplotlib
import sys

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import meep as mp
import numpy as np

# Optional import for user-defined spherical phase constructor
try:
    import src_spherical_phase as src  # expects make_spherical_phase(...)
except Exception:  # pragma: no cover
    src = None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SimConfig:
    resolution: int = 15
    sx: float = 20.0
    sy: float = 14.0
    dpml: float = 1.0

    # Sources
    src_lambda: float = 1.064               # free-space wavelength (um)
    w_src: float = 20.0                     # 1/e radius at source plane (y)
    amp_scale: float = 1.0                  # Scale this to real world powers to get SI force
    x_focus: float = -4.0
    x_src: float = -sx / 2 + dpml
    trap_sep: float = 2.0 / src_lambda      # separation in y between the two sources (units of λ)

    # Media / NA
    target_NA: float = 1.42
    n_medium: float = 1.518                 # immersion oil
    n_water: float = 1.33                   # sample medium
    n_sphere: float = 1.49                  # cylinder index

    # Cylinder
    sphere_radius: float = 0.9 / src_lambda # radius in Meep units

    # Water block (z along sim x)
    xmin_block: float = -sx / 2 + dpml + 1.0
    xmax_block: float = sx / 2

    # Run / averaging
    n_cycles_to_settle: int = 100
    samples_per_cycle: int = 10
    n_cycles_avg: int = 10

    # Plot normalization (set to None to auto-lock from first frame)
    vmin: float | None = None
    vmax: float | None = None


# --------------------------------------------------------------------------- #
# Focused phase amplitude helper
# --------------------------------------------------------------------------- #

def make_focused_amp_callable(
    w_src: float,
    aperture_radius: float,
    k0: float,
    d: float,
    amp_scale: float,
):
    """
    Fallback implementation when src_spherical_phase is unavailable.
    Returns a function f(mp.Vector3)->complex giving the amplitude at a source point.
    """
    def _amp(p: mp.Vector3) -> complex:
        y = p.y
        # Gaussian envelope
        amp = np.exp(-(y / w_src) ** 2)
        # Smooth aperture roll-off
        edge_sigma = max(0.01, 0.05 * aperture_radius)
        amp *= 1.0 / (1.0 + np.exp((abs(y) - aperture_radius) / edge_sigma))
        # Spherical converging phase to (x_focus,0): exp(-ik (sqrt(d^2+y^2)-d))
        phase = np.exp(-1j * k0 * (np.sqrt(d * d + y * y) - d))
        return complex(amp_scale * amp * phase)
    return _amp


# --------------------------------------------------------------------------- #
# MST force (2D TM: Ez,Hx,Hy)
# --------------------------------------------------------------------------- #

def _line_sampling(sim: mp.Simulation, center: mp.Vector3, size: mp.Vector3, comp):
    """Return 1D array sampled along a line (size zero in one axis)."""
    arr = sim.get_array(component=comp, center=center, size=size)
    return np.squeeze(arr)  # Meep returns a 2D array even for lines


def _mst_face_flux_tm(Ez, Hx, Hy, eps_b, mu_b, normal_xy, dl) -> Tuple[float, float]:
    """
    Instantaneous Maxwell stress tensor for TMz (Ez, Hx, Hy).
    T_ij = eps*E_i E_j + mu*H_i H_j - 0.5*(eps*|E|^2 + mu*|H|^2) delta_ij
    Here E=(0,0,Ez), H=(Hx,Hy,0). Returns (dFx, dFy) over the face by summing (T·n) dl.
    """
    E2 = np.real(Ez * Ez)
    Hx2 = np.real(Hx * Hx)
    Hy2 = np.real(Hy * Hy)
    H2 = Hx2 + Hy2

    Txx = mu_b * Hx2 - 0.5 * (eps_b * E2 + mu_b * H2)
    Tyy = mu_b * Hy2 - 0.5 * (eps_b * E2 + mu_b * H2)
    Txy = mu_b * Hx * Hy

    nx, ny = normal_xy
    Txn_x = Txx * nx + Txy * ny
    Txn_y = Txy * nx + Tyy * ny

    dFx = float(np.sum(Txn_x) * dl)
    dFy = float(np.sum(Txn_y) * dl)
    return dFx, dFy


def compute_mst_force_2d_tm(
    sim: mp.Simulation,
    cyl_center: mp.Vector3,
    cyl_radius: float,
    n_bg: float = 1.33,
    samples_per_cycle: int = 40,
    n_cycles_avg: int = 10,
) -> Tuple[float, float]:
    """
    Time-average the MST over n_cycles_avg cycles for 2D TM (Ez, Hx, Hy).
    The closed surface is an axis-aligned rectangle around the cylinder,
    offset by ~3 grid cells from the surface and kept inside the background medium.
    Returns (Fx, Fy) per unit length (z-invariant).
    """
    eps_b = n_bg ** 2
    mu_b = 1.0

    dx = 3.0 / sim.resolution
    hx = cyl_radius + dx
    hy = cyl_radius + dx

    cx, cy = cyl_center.x, cyl_center.y

    pad = getattr(sim.boundary_layers[0], "thickness", 0.0) if sim.boundary_layers else 0.0
    xmin = max(-0.5 * sim.cell_size.x + pad + 2 / sim.resolution, cx - hx)
    xmax = min(+0.5 * sim.cell_size.x - pad - 2 / sim.resolution, cx + hx)
    ymin = max(-0.5 * sim.cell_size.y + pad + 2 / sim.resolution, cy - hy)
    ymax = min(+0.5 * sim.cell_size.y - pad - 2 / sim.resolution, cy + hy)

    Nx = max(2, int(np.ceil((xmax - xmin) * sim.resolution)))
    Ny = max(2, int(np.ceil((ymax - ymin) * sim.resolution)))

    dl_x = (xmax - xmin) / Nx
    dl_y = (ymax - ymin) / Ny

    # Determine CW frequency
    f_list = [s.src.frequency for s in sim.sources if isinstance(s.src, mp.ContinuousSource)]
    if not f_list:
        raise RuntimeError("MST force averaging requires a steady CW; no ContinuousSource found.")
    f0 = f_list[0]
    period = 1.0 / f0
    dt = period / samples_per_cycle
    n_steps = samples_per_cycle * n_cycles_avg

    Fx_sum = 0.0
    Fy_sum = 0.0

    for _ in range(n_steps):
        # LEFT face (normal = -x̂)
        center = mp.Vector3(xmin, 0.5 * (ymin + ymax), 0)
        size = mp.Vector3(0, (ymax - ymin), 0)
        Ez_L = _line_sampling(sim, center, size, mp.Ez)
        Hx_L = _line_sampling(sim, center, size, mp.Hx)
        Hy_L = _line_sampling(sim, center, size, mp.Hy)
        FLx, FLy = _mst_face_flux_tm(Ez_L, Hx_L, Hy_L, eps_b, mu_b, normal_xy=(-1.0, 0.0), dl=dl_y)

        # RIGHT face (+x̂)
        center = mp.Vector3(xmax, 0.5 * (ymin + ymax), 0)
        size = mp.Vector3(0, (ymax - ymin), 0)
        Ez_R = _line_sampling(sim, center, size, mp.Ez)
        Hx_R = _line_sampling(sim, center, size, mp.Hx)
        Hy_R = _line_sampling(sim, center, size, mp.Hy)
        FRx, FRy = _mst_face_flux_tm(Ez_R, Hx_R, Hy_R, eps_b, mu_b, normal_xy=(+1.0, 0.0), dl=dl_y)

        # BOTTOM face (-ŷ)
        center = mp.Vector3(0.5 * (xmin + xmax), ymin, 0)
        size = mp.Vector3((xmax - xmin), 0, 0)
        Ez_B = _line_sampling(sim, center, size, mp.Ez)
        Hx_B = _line_sampling(sim, center, size, mp.Hx)
        Hy_B = _line_sampling(sim, center, size, mp.Hy)
        FBx, FBy = _mst_face_flux_tm(Ez_B, Hx_B, Hy_B, eps_b, mu_b, normal_xy=(0.0, -1.0), dl=dl_x)

        # TOP face (+ŷ)
        center = mp.Vector3(0.5 * (xmin + xmax), ymax, 0)
        size = mp.Vector3((xmax - xmin), 0, 0)
        Ez_T = _line_sampling(sim, center, size, mp.Ez)
        Hx_T = _line_sampling(sim, center, size, mp.Hx)
        Hy_T = _line_sampling(sim, center, size, mp.Hy)
        FTx, FTy = _mst_face_flux_tm(Ez_T, Hx_T, Hy_T, eps_b, mu_b, normal_xy=(0.0, +1.0), dl=dl_x)

        Fx_sum += (FLx + FRx + FBx + FTx)
        Fy_sum += (FLy + FRy + FBy + FTy)

        sim.run(until=dt)

    return Fx_sum / n_steps, Fy_sum / n_steps


# --------------------------------------------------------------------------- #
# Simulation + plotting (stable, non-jitter)
# --------------------------------------------------------------------------- #

# Cache for auto-locked normalization across runs (if cfg.vmin/vmax are None)
_locked_norm: tuple[float, float] | None = None

def run_once_and_plot(
    cfg: SimConfig,
    sphere_x: float,
    out_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[float, float]:
    """Run one steady-state simulation and create a plot (saved or shown)."""
    # Derived quantities
    cell = mp.Vector3(cfg.sx, cfg.sy, 0)
    pml_layers = [mp.PML(cfg.dpml)]
    src_freq = 1.0 / cfg.src_lambda
    fwidth = 0.3 * src_freq
    k0 = 2 * np.pi / cfg.src_lambda
    d = cfg.x_focus - cfg.x_src

    # NA check and aperture
    if cfg.target_NA > cfg.n_medium:
        raise ValueError(f"target_NA={cfg.target_NA} exceeds n_medium={cfg.n_medium}")
    theta_max = np.arcsin(cfg.target_NA / cfg.n_medium)
    aperture_radius = abs(d) * np.tan(theta_max)

    # Amplitude function (prefer user's src module if present)
    if src is not None and hasattr(src, "make_spherical_phase"):
        amp_func = src.make_spherical_phase(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)
    else:
        amp_func = make_focused_amp_callable(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)

    # Source region: line along y at x = x_src
    src_region = mp.Vector3(0, cfg.sy - 2 * cfg.dpml, 0)
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=src_freq, fwidth=fwidth),
            component=mp.Ez,
            center=mp.Vector3(cfg.x_src, -cfg.trap_sep / 2, 0),
            size=src_region,
            amp_func=amp_func,
        ),
        mp.Source(
            src=mp.ContinuousSource(frequency=src_freq, fwidth=fwidth),
            component=mp.Ez,
            center=mp.Vector3(cfg.x_src, +cfg.trap_sep / 2, 0),
            size=src_region,
            amp_func=amp_func,
        ),
    ]

    # Geometry: background oil, embedded water slab, cylinder
    block_center_x = 0.5 * (cfg.xmax_block + cfg.xmin_block)
    block_size_x = abs(cfg.xmax_block - cfg.xmin_block)
    sphere_center = mp.Vector3(sphere_x, 0, 0)

    geometry = [
        mp.Block(
            size=mp.Vector3(block_size_x, cfg.sy, mp.inf),
            center=mp.Vector3(block_center_x, 0, 0),
            material=mp.Medium(index=cfg.n_water),
        ),
        mp.Cylinder(
            radius=cfg.sphere_radius,
            center=sphere_center,
            height=mp.inf,
            material=mp.Medium(index=cfg.n_sphere),
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        resolution=cfg.resolution,
        dimensions=2,
        default_material=mp.Medium(index=cfg.n_medium),
    )

    # DFT monitor (avoid PML in plots)
    mon_center = mp.Vector3(0, 0, 0)
    mon_size = mp.Vector3(cfg.sx - 2 * cfg.dpml, cfg.sy - 2 * cfg.dpml, 0)
    dft = sim.add_dft_fields([mp.Ez], src_freq, 0, 1, center=mon_center, size=mon_size)

    # Steady-state run
    t_run = cfg.n_cycles_to_settle / src_freq
    sim.run(until=t_run)

    # MST force
    Fx, Fy = compute_mst_force_2d_tm(
        sim=sim,
        cyl_center=sphere_center,
        cyl_radius=cfg.sphere_radius,
        n_bg=cfg.n_water,
        samples_per_cycle=cfg.samples_per_cycle,
        n_cycles_avg=cfg.n_cycles_avg,
    )
    logging.info("[MST 2D TM] Force per unit length on cylinder: Fx=%.6e, Fy=%.6e (Meep units)", Fx, Fy)

    # Fetch DFT Ez and intensity
    ez_f = sim.get_dft_array(dft, mp.Ez, 0)
    I = np.abs(ez_f) ** 2

    # Axes mapping: treat propagation as 'z' (simulation x), transverse as 'x' (simulation y)
    nx, ny = ez_f.shape[0], ez_f.shape[1]
    z = np.linspace(-0.5 * mon_size.x, 0.5 * mon_size.x, nx)  # vertical
    x = np.linspace(-0.5 * mon_size.y, 0.5 * mon_size.y, ny)  # horizontal

    # ------------------------------ Plot (stable) ---------------------------------
    # Fixed figure size and axes/cbar slots; no tight_layout, no bbox='tight'
    fig = plt.figure(figsize=(6.0, 4.5), constrained_layout=False)
    ax = fig.add_axes([0.12, 0.12, 0.72, 0.76])    # left, bottom, width, height
    cax = fig.add_axes([0.86, 0.12, 0.03, 0.76])   # colorbar slot

    # Lock color normalization across frames:
    global _locked_norm
    if cfg.vmin is not None and cfg.vmax is not None:
        vmin, vmax = float(cfg.vmin), float(cfg.vmax)
    else:
        if _locked_norm is None:
            vmin = float(np.min(I))
            vmax = float(np.percentile(I, 99.5))  # robust upper cap
            _locked_norm = (vmin, vmax)
        vmin, vmax = _locked_norm

    pcm = ax.pcolormesh(x, z, I, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(rf"$|E_z(\lambda={cfg.src_lambda:.3f})|^2$", fontsize=11)

    ax.set_xlabel(r"$x\;(\lambda)$")
    ax.set_ylabel(r"$z\;(\lambda)$")

    # Fix axes limits and aspect; disable autoscale to prevent jitter
    full_z_half = 0.5 * cell.x
    full_x_half = 0.5 * cell.y
    ax.set_xlim(-full_x_half, full_x_half)
    ax.set_ylim(-full_z_half, full_z_half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.set_autoscale_on(False)
    ax.margins(0)

    # PML shading
    hatch = "///"
    for p in (
        patches.Rectangle((-full_x_half, -full_z_half), cfg.dpml, 2 * full_z_half, linewidth=0, hatch=hatch, zorder=2),
        patches.Rectangle((full_x_half - cfg.dpml, -full_z_half), cfg.dpml, 2 * full_z_half, linewidth=0, hatch=hatch, zorder=2),
        patches.Rectangle((-full_x_half, -full_z_half), 2 * full_x_half, cfg.dpml, linewidth=0, hatch=hatch, zorder=2),
        patches.Rectangle((-full_x_half, full_z_half - cfg.dpml), 2 * full_x_half, cfg.dpml, linewidth=0, hatch=hatch, zorder=2),
    ):
        ax.add_patch(p)

    # Water/oil interface (text inside axes so it doesn't affect layout)
    interface_z = cfg.xmin_block
    ax.hlines(interface_z, -full_x_half + cfg.dpml, full_x_half - cfg.dpml, colors="cyan", linewidth=2, zorder=6)

    x_text = full_x_half - cfg.dpml - 0.5
    ax.text(x_text, interface_z + 0.4, "water", color="cyan", fontsize=10, ha="right", va="bottom", zorder=7)
    ax.text(x_text, interface_z - 0.6, "oil",   color="white", fontsize=10, ha="right", va="top",    zorder=7)

    # Cylinder outline (remember plot axes are (x,y)->(x,z) with swap)
    theta = np.linspace(-np.pi, np.pi, 200)
    ax.plot(
        sphere_center.y + cfg.sphere_radius * np.cos(theta),
        sphere_center.x + cfg.sphere_radius * np.sin(theta),
        "w-",
        linewidth=2,
        label="Cylinder",
    )

    # Force arrow (map sim Fx,Fy -> plot (dy,dx))
    F_norm = float(np.hypot(Fx, Fy))
    if F_norm > 0:
        scale = cfg.sphere_radius / F_norm
        dx_plot, dy_plot = Fy * scale, Fx * scale
        ax.quiver(
            sphere_center.y, sphere_center.x,
            dx_plot, dy_plot,
            angles="xy", scale_units="xy", scale=1.0,
            color="red", width=0.01, zorder=10,
        )

    # Small annotation inside axes (constant layout)
    ax.text(
        -full_x_half + 0.2, full_z_half - 0.4,
        f"F=({Fx:.2e}, {Fy:.2e})",
        color="white", fontsize=9, ha="left", va="top",
        bbox=dict(facecolor=(0, 0, 0, 0.3), edgecolor="none", pad=2),
    )

    # Save or show (no bbox='tight' to keep canvas fixed)
    if out_path:
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        fig.savefig("frame.png", dpi=180)
        plt.close(fig)

    return Fx, Fy


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Focused Gaussian pair with moving cylinder animation")
    p.add_argument("--animate", action="store_true", help="Sweep cylinder x-position and save frames")
    p.add_argument("--x-start", type=float, default=-4.0, help="Start x position")
    p.add_argument("--x-end", type=float, default=4.0, help="End x position")
    p.add_argument("--x-step", type=float, default=0.1, help="Step size for x sweep")
    p.add_argument("--out-dir", type=str, default="frames", help="Directory to save frames")
    p.add_argument("--show", action="store_true", help="Show plot for a single run (ignored if --animate)")
    p.add_argument("--sphere-x", type=float, default=0.60 / 1.064, help="Cylinder center x for a single run")
    p.add_argument("--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    # Optional explicit color scale (for reproducibility across separate runs)
    p.add_argument("--vmin", type=float, default=None, help="Fixed color scale minimum (default: auto from first frame)")
    p.add_argument("--vmax", type=float, default=None, help="Fixed color scale maximum (default: auto from first frame)")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    cfg = SimConfig(vmin=args.vmin, vmax=args.vmax)

    if args.animate:
        xs = np.arange(args.x_start, args.x_end + 1e-12, args.x_step)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Animating %d frames from x=%.2f to x=%.2f in %.2f steps → %s",
                     len(xs), xs[0], xs[-1], args.x_step, out_dir)
        for i, x in enumerate(xs):
            # ffmpeg-friendly, fixed-width numeric sequence
            fname = out_dir / f"frame_{i:04d}.png"
            logging.info("[Frame %d/%d] sphere_x=%.3f → %s", i + 1, len(xs), x, fname)
            run_once_and_plot(cfg=cfg, sphere_x=x, out_path=str(fname), show=False)
        logging.info("Done. Create a video with, e.g.: ffmpeg -framerate 20 -i %s -c:v libx264 -pix_fmt yuv420p output.mp4",
                     str(out_dir / "frame_%04d.png"))
    else:
        run_once_and_plot(cfg=cfg, sphere_x=args.sphere_x, out_path=None, show=args.show)


if __name__ == "__main__":
    main()