#!/usr/bin/env python3
"""
visualize_langevin.py — Plot and analyze trajectories from langevin_driver.py

Examples
--------
# Basic trajectory + time series + forces
python visualize_langevin.py --traj traj.csv --outdir figs --all

# Overlay the path on trap intensity (recomputed without the particle)
python visualize_langevin.py --traj traj.csv --outdir figs --overlay-intensity

# Make an animation of the path
python visualize_langevin.py --traj traj.csv --outdir figs --animate

What it does
------------
- Loads CSV with columns: t_s,x,y,Fx_meep,Fy_meep
- Makes: path (x,y), x(t), y(t), Fx(t), Fy(t), speed(t)
- Computes MSD(tau) and radial histogram p(r)
- Optional: overlays trajectory on |E|^2 (trap w/o particle) using FocusedGaussPair.py
- Optional: saves an MP4 animation of the trajectory
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

def load_traj_csv(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    t = arr[:,0]
    x = arr[:,1]
    y = arr[:,2]
    Fx = arr[:,3]
    Fy = arr[:,4]
    return t, x, y, Fx, Fy

def plot_basic(t, x, y, Fx, Fy, outdir: Path, prefix=""):
    outdir.mkdir(parents=True, exist_ok=True)

    # Path
    plt.figure()
    plt.plot(x, y, lw=1)
    plt.scatter([x[0]], [y[0]], s=20, label="start")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trajectory")
    plt.axis("equal"); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{prefix}path.png", dpi=200)
    plt.close()

    # Time-series x(t), y(t)
    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, y, label="y(t)")
    plt.xlabel("t [s]")
    plt.ylabel("position")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{prefix}pos_vs_time.png", dpi=200)
    plt.close()

    # Forces
    plt.figure()
    plt.plot(t, Fx, label="Fx")
    plt.plot(t, Fy, label="Fy")
    plt.xlabel("t [s]")
    plt.ylabel("Force [Meep units]")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"{prefix}force_vs_time.png", dpi=200)
    plt.close()

    # Speed
    dt = np.gradient(t)
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = np.sqrt(vx*vx + vy*vy)
    plt.figure()
    plt.plot(t, speed)
    plt.xlabel("t [s]")
    plt.ylabel("speed [units/s]")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}speed_vs_time.png", dpi=200)
    plt.close()

def msd(t, x, y, max_lags=200):
    n = len(t)
    lags = np.arange(1, min(max_lags, n-1))
    taus = t[lags] - t[0]
    msd_vals = np.zeros_like(lags, dtype=float)
    for i, lag in enumerate(lags):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd_vals[i] = np.mean(dx*dx + dy*dy)
    return taus, msd_vals

def plot_msd(t, x, y, outdir: Path, prefix=""):
    taus, msd_vals = msd(t, x, y, max_lags=min(1000, len(t)//4))
    plt.figure()
    plt.loglog(taus, msd_vals, '-o', ms=3)
    plt.xlabel(r"$\tau$ [s]")
    plt.ylabel("MSD")
    plt.title("Mean-squared displacement")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}msd.png", dpi=200)
    plt.close()

def plot_radial_hist(x, y, outdir: Path, bins=50, prefix=""):
    r = np.sqrt((x-x.mean())**2 + (y-y.mean())**2)
    plt.figure()
    plt.hist(r, bins=bins, density=True)
    plt.xlabel("r")
    plt.ylabel("p(r)")
    plt.title("Radial histogram")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}radial_hist.png", dpi=200)
    plt.close()

def overlay_on_intensity(x, y, outdir: Path, prefix=""):
    # Recompute trap intensity w/o particle via FocusedGaussPair helper if available.
    try:
        import importlib
        FGP = importlib.import_module("FocusedGaussPair")
        import meep as mp
    except Exception as e:
        print("Overlay skipped (FocusedGaussPair or meep not importable):", e)
        return
    # Precompute |Ez|^2 like in the driver
    cfg = FGP.SimConfig()
    # Extract code block from driver to avoid circular import
    # Re-implement a small version here:
    cell = mp.Vector3(cfg.sx, cfg.sy, 0)
    src_freq = 1.0 / cfg.src_lambda
    fwidth = 0.3 * src_freq
    k0 = 2 * np.pi / cfg.src_lambda
    d = cfg.x_focus - cfg.x_src
    theta_max = np.arcsin(cfg.target_NA / cfg.n_medium)
    aperture_radius = abs(d) * np.tan(theta_max)

    try:
        import src_spherical_phase as src
        amp_func = src.make_spherical_phase(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)
    except Exception:
        amp_func = FGP.make_focused_amp_callable(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)

    src_region = mp.Vector3(0, cfg.sy - 2 * cfg.dpml, 0)
    sources = [
        mp.Source(mp.ContinuousSource(frequency=src_freq, fwidth=fwidth),
                  component=mp.Ez, center=mp.Vector3(cfg.x_src, -cfg.trap_sep/2, 0),
                  size=src_region, amp_func=amp_func),
        mp.Source(mp.ContinuousSource(frequency=src_freq, fwidth=fwidth),
                  component=mp.Ez, center=mp.Vector3(cfg.x_src, +cfg.trap_sep/2, 0),
                  size=src_region, amp_func=amp_func),
    ]
    block_center_x = 0.5*(cfg.xmax_block + cfg.xmin_block)
    block_size_x = abs(cfg.xmax_block - cfg.xmin_block)
    geometry = [mp.Block(size=mp.Vector3(block_size_x, cfg.sy, mp.inf),
                         center=mp.Vector3(block_center_x, 0, 0),
                         material=mp.Medium(index=cfg.n_water))]

    sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=sources,
                        boundary_layers=[mp.PML(cfg.dpml)], resolution=cfg.resolution,
                        dimensions=2, default_material=mp.Medium(index=cfg.n_medium))

    mon_center = mp.Vector3(0, 0, 0)
    mon_size = mp.Vector3(cfg.sx - 2*cfg.dpml, cfg.sy - 2*cfg.dpml, 0)
    dft = sim.add_dft_fields([mp.Ez], src_freq, 0, 1, center=mon_center, size=mon_size)
    sim.run(until=cfg.n_cycles_to_settle/src_freq)
    ez_f = sim.get_dft_array(dft, mp.Ez, 0)
    I = np.abs(ez_f)**2
    nx, ny = ez_f.shape[0], ez_f.shape[1]
    xx = np.linspace(-0.5*mon_size.x, 0.5*mon_size.x, nx)
    yy = np.linspace(-0.5*mon_size.y, 0.5*mon_size.y, ny)

    # Plot
    plt.figure()
    plt.pcolormesh(xx, yy, I.T, shading="auto")
    plt.plot(x, y, lw=1, c="k")
    plt.scatter([x[0]], [y[0]], s=10, c="k")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("|E|^2 (no particle) + trajectory")
    plt.axis("equal"); plt.tight_layout()
    plt.savefig(outdir / f"{prefix}path_on_intensity.png", dpi=200)
    plt.close()

def animate_path(t, x, y, outdir: Path, prefix="traj", fps=30):
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Trajectory (animated)")
    ax.axis("equal")
    line, = ax.plot([], [], lw=1)
    head, = ax.plot([], [], 'o', ms=4)
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))

    def init():
        line.set_data([], [])
        head.set_data([], [])
        return line, head

    def update(i):
        line.set_data(x[:i+1], y[:i+1])
        head.set_data(x[i], y[i])
        return line, head

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=1000/fps)
    # Try MP4 first
    mp4_path = outdir / f"{prefix}.mp4"
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(mp4_path, writer=writer, dpi=200)
        print(f"Saved animation: {mp4_path}")
    except Exception as e:
        # Fallback to GIF (requires ImageMagick or Pillow writer)
        gif_path = outdir / f"{prefix}.gif"
        ani.save(gif_path, writer="pillow", dpi=200)
        print(f"Saved animation: {gif_path}")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, required=True, help="trajectory CSV from langevin_driver.py")
    ap.add_argument("--outdir", type=str, default="figs")
    ap.add_argument("--all", action="store_true", help="make all static figures")
    ap.add_argument("--overlay-intensity", action="store_true", help="overlay path on trap |E|^2 (recomputed w/o particle)")
    ap.add_argument("--animate", action="store_true", help="save an MP4 (or GIF) animation")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    t, x, y, Fx, Fy = load_traj_csv(args.traj)

    if args.all:
        plot_basic(t, x, y, Fx, Fy, outdir)
        plot_msd(t, x, y, outdir)
        plot_radial_hist(x, y, outdir)

    if args.overlay_intensity:
        overlay_on_intensity(x, y, outdir)

    if args.animate:
        animate_path(t, x, y, outdir)

    # Always drop a quick summary
    # (Small text file with basic stats)
    with open(outdir / "summary.txt", "w") as f:
        f.write(f"Steps: {len(t)}\n")
        f.write(f"Total time [s]: {t[-1]-t[0]:.6g}\n")
        dr = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2)
        f.write(f"Net displacement: {dr:.6g}\n")

if __name__ == "__main__":
    main()
