#!/usr/bin/env python3
"""
brownian_runner_simple.py — Minimal Langevin-like loop using optical forces from FocusedGaussPair,
with tunable Brownian force acting independently on x and y and per-step PNGs.

Key features
------------
- Brownian force acts on BOTH axes with independent Gaussian draws.
- Two scaling modes:
  * --brown-mode rel (default): sigma = brown_rel * |F_opt(0,0)| * brown_scale
  * --brown-mode abs:  sigma = brown_abs * brown_scale (constant)
- Saves a PNG each step with |E|^2 background + cylinder + arrows:
    optical (white), Brownian (cyan), total (lime).
- Optional console diagnostics every N steps.

Usage
-----
python brownian_runner_simple.py --steps 200 --dt 0.1 \
  --brown-mode rel --brown-rel 1e-3 --brown-scale 1.0 \
  --outdir frames_restart --seed 1 --print-every 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import FocusedGaussPair (spelling tolerant)
try:
    import FocusedGaussPair as FGP
except ImportError:
    try:
        import FocussedGaussPair as FGP
    except ImportError as e:
        print("Could not import FocusedGaussPair/FocussedGaussPair. Make sure the file is on PYTHONPATH.", file=sys.stderr)
        raise

import meep as mp


def make_amp(cfg):
    src_freq = 1.0 / cfg.src_lambda
    k0 = 2 * np.pi / cfg.src_lambda
    d = cfg.x_focus - cfg.x_src
    theta_max = np.arcsin(cfg.target_NA / cfg.n_medium)
    aperture_radius = abs(d) * np.tan(theta_max)

    try:
        import src_spherical_phase as src
        return src.make_spherical_phase(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)
    except ImportError:
        return FGP.make_focused_amp_callable(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)


def build_sim(cfg, x_c: float, y_c: float):
    cell = mp.Vector3(cfg.sx, cfg.sy, 0)
    src_freq = 1.0 / cfg.src_lambda
    fwidth = 0.3 * src_freq
    amp_func = make_amp(cfg)
    pml_layers = [mp.PML(cfg.dpml)]

    # Two sources (±y)
    src_region = mp.Vector3(0, cfg.sy - 2 * cfg.dpml, 0)
    sources = [
        mp.Source(mp.ContinuousSource(frequency=src_freq, fwidth=fwidth),
                  component=mp.Ez, center=mp.Vector3(cfg.x_src, -cfg.trap_sep/2, 0),
                  size=src_region, amp_func=amp_func),
        mp.Source(mp.ContinuousSource(frequency=src_freq, fwidth=fwidth),
                  component=mp.Ez, center=mp.Vector3(cfg.x_src, +cfg.trap_sep/2, 0),
                  size=src_region, amp_func=amp_func),
    ]

    block_center_x = 0.5 * (cfg.xmax_block + cfg.xmin_block)
    block_size_x = abs(cfg.xmax_block - cfg.xmin_block)
    geometry = [
        mp.Block(size=mp.Vector3(block_size_x, cfg.sy, mp.inf),
                 center=mp.Vector3(block_center_x, 0, 0),
                 material=mp.Medium(index=cfg.n_water)),
        mp.Cylinder(radius=cfg.sphere_radius, center=mp.Vector3(x_c, y_c, 0),
                    height=mp.inf, material=mp.Medium(index=cfg.n_sphere)),
    ]

    sim = mp.Simulation(cell_size=cell, geometry=geometry, sources=sources,
                        boundary_layers=pml_layers, resolution=cfg.resolution,
                        dimensions=2, default_material=mp.Medium(index=cfg.n_medium))
    return sim, src_freq


def compute_force(cfg, x: float, y: float):
    sim, src_freq = build_sim(cfg, x, y)
    sim.run(until=cfg.n_cycles_to_settle / src_freq)
    Fx, Fy = FGP.compute_mst_force_2d_tm(
        sim=sim,
        cyl_center=mp.Vector3(x, y, 0),
        cyl_radius=cfg.sphere_radius,
        n_bg=cfg.n_water,
        samples_per_cycle=cfg.samples_per_cycle,
        n_cycles_avg=cfg.n_cycles_avg,
    )
    return float(Fx), float(Fy)


def compute_field_and_save(cfg, x, y, F_opt, F_brown, F_tot, outfile):
    sim, src_freq = build_sim(cfg, x, y)
    mon_center = mp.Vector3(0, 0, 0)
    mon_size = mp.Vector3(cfg.sx - 2 * cfg.dpml, cfg.sy - 2 * cfg.dpml, 0)
    dft = sim.add_dft_fields([mp.Ez], src_freq, 0, 1, center=mon_center, size=mon_size)
    sim.run(until=cfg.n_cycles_to_settle / src_freq)

    ez_f = sim.get_dft_array(dft, mp.Ez, 0)
    I = np.abs(ez_f)**2
    nx, ny = ez_f.shape
    xx = np.linspace(-0.5*mon_size.x, 0.5*mon_size.x, nx)
    yy = np.linspace(-0.5*mon_size.y, 0.5*mon_size.y, ny)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xx, yy, I.T, shading="auto")
    ax.add_patch(patches.Circle((x, y), cfg.sphere_radius, fill=False, ec="w", lw=1.5))

    # arrows
    arrow_scale = 5
    ax.arrow(x, y, F_opt[0]*arrow_scale, F_opt[1]*arrow_scale, color="w",
             length_includes_head=True, head_width=0.1)
    ax.arrow(x, y, F_brown[0]*arrow_scale, F_brown[1]*arrow_scale, color="c",
             length_includes_head=True, head_width=0.1)
    ax.arrow(x, y, F_tot[0]*arrow_scale, F_tot[1]*arrow_scale, color="lime",
             length_includes_head=True, head_width=0.1)

    ax.set_aspect("equal")
    ax.set_xlim(xx[0], xx[-1]); ax.set_ylim(yy[0], yy[-1])
    ax.set_title("|E|^2 with cylinder; arrows: opt (white), brown (cyan), total (lime)")
    fig.colorbar(pcm, ax=ax, label="|Ez|^2")
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def reflect(val, lo, hi):
    width = hi - lo
    v = (val - lo) % (2 * width)
    if v > width:
        v = 2 * width - v
    return lo + v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--brown-mode", choices=["rel","abs"], default="rel")
    ap.add_argument("--brown-rel", type=float, default=1e-3)
    ap.add_argument("--brown-abs", type=float, default=1e-3)
    ap.add_argument("--brown-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="frames_simpler")
    ap.add_argument("--print-every", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = FGP.SimConfig()
    r = np.array([0.0, 0.0], float)

    # Get reference optical force magnitude at start
    Fx0, Fy0 = compute_force(cfg, r[0], r[1])
    F_ref = max(1e-9, np.hypot(Fx0, Fy0))
    if args.brown_mode == "rel":
        sigma = args.brown_rel * F_ref * args.brown_scale
    else:
        sigma = args.brown_abs * args.brown_scale

    if args.print_every:
        print(f"Initial optical F=({Fx0:.2e},{Fy0:.2e}), |F|={F_ref:.2e}, Brownian σ={sigma:.2e}")

    # Domain bounds with a small safety margin (particle stays inside water slab)
    x_min = cfg.xmin_block + 1.05*cfg.sphere_radius
    x_max = cfg.xmax_block - 1.05*cfg.sphere_radius
    y_min = -0.5*cfg.sy + cfg.dpml + 1.05*cfg.sphere_radius
    y_max =  0.5*cfg.sy - cfg.dpml - 1.05*cfg.sphere_radius

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Initial frame
    F_opt = np.array([Fx0, Fy0])
    F_brown = np.zeros(2)
    F_tot = F_opt + F_brown
    compute_field_and_save(cfg, r[0], r[1], F_opt, F_brown, F_tot, str(outdir / f"frame_{0:04d}.png"))

    # Loop
    for i in range(1, args.steps):
        Fx, Fy = compute_force(cfg, r[0], r[1])
        F_opt = np.array([Fx, Fy])

        # Brownian force — independent draws on x and y
        F_brown = sigma * rng.standard_normal(2)

        F_tot = F_opt + F_brown
        r_next = r + args.dt * F_tot

        # Reflect into bounds
        r_next[0] = reflect(r_next[0], x_min, x_max)
        r_next[1] = reflect(r_next[1], y_min, y_max)

        compute_field_and_save(cfg, r_next[0], r_next[1], F_opt, F_brown, F_tot,
                               str(outdir / f"frame_{i:04d}.png"))

        if args.print_every and i % args.print_every == 0:
            print(f"Step {i:04d}: r=({r_next[0]:+.3e},{r_next[1]:+.3e}) "
                  f"F_opt=({F_opt[0]:+.2e},{F_opt[1]:+.2e}) "
                  f"F_brown=({F_brown[0]:+.2e},{F_brown[1]:+.2e}) |F_brown|={np.linalg.norm(F_brown):.2e}")

        r = r_next

    print(f"Saved {args.steps} frames to {outdir.resolve()}")


if __name__ == "__main__":
    main()