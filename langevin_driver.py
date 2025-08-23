#!/usr/bin/env python3
"""
langevin_driver.py — 2D overdamped Langevin simulation using MST forces from FocusedGaussPair.py,
with optional per-step frame export (PSF background + cylinder + force arrow).

Stability-focused updates:
- Accepts --dt-meep (dimensionless Meep time); or --dt-ms / --dt-s converted to Meep time.
- Warns if dt_meep is too large (>> 1) and suggests smaller values (1e-4–1e-2).
- Optional --clip-step to cap |Δr| per step (in Meep length units) to avoid domain jumps.
- Reflecting boundary instead of hard clipping.

Typical usage
-------------
# Accurate optical forces each step (MST), save frames and use small Meep timestep
python langevin_driver.py --mode mst --steps 2000 --dt-meep 1e-3 \
  --start-x 0.0 --start-y 0.0 \
  --save-frames frames_mst --frame-dpi 160 --arrow-scale 0.2 --clip-step 0.2

# Fast gradient forces, long run, thin frames
python langevin_driver.py --mode gradi --steps 10000 --dt-meep 5e-3 \
  --alpha-eff 1e-3 --save-frames frames_gradi --frame-every 5
"""

from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Import your existing script as a module
import importlib
FGP = importlib.import_module("FocusedGaussPair")

try:
    import meep as mp  # ensure meep is available
except Exception as e:
    print("Error: meep must be importable in this environment.", file=sys.stderr)
    raise

KB = 1.380649e-23         # J/K
C0 = 299_792_458.0        # m/s


@dataclass
class BrownianParams:
    T_K: float = 293.0                # Temperature [K]
    eta_Pa_s: float = 1.0e-3          # Dynamic viscosity [Pa·s]
    delta_rho: float = 50.0           # ρ_particle - ρ_medium [kg/m^3] (buoyant mismatch)
    g: float = 9.81                   # gravity [m/s^2]
    force_unit_SI_per_length: float = 10  # [N/m] represented by 1 Meep force unit (per unit length). Calibrate!


def meep_unit_conversions(cfg: FGP.SimConfig) -> Tuple[float, float]:
    """
    Return (L_unit_m, T_unit_s) mapping for your sim:
      - Length unit in Meep is µm (you pass src_lambda in µm, positions in µm)
      - Time unit maps via c: T_unit = L_unit / c
    """
    L_unit_m = cfg.src_lambda * 1e-6  # 1 "wavelength" unit in meters
    T_unit_s = L_unit_m / C0
    return L_unit_m, T_unit_s


def physical_props_to_meep(cfg: FGP.SimConfig, brown: BrownianParams):
    """
    Compute diffusion coefficient D' and gravitational force Fg' in Meep units.
    We leave the mobility μ' implicit and work with SI μ in the update after unit handling.
    """
    L_unit_m, T_unit_s = meep_unit_conversions(cfg)

    # Particle radius in meters (cfg.sphere_radius is in µm-like Meep length units)
    a_m = cfg.sphere_radius * 1e-6

    # 3D Stokes drag for a sphere (order-of-magnitude, avoids 2D Stokes paradox)
    mu_SI = 1.0 / (6.0 * math.pi * brown.eta_Pa_s * a_m)      # mobility [m/(N·s)]
    D_SI = KB * brown.T_K * mu_SI                              # [m^2/s]

    # Convert D to Meep units: D' = D * (T_unit / L_unit^2)
    T_unit_s = L_unit_m / C0
    D_meep = D_SI * (T_unit_s / (L_unit_m ** 2))

    # Gravity per unit length → Meep force units
    Fg_per_length_SI = brown.delta_rho * brown.g * math.pi * (a_m ** 2)  # [N/m]
    Fg_meep = Fg_per_length_SI / brown.force_unit_SI_per_length

    return D_meep, mu_SI, Fg_meep


def build_sim_and_force(cfg: FGP.SimConfig, xy: Tuple[float, float]) -> Tuple[float, float]:
    """
    Build a Meep Simulation at cylinder center (x,y)=xy, run to steady state,
    and return the MST force (Fx,Fy) in Meep units (per unit length).
    """
    x_c, y_c = xy

    cell = mp.Vector3(cfg.sx, cfg.sy, 0)
    pml_layers = [mp.PML(cfg.dpml)]
    src_freq = 1.0 / cfg.src_lambda
    fwidth = 0.3 * src_freq
    k0 = 2 * np.pi / cfg.src_lambda
    d = cfg.x_focus - cfg.x_src

    # NA & aperture
    if cfg.target_NA > cfg.n_medium:
        raise ValueError(f"target_NA={cfg.target_NA} exceeds n_medium={cfg.n_medium}")
    theta_max = np.arcsin(cfg.target_NA / cfg.n_medium)
    aperture_radius = abs(d) * np.tan(theta_max)

    # Amplitude function
    try:
        import src_spherical_phase as src
        amp_func = src.make_spherical_phase(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)
    except Exception:
        amp_func = FGP.make_focused_amp_callable(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)

    # Two sources (±y)
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

    # Geometry: water slab + cylinder at (x_c, y_c)
    block_center_x = 0.5 * (cfg.xmax_block + cfg.xmin_block)
    block_size_x = abs(cfg.xmax_block - cfg.xmin_block)
    cyl_center = mp.Vector3(x_c, y_c, 0)

    geometry = [
        mp.Block(
            size=mp.Vector3(block_size_x, cfg.sy, mp.inf),
            center=mp.Vector3(block_center_x, 0, 0),
            material=mp.Medium(index=cfg.n_water),
        ),
        mp.Cylinder(
            radius=cfg.sphere_radius,
            center=cyl_center,
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

    # Run to steady state
    t_run = cfg.n_cycles_to_settle / src_freq
    sim.run(until=t_run)

    Fx, Fy = FGP.compute_mst_force_2d_tm(
        sim=sim,
        cyl_center=cyl_center,
        cyl_radius=cfg.sphere_radius,
        n_bg=cfg.n_water,
        samples_per_cycle=cfg.samples_per_cycle,
        n_cycles_avg=cfg.n_cycles_avg,
    )
    return Fx, Fy


def precompute_trap_intensity(cfg: FGP.SimConfig):
    """
    Compute |Ez|^2 of the trap *without* the cylinder, for drawing as a background.
    Returns (x_grid, y_grid, I) with shapes (nx,), (ny,), (nx,ny).
    """
    cell = mp.Vector3(cfg.sx, cfg.sy, 0)
    src_freq = 1.0 / cfg.src_lambda
    fwidth = 0.3 * src_freq
    k0 = 2 * np.pi / cfg.src_lambda
    d = cfg.x_focus - cfg.x_src

    # NA & aperture
    if cfg.target_NA > cfg.n_medium:
        raise ValueError(f"target_NA={cfg.target_NA} exceeds n_medium={cfg.n_medium}")
    theta_max = np.arcsin(cfg.target_NA / cfg.n_medium)
    aperture_radius = abs(d) * np.tan(theta_max)

    # Amplitude
    try:
        import src_spherical_phase as src
        amp_func = src.make_spherical_phase(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)
    except Exception:
        amp_func = FGP.make_focused_amp_callable(cfg.w_src, aperture_radius, k0, d, cfg.amp_scale)

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

    # Geometry: water slab only (no particle)
    block_center_x = 0.5 * (cfg.xmax_block + cfg.xmin_block)
    block_size_x = abs(cfg.xmax_block - cfg.xmin_block)
    geometry = [
        mp.Block(
            size=mp.Vector3(block_size_x, cfg.sy, mp.inf),
            center=mp.Vector3(block_center_x, 0, 0),
            material=mp.Medium(index=cfg.n_water),
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=[mp.PML(cfg.dpml)],
        resolution=cfg.resolution,
        dimensions=2,
        default_material=mp.Medium(index=cfg.n_medium),
    )

    mon_center = mp.Vector3(0, 0, 0)
    mon_size = mp.Vector3(cfg.sx - 2 * cfg.dpml, cfg.sy - 2 * cfg.dpml, 0)
    dft = sim.add_dft_fields([mp.Ez], src_freq, 0, 1, center=mon_center, size=mon_size)

    t_run = cfg.n_cycles_to_settle / src_freq
    sim.run(until=t_run)

    ez_f = sim.get_dft_array(dft, mp.Ez, 0)
    I = np.abs(ez_f) ** 2

    nx, ny = ez_f.shape[0], ez_f.shape[1]
    x = np.linspace(-0.5 * mon_size.x, 0.5 * mon_size.x, nx)  # horizontal
    y = np.linspace(-0.5 * mon_size.y, 0.5 * mon_size.y, ny)  # vertical
    return x, y, I


def interp_force_from_gradI(xg, yg, I, xy, alpha_eff) -> Tuple[float, float]:
    """
    Approximate optical force via Rayleigh-like gradient: F ≈ alpha_eff ∇|E|^2.
    """
    x, y = xy
    # Finite-difference gradients on grid
    dIdx = np.gradient(I, xg, axis=0)
    dIdy = np.gradient(I, yg, axis=1)

    # Bilinear interpolation
    def interp2(Z, xg, yg, x, y):
        ix = np.searchsorted(xg, x) - 1
        iy = np.searchsorted(yg, y) - 1
        ix = np.clip(ix, 0, len(xg) - 2)
        iy = np.clip(iy, 0, len(yg) - 2)
        x1, x2 = xg[ix], xg[ix+1]
        y1, y2 = yg[iy], yg[iy+1]
        wx = (x - x1) / (x2 - x1 + 1e-18)
        wy = (y - y1) / (y2 - y1 + 1e-18)
        z11 = Z[ix, iy]
        z21 = Z[ix+1, iy]
        z12 = Z[ix, iy+1]
        z22 = Z[ix+1, iy+1]
        return (1-wx)*(1-wy)*z11 + wx*(1-wy)*z21 + (1-wx)*wy*z12 + wx*wy*z22

    Fx = alpha_eff * interp2(dIdx, xg, yg, x, y)
    Fy = alpha_eff * interp2(dIdy, xg, yg, x, y)
    return float(Fx), float(Fy)


def save_frame(idx: int,
               r: np.ndarray,
               F: np.ndarray,
               cfg: FGP.SimConfig,
               xg: Optional[np.ndarray],
               yg: Optional[np.ndarray],
               I: Optional[np.ndarray],
               folder: Path,
               dpi: int = 160,
               arrow_scale: float = 0.2,
               annotate: bool = True) -> None:
    """
    Save a PNG showing PSF background (if available), cylinder at r, and force arrow.
    """
    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()

    # Background intensity if provided
    if xg is not None and yg is not None and I is not None:
        ax.pcolormesh(xg, yg, I.T, shading="auto")

    # Cylinder
    circle = patches.Circle((r[0], r[1]), cfg.sphere_radius, fill=False, linewidth=1.5)
    ax.add_patch(circle)

    # Force arrow (scaled for visibility)
    ax.arrow(r[0], r[1], F[0]*arrow_scale, F[1]*arrow_scale,
             length_includes_head=True, head_width=0.05*cfg.sphere_radius)

    # Limits
    if xg is not None and yg is not None:
        ax.set_xlim(xg[0], xg[-1])
        ax.set_ylim(yg[0], yg[-1])
    else:
        ax.set_xlim(cfg.xmin_block, cfg.xmax_block)
        ax.set_ylim(-0.5*cfg.sy+cfg.dpml, 0.5*cfg.sy-cfg.dpml)

    ax.set_aspect('equal', adjustable='box')

    if annotate:
        ax.set_title(f"frame {idx:04d}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(folder / f"frame_{idx:04d}.png", dpi=dpi)
    plt.close(fig)


def reflect_into_bounds(val: float, lo: float, hi: float) -> float:
    """Reflect a scalar into [lo, hi] (handles large overshoots)."""
    if lo >= hi:
        return min(max(val, lo), hi)
    width = hi - lo
    # Translate to 0..width, reflect with modulus
    v = (val - lo) % (2*width)
    if v > width:
        v = 2*width - v
    return lo + v


def simulate_langevin(cfg: FGP.SimConfig,
                      brown: BrownianParams,
                      steps: int,
                      dt_meep: float,
                      start_xy: Tuple[float, float],
                      mode: str = "mst",
                      alpha_eff: float = 0.0,
                      seed: Optional[int] = None,
                      save_frames: Optional[Path] = None,
                      frame_every: int = 1,
                      frame_dpi: int = 160,
                      arrow_scale: float = 0.2,
                      clip_step: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate r_{n+1} = r_n + μ_eff F_total Δt + sqrt(2DΔt) ξ with gravity along -x.
    We work in Meep length units. Time is passed already in Meep units (dimensionless).
    Returns (traj, forces).
    """
    rng = np.random.default_rng(seed)

    # Physical props
    L_unit_m, T_unit_s = meep_unit_conversions(cfg)
    D_meep, mu_SI, Fg_meep = physical_props_to_meep(cfg, brown)

    # Effective mobility from SI to Meep-step in position:
    # Convert Meep force to SI force-per-length, multiply by an effective length ℓ_eff (heuristic = L_unit_m)
    # to avoid the 2D per-length mismatch. Then convert Δr_SI to Meep units by /L_unit_m.
    # Net: Δr_meep = mu_SI * (F_meep * F_unit_SI_per_length * ℓ_eff) * (dt_meep * T_unit_s) / L_unit_m.
    # Choose ℓ_eff = L_unit_m ⇒ μ_eff = mu_SI * F_unit_SI_per_length * T_unit_s
    mu_eff = mu_SI * brown.force_unit_SI_per_length * T_unit_s  # [dimensionless length / (MeepForce * MeepTime)]

    # Precompute trap intensity once for frame background (both modes).
    xg = yg = I = None
    if save_frames is not None:
        try:
            xg, yg, I = precompute_trap_intensity(cfg)
        except Exception as e:
            print("Warning: could not precompute PSF background; frames will omit background.", e)

    r = np.array(start_xy, dtype=float)
    traj = np.zeros((steps, 2), dtype=float)
    traj[0] = r
    forces = np.zeros((steps, 2), dtype=float)

    # Domain limits (stay within water block, minus safety margin)
    x_min = cfg.xmin_block + 1.1 * cfg.sphere_radius
    x_max = cfg.xmax_block - 1.1 * cfg.sphere_radius
    y_min = -0.5 * cfg.sy + cfg.dpml + 1.1 * cfg.sphere_radius
    y_max = +0.5 * cfg.sy - cfg.dpml - 1.1 * cfg.sphere_radius

    # Save initial frame with zero force for reference
    if save_frames is not None:
        save_frame(0, r, np.array([0.0, 0.0]), cfg, xg, yg, I, folder=save_frames, dpi=frame_dpi, arrow_scale=arrow_scale)

    for i in range(steps - 1):
        if mode == "mst":
            Fx, Fy = build_sim_and_force(cfg, tuple(r))
        elif mode == "gradi":
            if xg is None or yg is None or I is None:
                xg, yg, I = precompute_trap_intensity(cfg)
            Fx, Fy = interp_force_from_gradI(xg, yg, I, tuple(r), alpha_eff=alpha_eff)
        else:
            raise ValueError("mode must be 'mst' or 'gradi'")

        # Gravity along -x
        Fx_tot = Fx - Fg_meep
        Fy_tot = Fy
        forces[i] = (Fx_tot, Fy_tot)

        # Euler–Maruyama update in Meep units
        drift = mu_eff * np.array([Fx_tot, Fy_tot]) * dt_meep
        noise = math.sqrt(2.0 * D_meep * dt_meep) * rng.standard_normal(2)
        step_vec = drift + noise

        # Optional step capping
        if clip_step is not None:
            step_norm = float(np.linalg.norm(step_vec))
            if step_norm > clip_step and step_norm > 0:
                step_vec *= (clip_step / step_norm)

        r_next = r + step_vec

        # Reflecting bounds
        r_next[0] = reflect_into_bounds(r_next[0], x_min, x_max)
        r_next[1] = reflect_into_bounds(r_next[1], y_min, y_max)

        traj[i+1] = r_next

        # Save frame at the current step+1, using the force evaluated at r (displayed at r_next for clarity)
        if save_frames is not None and ((i+1) % frame_every == 0):
            save_frame(i+1, r_next, np.array([Fx_tot, Fy_tot]), cfg, xg, yg, I,
                       folder=save_frames, dpi=frame_dpi, arrow_scale=arrow_scale)

        r = r_next

    return traj, forces


def main():
    ap = argparse.ArgumentParser(description="2D Langevin simulation (Meep MST or ∇|E|²) with per-step frame export; SI↔Meep unit handling made stable.")
    ap.add_argument("--mode", choices=["mst", "gradi"], default="mst", help="Force model: 'mst' (accurate) or 'gradi' (fast)")
    ap.add_argument("--steps", type=int, default=300, help="Number of Langevin time steps")

    # Time step options
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--dt-meep", type=float, default=None, help="Time step in Meep time units (dimensionless). Recommended 1e-4 to 1e-2.")
    g.add_argument("--dt-s", type=float, default=None, help="Time step in seconds (will be converted).")
    g.add_argument("--dt-ms", type=float, default=None, help="Time step in milliseconds (will be converted).")

    ap.add_argument("--start-x", type=float, default=0.0, help="Initial x position (Meep length units; same as your script)")
    ap.add_argument("--start-y", type=float, default=0.0, help="Initial y position (Meep length units)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")

    # Brownian/physical parameters
    ap.add_argument("--T-K", type=float, default=293.0, help="Temperature [K]")
    ap.add_argument("--eta", type=float, default=1.0e-3, help="Dynamic viscosity [Pa·s]")
    ap.add_argument("--delta-rho", type=float, default=50.0, help="Density contrast ρ_p-ρ_m [kg/m^3]")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity [m/s^2]")
    ap.add_argument("--force-unit-SI-per-length", type=float, default=1.0,
                    help="How many N/m correspond to 1 Meep force unit (per length). Calibrate this.")

    # Gradient model param
    ap.add_argument("--alpha-eff", type=float, default=1e-3, help="Effective polarizability scaling for gradient model (Meep force units)")

    # Output
    ap.add_argument("--out", type=str, default="traj.csv", help="CSV with columns: t_s,x,y,Fx,Fy")
    ap.add_argument("--plot", type=str, default="traj.png", help="PNG path for a quick path plot")

    # Frame export
    ap.add_argument("--save-frames", type=str, default=None, help="Directory to save PNG frames; if omitted, no frames are saved")
    ap.add_argument("--frame-every", type=int, default=1, help="Save every Nth frame (≥1)")
    ap.add_argument("--frame-dpi", type=int, default=160, help="DPI for saved frames")
    ap.add_argument("--arrow-scale", type=float, default=0.2, help="Multiply force by this to draw arrow length")
    ap.add_argument("--clip-step", type=float, default=None, help="Max allowed |Δr| per step (in Meep length units). If set, large steps are rescaled.")

    args = ap.parse_args()

    cfg = FGP.SimConfig()  # use your defaults; edit in FocusedGaussPair.py if desired
    brown = BrownianParams(T_K=args.T_K, eta_Pa_s=args.eta, delta_rho=args.delta_rho, g=args.g,
                           force_unit_SI_per_length=args.force_unit_SI_per_length)

    # Determine dt in Meep units
    L_unit_m, T_unit_s = meep_unit_conversions(cfg)
    if args.dt_meep is not None:
        dt_meep = args.dt_meep
        t_s = np.arange(args.steps) * (dt_meep * T_unit_s)
    elif args.dt_s is not None:
        dt_meep = args.dt_s / T_unit_s
        t_s = np.arange(args.steps) * args.dt_s
    elif args.dt_ms is not None:
        dt_meep = (args.dt_ms * 1e-3) / T_unit_s
        t_s = np.arange(args.steps) * (args.dt_ms * 1e-3)
    else:
        # Sensible default
        dt_meep = 1e-3
        t_s = np.arange(args.steps) * (dt_meep * T_unit_s)

    # Warnings for stability
    if dt_meep > 1e-1:
        print(f"WARNING: dt_meep={dt_meep:.3e} is very large; expect unstable jumps. Recommended 1e-4–1e-2.", file=sys.stderr)

    save_dir = Path(args.save_frames) if args.save_frames else None

    traj, forces = simulate_langevin(cfg, brown, steps=args.steps, dt_meep=dt_meep,
                                     start_xy=(args.start_x, args.start_y),
                                     mode=args.mode, alpha_eff=args.alpha_eff, seed=args.seed,
                                     save_frames=save_dir, frame_every=args.frame_every,
                                     frame_dpi=args.frame_dpi, arrow_scale=args.arrow_scale,
                                     clip_step=args.clip_step)

    # Save CSV with real (SI) time stamps for convenience
    arr = np.column_stack([t_s, traj[:,0], traj[:,1], forces[:,0], forces[:,1]])
    np.savetxt(args.out, arr, delimiter=",", header="t_s,x,y,Fx_meep,Fy_meep", comments="")

    # Quick path plot
    plt.figure()
    plt.plot(traj[:,0], traj[:,1], lw=1)
    plt.scatter([traj[0,0]], [traj[0,1]], label="start", s=20)
    plt.xlabel("x (Meep length units ≈ µm)")
    plt.ylabel("y (Meep length units ≈ µm)")
    plt.title(f"Langevin path ({args.mode}) — steps={args.steps}, dt_meep={dt_meep:g}")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(args.plot, dpi=200)

    print(f"dt_meep used = {dt_meep:.6g} (1 Meep time unit ≈ {T_unit_s:.3e} s at λ={cfg.src_lambda} µm)")
    print(f"Wrote {args.out} and {args.plot}")
    if save_dir is not None:
        print(f"Saved frames to: {save_dir.resolve()}")

if __name__ == "__main__":
    main()
