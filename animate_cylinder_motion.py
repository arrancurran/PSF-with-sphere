"""
Animate a cylinder moving along the propagation axis (z) in a 2D Meep simulation.
This script does NOT modify `FocusedGaussian2D.py`.

Behavior:
- Moves a cylinder center from z = -3 to z = +1 in steps of dz=0.1 (simulation units).
- For each step it creates a fresh 2D Meep Simulation, runs it to steady state at the
  frequency-domain monitor, extracts |Ez(fcen)|^2, renders a frame, and saves it.
- After finishing, frames are combined into an AVI using imageio/ffmpeg.

Notes:
- Writing AVI relies on ffmpeg being available on PATH. If ffmpeg is not present
  you can change the writer to save an mp4 or a gif instead.
- This is a relatively expensive operation (many Meep runs). Reduce `dz` or
  `resolution` to speed up testing.

Usage:
    python animate_cylinder_motion.py

"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import imageio
import meep as mp

# ---------------- user parameters -------------------------------------------------
z_start = -4.0
z_end = 2.0
dz = 0.1
output_dir = "frames_cylinder_move"
out_movie = "cylinder_move.mp4"  # H.264 mp4 for mac compatibility
fps = 10

# Simulation parameters (based on your existing FocusedGaussian2D.py)
resolution = 10               # reduce for a faster run if needed
lambda0 = 1.064
fcen = 1.0 / lambda0
fwidth = 0.3 * fcen

# cell and PML
sx, sy = 20.0, 20.0
dpml = 1.0
cell = mp.Vector3(sx, sy, 0)

# source/focus geometry (keep propagation along simulation x)
x_src = -8.0
x_focus = -4.0

w_src = 20.0
amp_scale = 1.0
k0 = 2 * np.pi / lambda0

# cylinder properties
sphere_radius = 1.2 / lambda0
# we'll set the moving cylinder center at (z_pos, 0, 0) in simulation coords

# monitor region (same as in FocusedGaussian2D.py)
mon_center = mp.Vector3(0, 0, 0)
mon_size = mp.Vector3(sx - 2 * dpml, sy - 2 * dpml, 0)

# make frames dir
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# helper: create a focused beam amp func (same as in your script)
# using a smooth aperture computed from a target NA (optional)

target_NA = 1.42
n_medium = 1.518
if target_NA > n_medium:
    raise ValueError("target_NA must be <= n_medium")

d = x_focus - x_src
theta_max = np.arcsin(target_NA / n_medium)
aperture_radius = abs(d) * np.tan(theta_max)

def make_focused_amp(w_src=w_src, amp_scale=amp_scale, aperture_radius=aperture_radius):
    def focused_beam_amp(pt: mp.Vector3) -> complex:
        y = pt.y
        amp = np.exp(-(y / w_src) ** 2)
        edge_sigma = max(0.01, 0.05 * aperture_radius)
        edge = 1.0 / (1.0 + np.exp((abs(y) - aperture_radius) / edge_sigma))
        amp *= edge
        phase = np.exp(-1j * k0 * (np.sqrt(d * d + y * y) - d))
        return complex(amp_scale * amp * phase)
    return focused_beam_amp

focused_amp = make_focused_amp()

# iterate positions
z_positions = np.arange(z_start, z_end + 1e-12, dz)
frame_files = []

print(f"Running {len(z_positions)} frames ({z_start} -> {z_end} step {dz})")
# Whether to run a first pass to determine the global maximum intensity across
# all frames. If True, the script will run each simulation twice (first pass
# only computes max(I), second pass renders frames). This ensures a constant
# colormap scale but doubles runtime. Set to False to use the first frame's
# peak as vmax.
compute_global_vmax = True
global_vmax = None
if compute_global_vmax:
    print("Computing global vmax (first pass) — this will run all simulations once to find max intensity")
    for i, zpos in enumerate(z_positions):
        # build simulation geometry for this frame
        pml_layers = [mp.PML(dpml)]
        src_region = mp.Vector3(0, sy - 2 * dpml, 0)
        sources = [
            mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                      component=mp.Ez,
                      center=mp.Vector3(x_src, -0.0, 0),
                      size=src_region,
                      amp_func=focused_amp),
        ]

        geometry = [
            mp.Block(size=mp.Vector3(abs(10.0 - -5.0), sy, mp.inf),
                     center=mp.Vector3((10.0 + -5.0) / 2.0, 0, 0),
                     material=mp.Medium(index=1.33)),
            mp.Cylinder(radius=sphere_radius,
                        center=mp.Vector3(zpos, 0, 0),
                        height=mp.inf,
                        material=mp.Medium(index=1.49)),
        ]

        sim = mp.Simulation(cell_size=cell,
                            geometry=geometry,
                            sources=sources,
                            boundary_layers=pml_layers,
                            resolution=resolution,
                            dimensions=2,
                            default_material=mp.Medium(index=n_medium))
        dft = sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=mon_center, size=mon_size)
        sim.run(until=100)
        ez_f = sim.get_dft_array(dft, mp.Ez, 0)
        I = np.abs(ez_f) ** 2
        local_max = float(I.max())
        if global_vmax is None or local_max > global_vmax:
            global_vmax = local_max
    print(f"Global vmax = {global_vmax:.6e}")
for i, zpos in enumerate(z_positions):
    print(f"Frame {i+1}/{len(z_positions)}: cylinder at z={zpos:.3f}")

    # build simulation geometry for this frame
    pml_layers = [mp.PML(dpml)]

    src_region = mp.Vector3(0, sy - 2 * dpml, 0)
    sources = [
        mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                  component=mp.Ez,
                  center=mp.Vector3(x_src, -0.0, 0),
                  size=src_region,
                  amp_func=focused_amp),
    ]

    geometry = [
        mp.Block(size=mp.Vector3(abs(10.0 - -5.0), sy, mp.inf),
                 center=mp.Vector3((10.0 + -5.0) / 2.0, 0, 0),
                 material=mp.Medium(index=1.33)),
        mp.Cylinder(radius=sphere_radius,
                    center=mp.Vector3(zpos, 0, 0),
                    height=mp.inf,
                    material=mp.Medium(index=1.49)),
    ]

    sim = mp.Simulation(cell_size=cell,
                        geometry=geometry,
                        sources=sources,
                        boundary_layers=pml_layers,
                        resolution=resolution,
                        dimensions=2,
                        default_material=mp.Medium(index=n_medium))

    # add DFT monitor and run
    dft = sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=mon_center, size=mon_size)
    sim.run(until=100)
    ez_f = sim.get_dft_array(dft, mp.Ez, 0)
    I = np.abs(ez_f) ** 2

    # build plot (map simulation x->z vertical and y->x horizontal as in your script)
    nx, ny = I.shape[0], I.shape[1]
    z = np.linspace(-0.5 * mon_size.x, 0.5 * mon_size.x, nx)
    x = np.linspace(-0.5 * mon_size.y, 0.5 * mon_size.y, ny)

    fig, ax = plt.subplots(figsize=(6, 4))
    # use a fixed colormap scale so colorbar is consistent across frames
    if compute_global_vmax and global_vmax is not None:
        pcm = ax.pcolormesh(x, z, I, shading='auto', cmap='jet', vmin=0.0, vmax=global_vmax)
    else:
        # fallback: scale to the first frame's max (or auto)
        pcm = ax.pcolormesh(x, z, I, shading='auto', cmap='jet')
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(rf'$|E_z|^2$')

    # draw interface line at z = -5.0 (water above, oil below) similar to your overlay
    interface_z = -5.0
    ax.hlines(interface_z, -0.5 * sy + dpml, 0.5 * sy - dpml, colors='cyan', linewidth=2, zorder=6)
    x_text = 0.5 * sy - dpml - 0.5
    ax.text(x_text, interface_z + 0.4, 'water', color='cyan', fontsize=10, ha='right', va='bottom')
    ax.text(x_text, interface_z - 0.6, 'oil', color='white', fontsize=10, ha='right', va='top')

    # plot cylinder outline (mapped coords)
    theta = np.linspace(-np.pi, np.pi, 200)
    ax.plot(0 + sphere_radius * np.cos(theta), zpos + sphere_radius * np.sin(theta), 'w-', linewidth=1.5)

    ax.set_xlabel('x (transverse)')
    ax.set_ylabel('z (propagation)')
    ax.set_title(f'Cylinder z={zpos:.2f}')
    ax.set_aspect('equal')
    plt.tight_layout()

    frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)
    frame_files.append(frame_path)

# assemble AVI
print(f"Writing movie to {out_movie} (fps={fps})")
# Use H.264 (libx264) with yuv420p pixel format for broad mac/QuickTime compatibility.
try:
    with imageio.get_writer(out_movie, fps=fps, codec='libx264', ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
        for fname in frame_files:
            image = imageio.imread(fname)
            writer.append_data(image)
except Exception:
    # fallback: try without codec params (depends on imageio/ffmpeg installation)
    with imageio.get_writer(out_movie, fps=fps) as writer:
        for fname in frame_files:
            image = imageio.imread(fname)
            writer.append_data(image)

print("Done. Frames in:", output_dir)
print("Movie:", out_movie)
