import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sources as src

# Simulate a cell with dimensions (sx, sy, sz) with a spherical phase front entering from the bottom.

# --------------------------- Simulation Parameters ---------------------------
sx, sy, sz      = 24.0, 0, 16.0
cell            = mp.Vector3(sx, sy, sz)
resolution      = 10
dpml            = 1 # Anti reflection layers surrounding the cell

n_oil           = 1.518
n_water         = 1.333

# ----------------------------- Source Parameters -----------------------------
src_lambda      = 1.064 # free-space wavelength (um). All other units are in src_lambda
src_freq        = 1 / src_lambda
src_wavenumber  = 2 * np.pi / src_lambda
src_amp         = 1.0 # scale source field amplitude (intensity ~ amp^2)
src_center      = mp.Vector3(-2, 0, - sz/2 + dpml)
src_center2      = mp.Vector3(2, 0, - sz/2 + dpml)

src_NA          = 1.2 # Effective NA of a 1.42 oil immersion objective focusing into water

# compute required focus for the NA
theta_max = np.arcsin(src_NA / n_water)
aperture_radius = sx/2 - dpml - 2
focus = aperture_radius / np.tan(theta_max)

amp_func = src.make_spherical_phase(aperture_radius, src_wavenumber, focus, src_amp)

# ---------------------------- Geometry Parameters ----------------------------
sphere_radius   = 1.2 / src_lambda
sphere_center   = mp.Vector3(0, 0, 0)
sphere_material = mp.Medium(index=1.49)

sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=src_freq),
        component=mp.Ez,
        center=src_center,
        size=mp.Vector3(sx, 0, 0),
        amp_func=amp_func,
    ),
    mp.Source(
        src=mp.ContinuousSource(frequency=src_freq),
        component=mp.Ez,
        center=src_center2,
        size=mp.Vector3(sx, 0, 0),
        amp_func=amp_func,
    )
]

geometry = [    
    mp.Cylinder(
        radius    = sphere_radius,
        center    = sphere_center,
        height    = mp.inf,
        axis      = mp.Vector3(0,1,0),
        material  = sphere_material
    ),
    ]

sim = mp.Simulation(
    cell_size       = cell,
    # geometry        = geometry,
    resolution      = resolution,
    sources         = sources,
    boundary_layers = [mp.PML(dpml)],
    dimensions      = 3,
    default_material=mp.Medium(index=n_water)
)

# Record the frequency-domain Ez field at src_freq (smooth intensity)
mon_center = mp.Vector3(0, 0, 0)
mon_size   = mp.Vector3(sx - 2 * dpml, 0, sz - 2 * dpml)  # avoid PML in plots

dft = sim.add_dft_fields([mp.Ez], src_freq, 0, 1, center=mon_center, size=mon_size)

sim.run(until=100)

# Get the Ez field after the simulation
ez_data = sim.get_dft_array(dft, mp.Ez, 0)

I = np.abs(ez_data)**2

# I[:, 0:2] = 0

# Create coordinate grids for plotting
nx, nz = ez_data.shape[0], ez_data.shape[1]
z = np.linspace(-0.5 * mon_size.z, 0.5 * mon_size.z, nz)  # propagation
x = np.linspace(-0.5 * mon_size.x, 0.5 * mon_size.x, nx)  # transverse

plt.figure()
plt.pcolormesh(x, z, I.T, shading='auto', cmap='jet')
plt.tight_layout()
plt.gca().set_aspect('equal')
cbar = plt.colorbar()
cbar.set_label(rf'$\left|E_z(\lambda={src_lambda:.2f}\mu m)\right|^2$', fontsize=12)
plt.xlabel('x ('r'$\lambda$)')
plt.ylabel('z ('r'$\lambda$)')

# Overlay PML regions (shaded)
ax = plt.gca()
full_z_half = 0.5 * cell.z
full_x_half = 0.5 * cell.x

# Set axis limits to show full cell including PML
ax.set_xlim(-full_x_half, full_x_half)
ax.set_ylim(-full_z_half, full_z_half)

hatch_style = '///'
left_pml = patches.Rectangle((-full_x_half, -full_z_half), dpml, 2 * full_z_half,
                            linewidth=0, hatch=hatch_style, zorder=2)
right_pml = patches.Rectangle((full_x_half - dpml, -full_z_half), dpml, 2 * full_z_half,
                            linewidth=0, hatch=hatch_style, zorder=2)
bottom_pml = patches.Rectangle((-full_x_half, -full_z_half), 2 * full_x_half, dpml,
                            linewidth=0, hatch=hatch_style, zorder=2)
top_pml = patches.Rectangle((-full_x_half, full_z_half - dpml), 2 * full_x_half, dpml,
                            linewidth=0, hatch=hatch_style, zorder=2)
for p in (left_pml, right_pml, bottom_pml, top_pml):
    ax.add_patch(p)

sphere_circumference = 2 * np.pi * sphere_radius
theta = np.linspace(-np.pi, np.pi, int(resolution*sphere_circumference)) # Set the resolution based on the sphere circumference and sim resolution
plt.plot(sphere_center.x + sphere_radius * np.cos(theta), sphere_center.z + sphere_radius * np.sin(theta), 'w-', linewidth=2, label='Sphere Outline')

plt.show()
