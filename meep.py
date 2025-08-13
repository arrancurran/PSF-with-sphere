import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Beam / optical parameters
wavelength = 1.0
n_medium = 1.0
NA = 0.8
k0 = 2 * np.pi / wavelength
theta_max = np.arcsin(NA / n_medium)

# Simulation cell
cell_size = mp.Vector3(16, 16, 0)
resolution = 40
pml_thickness = 1.0

# Cylinder (infinite along z, radius=2)
geometry = [
    mp.Cylinder(radius=2.0, height=mp.inf,
                material=mp.Medium(index=1.5))
]

# --- Generate high-NA field profile in a source plane ---
def debye_wolf_field(x_coords):
    """Return Ez(x) at source plane y = y0 for high-NA beam."""
    field = np.zeros_like(x_coords, dtype=np.complex128)
    for i, x in enumerate(x_coords):
        # Integrate over theta for given x
        thetas = np.linspace(0, theta_max, 200)
        integrand = np.sin(thetas) * np.exp(1j * k0 * x * np.sin(thetas))
        field[i] = np.trapz(integrand, thetas)
    # Normalize
    field /= np.max(np.abs(field))
    return field

<<<<<<< HEAD
sources = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(0, 0),
                     size=mp.Vector3(0, 4),
                     amplitude=1.0)]
=======
# Source plane sampling
src_x = np.linspace(-cell_size.x/2, cell_size.x/2, int(cell_size.x*resolution))
Ez_profile = debye_wolf_field(src_x)
>>>>>>> 43858928ede56fbad67f6a1236d1e0afabda4054

# --- Wrap into a MEEP custom source ---
def src_time(t):
    # Gaussian temporal profile
    fwidth = 0.2 / wavelength
    return np.exp(- (t - 20)**2 / (2*(5**2))) * np.cos(2*np.pi*t / wavelength)

def src_func(p):
    # Map simulation position to our Ez profile
    ix = int((p.x + cell_size.x/2) * resolution)
    if 0 <= ix < len(Ez_profile):
        return np.real(Ez_profile[ix])
    return 0.0

sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=1/wavelength),
        component=mp.Ez,
        center=mp.Vector3(0, -6),   # source plane before focus
        size=mp.Vector3(cell_size.x, 0),
        amp_func=src_func
    )
]

# --- Run simulation ---
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    boundary_layers=[mp.PML(pml_thickness)],
    dimensions=2
)

sim.run(until=200)

# --- Extract and plot ---
ez = sim.get_array(center=mp.Vector3(), size=cell_size, component=mp.Ez)
plt.figure(figsize=(8, 8))
plt.imshow(np.rot90(np.abs(ez)**2), cmap='jet',
           extent=[-cell_size.x/2, cell_size.x/2,
                   -cell_size.y/2, cell_size.y/2])
plt.colorbar(label='|E|²')
plt.xlabel('x (λ)')
plt.ylabel('y (λ)')
plt.title(f'High-NA (NA={NA}) Focused Beam on Dielectric Cylinder')
plt.show()