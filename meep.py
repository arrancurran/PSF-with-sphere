import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
wavelength = 1.0  # in arbitrary units
frequency = 1 / wavelength
n_sphere = 1.5
radius = 2.0
pml_thickness = 1.0
domain_size = mp.Vector3(16, 12)
resolution = 40  # pixels per unit

# Geometry

geometry = [
    mp.Cylinder(
        radius=2.0,           # radius of the cylinder
        height=mp.inf,        # infinite along z => 2D cross-section
        material=mp.Medium(index=1.5)
    )
]

# Source: Gaussian beam approximation
fcen = frequency
df = 0.1 * fcen
beam_width = 1.5

sources = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(-5, 0),
                     size=mp.Vector3(0, 4),
                     amplitude=1.0)]

# Simulation
sim = mp.Simulation(cell_size=domain_size,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution,
                    boundary_layers=[mp.PML(pml_thickness)])

sim.run(until=200)

# Extract field
eps = sim.get_array(center=mp.Vector3(), size=domain_size, component=mp.Dielectric)
ez = sim.get_array(center=mp.Vector3(), size=domain_size, component=mp.Ez)

# Plotting
plt.figure(figsize=(10, 6))
plt.imshow(np.rot90(np.abs(ez)**2), cmap='jet',
           extent=[-domain_size.x/2, domain_size.x/2,
                   -domain_size.y/2, domain_size.y/2])
circle = plt.Circle((0, 0), radius, color='white', fill=False)
plt.gca().add_artist(circle)
plt.colorbar(label='|E|²')
plt.xlabel('x (in λ)')
plt.ylabel('y (in λ)')
plt.title('Focused Beam on Dielectric Sphere')
plt.tight_layout()
plt.show()
