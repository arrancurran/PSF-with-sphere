import meep as mp
import numpy as np

# Simulation parameters
cell = mp.Vector3(16, 8, 0)  # 2D cell: 16x8 units, z=0 for 2D
geometry = []  # No objects, just free space

# Source: a continuous wave (CW) at frequency 0.15, placed at x=2
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,  # Out-of-plane electric field
                     center=mp.Vector3(2, 0))]

# Boundary conditions: perfectly matched layers (PML) to absorb outgoing waves
pml_layers = [mp.PML(1.0)]

# Create simulation object
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    sources=sources,
                    boundary_layers=pml_layers,
                    resolution=10)  # 10 pixels per unit length

# Run the simulation for 200 time units
sim.run(until=200)

# To visualize, you can use sim.plot2D() or output data for plotting
x_coords = np.linspace(-cell.x/2, cell.x/2, 100)
y_coords = np.linspace(-cell.y/2, cell.y/2, 100)
Ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.imshow(np.abs(Ez_data)**2, extent=[-cell.x/2, cell.x/2, -cell.y/2, cell.y/2],
           origin='lower', cmap='jet')
plt.colorbar(label='|Ez|^2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Electric Field Intensity |Ez|^2')
plt.show()
