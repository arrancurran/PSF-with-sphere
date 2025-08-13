import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
wavelength = 1.0
frequency = 1 / wavelength
cell_x = 200
cell_y = 160
resolution = 50

# Lens parameters
lens_center_x = 0
lens_center_y = 0
lens_radius_front = 6.0   # Radius of curvature (front)
lens_radius_back = 3.0    # Radius of curvature (back)
lens_thickness = 4.0      # Thickness at lens center
lens_height = 160         # Height of lens (aperture)
lens_index = 1.5

# Define lens surfaces (bi-convex)
geometry = [
    # Front curved surface (arc)
    mp.Block(
        size=mp.Vector3(lens_thickness, lens_height),
        center=mp.Vector3(lens_center_x, lens_center_y),
        material=mp.Medium(index=lens_index)
    ),
    # mp.Cylinder(
    #     radius=lens_radius_front,
    #     center=mp.Vector3(lens_center_x - lens_radius_front + lens_thickness/2, lens_center_y),
    #     height=mp.inf,
    #     material=mp.air
    # ),
    # mp.Cylinder(
    #     radius=lens_radius_back,
    #     center=mp.Vector3(lens_center_x + lens_radius_back - lens_thickness/2, lens_center_y),
    #     height=mp.inf,
    #     material=mp.air
    # )
]

# Collimated plane wave source
sources = [
    mp.Source(
        src=mp.ContinuousSource(frequency=frequency),
        component=mp.Ez,
        center=mp.Vector3(-7, 0),
        size=mp.Vector3(0, cell_y)
    )
]

cell = mp.Vector3(cell_x, cell_y, 0)

sim = mp.Simulation(
    cell_size=cell,
    resolution=resolution,
    geometry=geometry,
    sources=sources,
    boundary_layers=[mp.PML(1.0)],
    dimensions=2
)

sim.run(until=100)

# Get the Ez field after the simulation
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

# Create coordinate grids for plotting
x = np.linspace(-cell_x/2, cell_x/2, int(cell_x*resolution))
y = np.linspace(-cell_y/2, cell_y/2, int(cell_y*resolution))

plt.figure(figsize=(10, 6))
plt.pcolormesh(x, y, np.abs(ez_data.T)**2, shading='auto', cmap='jet')
plt.colorbar(label='|Ez|^2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Collimated Beam Focused by a Bi-Convex Lens')

# Overlay the lens outline (approximate)
theta = np.linspace(-np.pi/2, np.pi/2, 200)
front_x = lens_center_x - lens_radius_front + lens_thickness/2 + lens_radius_front * np.cos(theta)
front_y = lens_center_y + lens_radius_front * np.sin(theta)
back_x = lens_center_x + lens_radius_back - lens_thickness/2 - lens_radius_back * np.cos(theta)
back_y = lens_center_y + lens_radius_back * np.sin(theta)
# plt.plot(front_x, front_y, 'w--', linewidth=2, label='Lens Front')
# plt.plot(back_x, back_y, 'w--', linewidth=2, label='Lens Back')

plt.legend()
plt.show()