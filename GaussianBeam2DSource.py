## launch a Gaussian beam
import math

import matplotlib

import meep as mp

import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt

%matplotlib inline


s = 14
resolution = 20
dpml = 2

cell_size = mp.Vector3(s, s)

boundary_layers = [mp.PML(thickness=dpml)]

beam_x0 = mp.Vector3(0, 5.0)  # beam focus (relative to source center)
rot_angle = 0  # CCW rotation angle about z axis (0: +y axis)
beam_kdir = mp.Vector3(0, 1, 0)# beam propagation direction
beam_w0 = 1  # beam waist radius
beam_E0 = mp.Vector3(0, 0, 1)
fcen = 1
sources = [
    mp.GaussianBeamSource(
        src=mp.ContinuousSource(fcen),
        center=mp.Vector3(0, -0.5 * s + dpml + 1.0),
        size=mp.Vector3(s),
        beam_x0=beam_x0,
        beam_kdir=beam_kdir,
        beam_w0=beam_w0,
        beam_E0=beam_E0,
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=boundary_layers,
    sources=sources,
)

sim.run(until=20)

# sim.plot2D(
#     fields=mp.Ez,
#     output_plane=mp.Volume(
#         center=mp.Vector3(), size=mp.Vector3(s - 2 * dpml, s - 2 * dpml)
#     ),
# )
ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(s - 2 * dpml, s - 2 * dpml), component=mp.Ez)
plt.figure(figsize=(6,6))
plt.imshow(np.abs(ez)**2, interpolation='spline36', cmap='jet', origin='lower')
plt.colorbar(label='|Ez|^2')
plt.title('Gaussian Beam Intensity')
plt.show()

# plt.savefig(f"Ez_angle{rot_angle}.png", bbox_inches="tight", pad_inches=0)