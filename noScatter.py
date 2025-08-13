import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# ---------- optics / units ----------
lam = 1.0                 # choose λ=1 as the unit
n_medium = 1.0
NA = 0.8
k0 = 2*np.pi/lam
theta_max = np.arcsin(NA/n_medium)

# ---------- domain ----------
res = 40                  # px per λ
sx, sy = 16, 16           # cell size (λ)
pml = 1.0
cell = mp.Vector3(sx, sy, 0)

# ---------- Debye–Wolf-ish scalar field at source plane ----------
def debye_scalar_Ex(x_arr):
    # field vs x at the source plane (y = y_src), scalar Ez-polarized 2D model
    thetas = np.linspace(0, theta_max, 400)
    E = np.zeros_like(x_arr, dtype=complex)
    for i, x in enumerate(x_arr):
        # simple pupil amplitude = 1, aplanatic weight ~ sinθ
        integrand = np.sin(thetas) * np.exp(1j * k0 * x * np.sin(thetas))
        E[i] = np.trapz(integrand, thetas)
    E /= np.max(np.abs(E))
    return E

# sample the source profile over the x-extent of the source
Nx = int(sx * res)
xs = np.linspace(-sx/2, sx/2, Nx)
Ez_profile = debye_scalar_Ex(xs).astype(np.complex128)

# map simulation point -> amplitude at the plane
def amp_func(p):
    ix = int((p.x + sx/2) * res)
    if 0 <= ix < Ez_profile.size:
        # inject real part; MEEP steps the time dependence via src (ContinuousSource)
        return float(np.real(Ez_profile[ix]))
    return 0.0

# ---------- sources ----------
y_src = -6.0  # put the source a few λ before the focus at y=0
sources = [mp.Source(src=mp.ContinuousSource(frequency=1/lam),
                     component=mp.Ez,
                     center=mp.Vector3(0, y_src),
                     size=mp.Vector3(sx, 0, 0),
                     amp_func=amp_func)]

# ---------- simulation (no geometry) ----------
sim = mp.Simulation(cell_size=cell,
                    geometry=[],                 # <-- removed cylinder
                    sources=sources,
                    resolution=res,
                    boundary_layers=[mp.PML(pml)],
                    dimensions=2)

sim.run(until=200)

# ---------- field dump ----------
ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
Iy = np.abs(ez)**2

plt.figure(figsize=(7,7))
plt.imshow(np.rot90(Iy), extent=[-sx/2, sx/2, -sy/2, sy/2], cmap='jet')
plt.xlabel('x (λ)'); plt.ylabel('y (λ)')
plt.title(f'Focused beam (scalar Debye, NA={NA})')
plt.colorbar(label='|E|²')
plt.tight_layout(); plt.show()

# optional: inspect on-axis intensity vs y
y_coords = np.linspace(-sy/2, sy/2, int(sy*res))
x0_idx = Iy.shape[1]//2
plt.figure()
plt.plot(y_coords, Iy[:, x0_idx])
plt.xlabel('y (λ)'); plt.ylabel('On-axis |E|²')
plt.title('Axial profile (focus near y=0)')
plt.tight_layout(); plt.show()