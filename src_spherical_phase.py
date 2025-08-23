import numpy as np
import meep as mp


def spherical_phase(x: mp.Vector3, w_src, aperture_radius, src_wavenumber, d, amp_scale) -> complex:
    y = x.y
    # base Gaussian envelope
    amp = np.exp(-(y / w_src) ** 2)
    # smooth aperture (soft roll-off at the edge to reduce diffraction)
    # edge_sigma = max(0.01, 0.05 * aperture_radius)
    # edge = 1.0 / (1.0 + np.exp((abs(y) - aperture_radius) / edge_sigma))
    # amp *= edge
    phase = np.exp(-1j * src_wavenumber * (np.sqrt(d * d + y * y) - d))
    
    return complex(amp_scale * amp * phase)


def make_spherical_phase(w_src, aperture_radius, src_wavenumber, d, amp_scale):
    """Return a callable amp_func(pt: mp.Vector3) -> complex bound to these params."""
    def amp_func(pt: mp.Vector3) -> complex:
        return spherical_phase(pt, w_src, aperture_radius, src_wavenumber, d, amp_scale)
    return amp_func