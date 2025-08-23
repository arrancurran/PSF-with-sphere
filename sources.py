import numpy as np
import meep as mp

def spherical_phase(pt: mp.Vector3, aperture_radius, src_wavenumber, focus, amp_scale) -> complex:
    x = pt.x  # Use x (transverse)
    # base Gaussian envelope
    amp = np.exp(-(x / aperture_radius) ** 2)
    phase = np.exp(-1j * src_wavenumber * (np.sqrt(focus**2 + x**2) - focus))
    return complex(amp_scale * amp * phase)


def make_spherical_phase(aperture_radius, src_wavenumber, focus, amp_scale):
    """Return a callable amp_func(pt: mp.Vector3) -> complex bound to these params."""
    def amp_func(pt: mp.Vector3) -> complex:
        return spherical_phase(pt, aperture_radius, src_wavenumber, focus, amp_scale)
    return amp_func