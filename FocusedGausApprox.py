import numpy as np
import matplotlib.pyplot as plt

# Parameters
wavelength = 1.0      # wavelength (arbitrary units)
w0 = 1.0              # beam waist at focus (y=0)
I0 = 1.0              # peak intensity

zR = np.pi * w0**2 / wavelength  # Rayleigh range

# Grid
x = np.linspace(-8, 8, 400)
y = np.linspace(-8, 8, 400)
X, Y = np.meshgrid(x, y)

# Beam width as a function of y
wY = w0 * np.sqrt(1 + (Y / zR)**2)

# Intensity profile
I = I0 * (w0 / wY)**2 * np.exp(-2 * X**2 / wY**2)

# Plot
plt.figure(figsize=(7, 6))
plt.pcolormesh(x, y, I, shading='auto', cmap='jet')
plt.colorbar(label='Intensity')
plt.xlabel('x (λ)')
plt.ylabel('y (λ)')
plt.title('2D Gaussian Beam Intensity Profile (focus at y=0)')
plt.show()