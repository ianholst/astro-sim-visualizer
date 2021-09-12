import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import camb

npix = 200
size = 2 # Mpc
kmin = 2 * np.pi / size # 1/Mpc
kmax = 2 * np.pi / (size/npix) # 1/Mpc

kx = np.fft.rfftfreq(npix) * npix * kmin # 1/Mpc
ky = np.fft.fftfreq(npix) * npix * kmin # 1/Mpc
kz = np.fft.fftfreq(npix) * npix * kmin # 1/Mpc
k_grid = np.sqrt(kx[None,None,:]**2 + ky[None,:,None]**2 + kz[:,None,None]**2)

H0 = 67.5 # [km/s/Mpc]
As = 2.198e-9 # Scalar amplitude
ns = 0.9655 # Scalar spectral index
params = camb.CAMBparams()
params.set_cosmology(H0=H0)
params.InitPower.set_params(As=As, ns=ns)
matter_power_interpolator = camb.get_matter_power_interpolator(params, nonlinear=True, hubble_units=False, k_hunit=False, kmax=1.1*kmax, zmax=1, log_interp=True)

ks_interp = np.geomspace(1e-5, kmax, 1000) # 1/Mpc
P_interp = np.exp(matter_power_interpolator(0, np.log(ks_interp))[0]) # Mpc^3
Pmm = interpolate.CubicSpline(ks_interp, P_interp) # Mpc^3

plt.figure()
plt.plot(ks_interp, P_interp)
plt.xscale("log")
plt.yscale("log")
plt.show()


P_evaluated_on_grid = Pmm(k_grid)

# Generate real and imaginary parts
np.random.seed(0)
real_part = np.sqrt(0.5*P_evaluated_on_grid) * np.random.normal(loc=0.0,scale=1.0,size=k_grid.shape) * npix/(2*np.pi)
imaginary_part = np.sqrt(0.5*P_evaluated_on_grid) * np.random.normal(loc=0.0,scale=1.0,size=k_grid.shape) * npix/(2*np.pi)

# Get map in real space and return
ft_map = (real_part + imaginary_part*1.0j) * npix**2
ft_map[0,0] = 0.0
density_map = np.fft.irfftn(ft_map)

plt.figure(figsize=[10,10])
plt.imshow(density_map[-1], interpolation="none")
plt.show()

normalized_density_map = ((density_map - density_map.min()) / (density_map.max() - density_map.min()) * 255).astype(np.uint8)
normalized_density_map.tofile("test_uint8.dat")

