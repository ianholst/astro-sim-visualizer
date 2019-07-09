import camb
import numpy as np
import matplotlib.pyplot as plt
import yt
import scipy.interpolate

H0 = 67.5 # [km/s/Mpc]
h = 67.5 / 100 # [dimensionless]
c = 299792.458 # [km/s]

Omega_m = 0.3175 # Matter density parameter
Omega_K = 0.0 # Curvature density parameter
Omega_r = 4.15e-5 # Radiation density parameter
omch2 = 0.12470
ombh2 = 0.02222
Omega_c = omch2/h**2 # Cold dark matter density parameter
Omega_b = ombh2/h**2 # Baryon density parameter
Omega_m = Omega_b + Omega_c
Omega_L = 1 - Omega_K - Omega_m - Omega_r # Lambda density parameter

As = 2.198e-9 # Scalar amplitude
ns = 0.9655 # Scalar spectral index


params = camb.CAMBparams()
params.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
params.InitPower.set_params(As=As, ns=ns)

matter_power_interpolator = camb.get_matter_power_interpolator(params,
    nonlinear=True, hubble_units=False, k_hunit=False, log_interp=True, kmax=100, zmax=1)#, extrap_kmax=15000)


ks = np.logspace(-5,2, 1000)
z = 1

P_interp = lambda z, k: np.exp(matter_power_interpolator(z, np.log(k))[0])

plt.plot(ks, P_interp(z,ks))
plt.xscale("log")
plt.yscale("log")
2*np.pi / 1e-1 # Mpc

P = P_interp(z,ks)

Pspline = scipy.interpolate.CubicSpline(ks, P)

if seed is not None:
	np.random.seed(seed)

npix = 50
size = 100 # Mpc
kmin = 2 * np.pi / size # Mpc^-1
kmax = 2 * np.pi / (size/npix) # Mpc^-1

kx = np.fft.rfftfreq(npix) * npix * kmin
ky = np.fft.fftfreq(npix) * npix * kmin
kz = np.fft.fftfreq(npix) * npix * kmin
k_grid = np.sqrt(kx[None,None,:]**2 + ky[None,:,None]**2 + kz[:,None,None]**2)

#Compute the multipole moment of each FFT pixel
P_evaluated_on_grid = Pspline(k_grid)


#Generate real and imaginary parts
real_part = np.sqrt(0.5*P_evaluated_on_grid) * np.random.normal(loc=0.0,scale=1.0,size=k_grid.shape) * npix/(2.0*np.pi)
imaginary_part = np.sqrt(0.5*P_evaluated_on_grid) * np.random.normal(loc=0.0,scale=1.0,size=k_grid.shape) * npix/(2.0*np.pi)

#Get map in real space and return
ft_map = (real_part + imaginary_part*1.0j) * k_grid.shape[0]**2
ft_map[0,0] = 0.0

noise_map = np.fft.irfftn(ft_map)

plt.imshow(noise_map[0])


a = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
a.tofile("generated50.dat")


def calculate_bias_grid(bias, kbins, kgrid):
    # Create 3D field in k space with radially symmetric bias
    k_bin_centers = kbins[:-1] + 1/2
    bias_interpolator = interpolate.interp1d(k_bin_centers, bias, kind="linear", bounds_error=False, fill_value=(0.0,0.0))
    bias_grid = bias_interpolator(kgrid)
    return bias_grid


a=np.fromfile("d512.dat", dtype=np.float32)
a=a.reshape((512,512,512))

b = a[0:256,0:256,0:256]
b.shape
b.tofile("a.dat")


a=np.fromfile("zreion_0.dat", dtype=np.float32)
a=a.reshape((512,512,512))

b = a[0:256,0:256,0:256]
b.shape
b.tofile("z256.dat")




a=yt.load("data/ArepoBullet/snapshot_150.hdf5")
a.all_data()
dir(a)
dir(a.field_list)
a.field_list


ad = a.all_data()
ad["Density"]
