import batman
import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())
np.__version__

limb_dark_data = np.genfromtxt("/ca25/ext_volume/ldc_result.csv", skip_header=17)
u1 = np.mean(limb_dark_data[:, 8])
u2 = np.mean(limb_dark_data[:, 10])
print(u1, u2)

#planet TOI-2322 b
params = batman.TransitParams()
params.t0   = 0.                #time of inferior conjunction
params.per  = 11.30717        #orbital period in days
params.rp   = 0.0134475255815366     #planet radius (in units of stellar radii)
params.a    = 29.98596630444186      #semi-major axis (in units of stellar radii)
params.inc  = 89.57             #orbital inclination (in degrees)
params.ecc  = 0.                #eccentricity
params.w    = 90.               #longitude of periastron (in degrees)
params.u    = [u1, u2]          #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"  #limb darkening model

t = np.linspace(-0.2, 0.2, 1000)

m = batman.TransitModel(params, t)	        #initializes model
flux = m.light_curve(params)

plt.plot(t, flux)
plt.xlabel("Time from central transit (days)")
plt.ylabel("Relative flux")
# plt.ylim((0.989, 1.001))
plt.savefig("TOI-2322b_assignment1_taskF.png") 
plt.show()
