import batman
import numpy as np
import matplotlib.pyplot as plt
import os
print(os.getcwd())
np.__version__

limb_dark_data = np.genfromtxt("/ca25/ext_volume/ldc_result_TOI6158.csv", skip_header=17)
u1 = np.mean(limb_dark_data[:, 8])
u2 = np.mean(limb_dark_data[:, 10])
print(u1, u2)

#planet TOI 6158 b
params = batman.TransitParams()
params.t0   = 0.                #time of inferior conjunction
params.per  = 3.0446899        #orbital period in days
params.rp   = 0.1961553541771708     #planet radius (in units of stellar radii)
params.a    = 14.78115660954045      #semi-major axis (in units of stellar radii)
params.inc  = 86.51             #orbital inclination (in degrees)
params.ecc  = 0.059                #eccentricity
params.w    = 88.               #longitude of periastron (in degrees)
params.u    = [u1, u2]          #limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"  #limb darkening model

t = np.linspace(-0.06, 0.06, 1000)

m = batman.TransitModel(params, t)	        #initializes model
flux = m.light_curve(params)

plt.plot(t, flux)
plt.title('TOI-6158 b')
plt.xlabel("Time from central transit (days)")
plt.ylabel("Relative flux")
#plt.ylim((0.96, 1.01))
plt.savefig("TOI-6158b_assignment2_taskA.png") 
plt.show()
