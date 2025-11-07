import batman
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc



def transit(args):

    limb_dark_data = np.genfromtxt("/ca25/ext_volume/ldc_result.csv", skip_header=17)
    u1 = np.mean(limb_dark_data[:, 8])
    u2 = np.mean(limb_dark_data[:, 10])

    #u1=0.1
    #u2=0.3
    params = batman.TransitParams()
    params.t0 = args['t0']
    params.per = args['period']
    params.rp = args['r_planet']
    params.a = args['a']
    params.inc = args['inc']
    params.ecc = args['ecc']
    params.w = args['w']
    params.u    = [u1, u2]
    params.limb_dark = args['limb_dark']

    t = np.linspace(-0.2, 0.2, 1000)

    m = batman.TransitModel(params, t)	        #initializes model
    flux = m.light_curve(params)

    plt.plot(t, flux)
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    # plt.ylim((0.989, 1.001))
    plt.savefig("TOI-2322b_assignment1_taskF.png") 
    plt.show()