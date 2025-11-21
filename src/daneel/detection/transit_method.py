import batman
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
import yaml




def transit(args, planet_name):

    star_name = planet_name[:-2]
   # comp_astro_25/src/daneel/parameters/limb_dark_data/ldc_result_TOI-2322.csv
    limb_dark_data = np.genfromtxt(f"/ca25/comp_astro_25/src/daneel/parameters/limb_dark_data/ldc_result_{star_name}.csv")
    
    u1 = np.mean(limb_dark_data[1:, 8])
    u2 = np.mean(limb_dark_data[1:, 10])

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
    params.planet_name= args['planet_name']

    
    t = np.linspace(-0.2, 0.2, 1000)
    
    m = batman.TransitModel(params, t)	        #initializes model
    flux = m.light_curve(params)


    #### ######  ###### PLOT ###### ###### ######
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, flux)
    plt.title(f"{planet_name}_light_curve.png")
    
    plt.grid()
    plt.xlabel("Time [days]")
    plt.ylabel("Relative flux")
   
    plt.plot(t, flux, label="Transit model")
    plt.legend()
    
    
    plt.savefig(f"light_curves/{planet_name}_light_curve.png",dpi=300)
    plt.show()


planets_target = ["TOI-2322_b" ]


for n, p in enumerate(planets_target):
    print("Analysing planet", n+1, ": ",p)

    yaml_path = "/ca25/comp_astro_25/src/daneel/parameters/parameter_" + p + ".yaml"

    with open(yaml_path, "r") as f:
        args = yaml.safe_load(f)

    transit(args,p)