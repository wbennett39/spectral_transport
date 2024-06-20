"""This is me messing around trying to remember how to read data from a text file 
    into a Python array to plot it. - Stephen"""

import numpy as np
import matplotlib.pyplot as plt

#with open("Su-Olson_rad_en_dens.txt") as rad:
#    lines = rad.read().splitlines()  

infile = open("Su-Olson_rad_en_dens.txt", "r")

linetxt = infile.readlines()

infile.close()  

#print(linetxt[0])

y = np.array(linetxt[0])
print(type(y))

#print(y)

#print(type(linetxt))

#linetxt = np.array(linetxt)

#print(type(linetxt))
#print(linetxt)
#print(linetxt.shape)

x = np.array([0.10000, 0.31623, 1.00000, 3.16228, 10.0000, 31.6228, 100.000], dtype=np.float64)
#print(type(x))

# energy densities for t=0.01000

rad = np.array([0.09531,0.27526,0.64308,1.20052,2.23575,0.69020,0.35720])
mat = np.array([0.00468,0.04093,0.27126,0.94670,2.11186,0.70499,0.35914])

plt.plot(x, rad, label='Radiation Energy Density')
plt.plot(x, mat, label='Material Energy Density')

plt.xlabel("x")
plt.ylabel("Energy Density")
plt.legend()
plt.show()
