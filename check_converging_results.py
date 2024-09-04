import h5py 
from converging_heat.converging_heat_wave import run_converging
import numpy as plt
import matplotlib.pyplot as plt
import numpy as np

def check(t, iterator, spaces = 22, M = 1):
#     if t == 20.420877:
#         iterator = 0
#     elif t == 52.3519754:

#          iterator = 1
#     elif t == 69.2759852:

#          iterator = 2
#     elif t == 81.3786114:

#          iterator = 3
    tfinal = t
    a = 0.0137225 
    c = 29.98
    ff = h5py.File('converging_heat/converging_heat_wave_results2.h5', 'r+')
    f = ff[f'M=[{M}]_[{spaces}]_cells']
    print(f.keys()) 
    e = f['energy_density'][:]
    xs = f['xs'][:]
    phi = f['scalar_flux'][:]

    phi = phi[iterator]
    xs = xs[iterator] 
    e = e[iterator]
    print(xs, 'xs')
    print(phi, 'phi')


    phi_dim = phi * a * c

    ff.close()


    f = h5py.File('heat_wavepos.h5', 'r+')
    boundary_temp = f['temperature'][:] / 10 # convert from HeV to keV
    boundary_temp[0] = boundary_temp[1]

    boundary_time = (f['times'][:] - f['times'][0]) 
    # print(f['times'][:], 'times')
    f.close()

    time_arg = np.argmin(np.abs(boundary_time  - tfinal/c))


    dimensional_t = tfinal/c 
    menis_t = -29.6255 + dimensional_t
    rf= 0.01 * (-menis_t) ** 0.679502 

    ee = e * a  / 10**-3 * (0.1)**1.6
    T1 = (np.abs(ee))
    T = np.power(T1, 0.625)
    r_meni, T_meni = run_converging(tfinal/ c)
    plt.ion()
    plt.figure(2)
    plt.plot(r_meni, T_meni, 'bx', label = 'T Meni')
    plt.plot(xs, T*10, 'k-', label = 'T')
    plt.plot(xs, 10*(np.abs(phi_dim)/a/c)**.25, 'b-', label = 'radiation temp')
    # plt.xlim(rf-5e-4, 0.1)
    plt.plot(np.array([0.1]), boundary_temp[time_arg] * 10, 'rx')
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+2] * 10, 'gx')
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+1] * 10, 'yo')
#     print(boundary_temp)
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(r_meni, T_meni, 'bx', label = 'T Meni')
    plt.plot(xs, T*10, 'k-', label = 'T')
    plt.plot(xs, 10*(np.abs(phi_dim)/a/c)**.25, 'b-', label = 'radiation temp')
    plt.xlim(rf, 0.1)
    plt.plot(np.array([0.1]), boundary_temp[time_arg] * 10, 'rx')
    plt.xlabel('x [cm]')

    plt.ylabel('T [HeV]')
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+2] * 10, 'gx')
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+1] * 10, 'yo')
#     print(boundary_temp)
#     if iterator == 3: 
#           plt.legend()
    plt.show()

    plt.figure(4)
    # plt.plot(r_meni, T_meni, 'bx', label = 'T Meni')
    plt.plot(xs, ee, 'k-', label = 'energy density')
    plt.plot(xs, (np.abs(phi_dim)/a/c), 'b-', label = 'radiation energy density')
    plt.xlim(rf, 0.1)
    # plt.plot(np.array([0.1]), boundary_temp[time_arg] * 10, 'rx')
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+2] * 10, 'gx')
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+1] * 10, 'yo')
#     print(boundary_temp)
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.plot(xs, phi_dim, label = 'scalar flux')
    plt.plot(xs, a*c*(T)**4, label = r'$T^4$' )
    plt.legend()
    plt.show()

# check(204.20877,0)
# check(523.519754,1)
# check(692.759852,2)
# check(813.786114,3)


check(2.0420877,0)
check(5.23519754,1)
check(6.92759852,2)
check(8.13786114,3)
# check(.10,0)
# check(1.00,1)
# check(5.0,2)
# check(15.0,3)

# check(20.420877,0)
# check(52.3519754,1)
# check(69.2759852,2)
# check(81.3786114,3)