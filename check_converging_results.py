import h5py 
from converging_heat.converging_heat_wave import run_converging
import numpy as plt
import matplotlib.pyplot as plt
import numpy as np

def check(t):
    if t == 204.20877:
        iterator = 0
    elif t == 523.519754:

         iterator = 1
    elif t == 692.759852:

         iterator = 2
    elif t == 813.786114:

         iterator = 3
    tfinal = t
    a = 0.0137225 
    c = 29.98
    f = h5py.File('converging_heat/converging_heat_wave_results2.h5', 'r+')
    e = f['energy_density'][:]
    xs = f['xs'][:]
    phi = f['scalar_flux'][:]

    phi = phi[iterator]
    xs = xs[iterator] 
    e = e[iterator]
    print(xs, 'xs')
    print(phi, 'phi')


    phi_dim = phi * a * c

    f.close()


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
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+2] * 10, 'gx')
    # plt.plot(np.array([0.1]), boundary_temp[time_arg+1] * 10, 'yo')
#     print(boundary_temp)
    plt.legend()
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

check(204.20877)
check(523.519754)
check(692.759852)
check(813.786114)
