import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import h5py
matplotlib.style.use('classic')
matplotlib.rcParams.update({
    'font.size': 16,        # Default font size
})


def plot_answers(spaces):
    # plt.ion()
    #####
    # spaces = 35
    M = 1
    #####
    #####
    a = 0.0137225
    c = 29.98
    #####

    diff = np.loadtxt("test1_diff.txt")
    mc = np.loadtxt("test1_mc.txt")
    sn_transport = h5py.File('converging_heat_wave_results_test1_1025.h5', 'r+')
    tr = sn_transport[f'M=[{M}]_[{spaces}]_cells']
    e = tr['energy_density'][:]
    xs = tr['xs'][:]
    phi = tr['scalar_flux'][:]
    edges = tr['edges'][:]
    print(edges, 'edges')
    phi_dim = phi * a * c
    mat_T = phi_dim * 0
    rho = 19.3

    ee1 = e[0,:] * a  * (0.1**1.6) / 10**-3  / 3.4 / (rho **.86)

    mat_T[0,:] = np.abs(ee1)**.625

    ee2 = e[1,:]* a  * (0.1**1.6) / 10**-3  / 3.4 / (rho **.86)
    mat_T[1,:] = np.abs(ee2)**.625

    ee3 = e[2,:] * a  * (0.1**1.6) / 10**-3  / 3.4 / (rho **.86)
    mat_T[2,:] = np.abs(ee3)**.625
    sn_transport.close()
    # analytical solution
    R = 0.001
    delta = 0.6795011543956738
    xsi_rt = np.vectorize(lambda r,t: r/(R/10.)/(t/(-1e-9))**delta)
    Vxsi = np.vectorize(lambda x: 0.434451*x**-2.75208+0.245073*x**-1.4541 if x>=1. else 0.)
    Wxsi = np.vectorize(lambda x: (x-1)**0.295477 * (1.08165-0.0271785*x+0.00105539*x*x) if x>=2. else (x-1)**0.40574 * (1.52093-0.376185*x+0.0655796*x*x) if x>1. else 0.)
    Trt_fit = np.vectorize(lambda r,t: 1.3450346510206614*(t/-1e-9)**0.0920519*Wxsi(xsi_rt(r,t))**0.625)

    # ------- plot surface and bath temperatures
    t_init = -(10**(1./delta)) * 1e-9
    times = np.linspace(t_init, -1e-9, 1000)
    xsiR = xsi_rt(R,times)
    Lambda = xsiR**1*Vxsi(xsiR)*Wxsi(xsiR)**-1.5

    Ts = Trt_fit(R,times)
    Tbath = Ts * (1.+0.103502*(times/-1e-9)**(-0.541423)*Lambda)**0.25

    # plt.plot(times/1e-9, Ts, "r", label="surface")
    # plt.plot(times/1e-9, Tbath, "--b", label="bath")
    # plt.xlabel("$t$ [ns]")
    # plt.ylabel("$T(t)$ [HeV]")
    # plt.grid()
    # plt.legend(loc="best")
    # plt.show()

    # ------- plot simulation profiles
    r_anal = np.linspace(0., R, 1000)
    t1 = -2.2122309472889687e-08
    t2 = -9.448424353186857e-09
    t3 = -1e-9

    #t1 = -2.4e-8
    #t2 =-2.2122309472889687e-08
    #t3 = -9.448424353186857e-09

    #t1 = -2.4e-8
    #t2 =-2.2122309472889687e-08
    #t3 =-9.448424353186857e-09

    #t3 = -2.2122309472889687e-08
    #t2 =-2.3e-08
    #t1 = -2.5e-8

    #t1 = -2.9e-8
    #t2 = - 2.85e-8
    #t3 = -2.8e-8
    #menis_times = np.array([-29.625, -29.6, -29.5])
    #t1 =-2.9625e-08
    #t2 =-2.96e-08
    #t3 =-2.5e-08

    #t1 = -2.5e-08
    #t2 = -2.3e-08
    #t3 = -2.2122309472889687e-08


    rho0 = 19.3
    omega = 0.
    beta, mu = 1.6, 0.14
    f = 3.4e13
    urt = np.vectorize(lambda r,t: 1e-13*f*Trt_fit(r,t)**beta*(rho0*r**-omega)**(1.-mu))
    #phi = phi[iterator]
    #xs = xs[iterator]
    #e = e[iterator]


    plt.figure("T")
    plt.plot(mc[:,0]/1e-4, mc[:,1], color="k", lw=2.5,  label="Transport IMC")
    plt.plot(mc[:,0]/1e-4, mc[:,3], color="k", lw=2.5, )
    plt.plot(mc[:,0]/1e-4, mc[:,5], color="k", lw=2.5, )

    plt.plot(diff[:,0]/1e-4, diff[:,1], c="lime", ls="-", lw=2, label="Diffusion Simulation")
    plt.plot(diff[:,0]/1e-4, diff[:,3], c="lime", ls="-", lw=2, )
    plt.plot(diff[:,0]/1e-4, diff[:,5], c="lime", ls="-", lw=2, )

    plt.plot(r_anal/1e-4, Trt_fit(r_anal, t3), c="r", ls="--", lw=2, label="Diffusion Analytic")
    plt.plot(r_anal/1e-4, Trt_fit(r_anal, t2), c="r", ls="--", lw=2)
    plt.plot(r_anal/1e-4, Trt_fit(r_anal, t1), c="r", ls="--", lw=2)

    plt.plot(xs[0,:]/1e-4, 10*(np.abs(phi_dim[0,:])/a/c)**.25, 'b-x', label = 'radiation temp')
    plt.plot(xs[1,:]/1e-4, 10*(np.abs(phi_dim[1,:])/a/c)**.25, 'b-x')
    plt.plot(xs[2,:]/1e-4, 10*(np.abs(phi_dim[2,:])/a/c)**.25, 'b-x')
    plt.plot(edges/1e-4, edges*0, 'k|', markersize = 40)


    plt.plot(xs[0,:]/1e-4, 10*mat_T[0,:], 'k--', label = 'radiation temp')
    plt.plot(xs[1,:]/1e-4, 10*mat_T[1,:], 'k--')
    plt.plot(xs[2,:]/1e-4, 10*mat_T[2,:], 'k--')





    plt.ylabel("$T \\ [\\mathrm{{HeV}}]$", fontsize=24)
    plt.xlabel("$r \\ [\\mathrm{{\\mu m}}]$", fontsize=24)
    plt.title("$\\mathrm{{Test \\ 1}}$", fontsize=22)
    #plt.legend(loc="upper left", fontsize=16).set_draggable(True)
    plt.ylim(ymax=2.5)
    ticks = np.linspace(0,R/1e-4,11)
    lticks = [f"{t:g}" for t in ticks]
    plt.xticks(ticks, lticks)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"Test1_T.pdf", bbox_inches='tight')

    ff = h5py.File('SN.h5', 'r+')
    ff['test1']['xs'] = xs
    ff['test1']['T4'] = 10* mat_T
    ff['test1']['u'] = e * 10**16 #convert GJ to kelvin
    ff.close()



    # plt.figure("u")
    # plt.plot(mc[:,0]/1e-4, mc[:,2], color="k", lw=2.5,  label="Transport IMC")
    # plt.plot(mc[:,0]/1e-4, mc[:,4], color="k", lw=2.5, )
    # plt.plot(mc[:,0]/1e-4, mc[:,6], color="k", lw=2.5, )

    # plt.plot(diff[:,0]/1e-4, diff[:,2], c="lime", ls="-", lw=2, label="Diffusion Simulation")
    # plt.plot(diff[:,0]/1e-4, diff[:,4], c="lime", ls="-", lw=2, )
    # plt.plot(diff[:,0]/1e-4, diff[:,6], c="lime", ls="-", lw=2, )

    # plt.plot(r_anal/1e-4, urt(r_anal, t3), c="r", ls="--", lw=2, label="Diffusion Analytic")
    # plt.plot(r_anal/1e-4, urt(r_anal, t2), c="r", ls="--", lw=2)
    # plt.plot(r_anal/1e-4, urt(r_anal, t1), c="r", ls="--", lw=2)

    # plt.ylabel("$u \\ [10^{{13}} \\ \\mathrm{{erg/cm^{{3}}}}]$", fontsize=24)
    # plt.xlabel("$r \\ [\\mathrm{{\\mu m}}]$", fontsize=24)
    # plt.legend(loc="upper left", fontsize=16).set_draggable(True)
    # plt.ylim(ymax=140)
    # ticks = np.linspace(0,R/1e-4,11)
    # lticks = [f"{t:g}" for t in ticks]
    # plt.xticks(ticks, lticks)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(f"Test1_u.pdf", bbox_inches='tight')

    plt.show()



# plot_answers(20)
# plot_answers(25)

# plot_answers(30)

# plot_answers(35)

# plot_answers(40)
# plot_answers(45)
plot_answers(100)