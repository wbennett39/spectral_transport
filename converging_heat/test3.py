import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import h5py
matplotlib.style.use('classic')
matplotlib.rcParams.update({
    'font.size': 16,        # Default font size
})


#####
spaces =150
M = 1
#####
#####
a = 0.0137225
c = 29.98
#####


sn_transport = h5py.File('results_test3_1030.h5', 'r+')
tr = sn_transport[f'M=[{M}]_[{spaces}]_cells']
e = tr['energy_density'][:]
xs = tr['xs'][:]
phi = tr['scalar_flux'][:]
edges = tr['edges'][:]
phi_dim = phi * a * c
sn_transport.close()
diff = np.loadtxt("test3_diff.txt")
mc = np.loadtxt("test3_mc.txt")

# analytical solution
R = 0.001
delta=1.1157535873060416
xsi_rt = np.vectorize(lambda r,t: r/(R/10.)/(t/(-1e-9))**delta)
Vxsi = np.vectorize(lambda x: 0.887854*x**-2.23268+0.227769*x**-1.03679 if x>=1. else 0.)
Wxsi = np.vectorize(lambda x: (x-1)**0.210071 * (1.27048-0.0470724*x+0.00179721*x*x) if x>=2. else (x-1)**0.357506 * (1.9792-0.619497*x+0.110644*x*x) if x>1. else 0.)
Trt_fit = np.vectorize(lambda r,t: 1.198199926365715*(t/-1e-9)**0.0276392*Wxsi(xsi_rt(r,t))**0.5)

# ------- plot surface and bath temperatures
t_init = -(10**(1./delta)) * 1e-9
times = np.linspace(t_init, -1e-9, 1000)
xsiR = xsi_rt(R,times)
Lambda = xsiR**0.6625*Vxsi(xsiR)*Wxsi(xsiR)**-1

Ts = Trt_fit(R,times)
Tbath = Ts * (1.+0.075821*(times/-1e-9)**(-0.316092)*Lambda)**0.25

# plt.plot(times/1e-9, Ts, "r", label="surface")
# plt.plot(times/1e-9, Tbath, "--b", label="bath")
# plt.xlabel("$t$ [ns]")
# plt.ylabel("$T(t)$ [HeV]")
# plt.grid()
# plt.legend(loc="best")
# plt.show()

# ------- plot simulation profiles
r_anal = np.linspace(R*1e-10, R, 1000)
# t1 = -7.5e-09
# t2 = -7.4e-09
# t3 = -7e-09
t1 = -6.591897629554719e-09
t2 = -3.926450981261105e-09
t3 = -1e-9

rho0 = 1.
omega = 0.45
beta, mu = 2., 0.25
f = 1e13
urt = np.vectorize(lambda r,t: 1e-13*f*Trt_fit(r,t)**beta*(rho0*r**-omega)**(1.-mu))


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
rad_T = phi*0
rad_T[0, :] = 10*(np.abs(phi_dim[0,:])/a/c)**.25
rad_T[1, :] = 10*(np.abs(phi_dim[1,:])/a/c)**.25
rad_T[2, :] = 10*(np.abs(phi_dim[2,:])/a/c)**.25

plt.plot(edges/1e-4, edges*0, 'k|', markersize = 40)
plt.ylabel("$T \\ [\\mathrm{{HeV}}]$", fontsize=24)
plt.xlabel("$r \\ [\\mathrm{{\\mu m}}]$", fontsize=24)
plt.title("$\\mathrm{{Test \\ 3}}$", fontsize=22)
plt.legend(loc="upper left", fontsize=16).set_draggable(True)
plt.ylim(ymax=2.)
ticks = np.linspace(0,R/1e-4,11)
lticks = [f"{t:g}" for t in ticks]
plt.xticks(ticks, lticks)
plt.grid()
plt.tight_layout()
plt.savefig(f"Test3_T.pdf", bbox_inches='tight')


plt.figure("u")
plt.plot(mc[:,0]/1e-4, mc[:,2], color="k", lw=2.5,  label="Transport IMC")
plt.plot(mc[:,0]/1e-4, mc[:,4], color="k", lw=2.5, )
plt.plot(mc[:,0]/1e-4, mc[:,6], color="k", lw=2.5, )

plt.plot(diff[:,0]/1e-4, diff[:,2], c="lime", ls="-", lw=2, label="Diffusion Simulation")
plt.plot(diff[:,0]/1e-4, diff[:,4], c="lime", ls="-", lw=2, )
plt.plot(diff[:,0]/1e-4, diff[:,6], c="lime", ls="-", lw=2, )

plt.plot(r_anal/1e-4, urt(r_anal, t3), c="r", ls="--", lw=2, label="Diffusion Analytic")
plt.plot(r_anal/1e-4, urt(r_anal, t2), c="r", ls="--", lw=2)
plt.plot(r_anal/1e-4, urt(r_anal, t1), c="r", ls="--", lw=2)

plt.ylabel("$u \\ [10^{{13}} \\ \\mathrm{{erg/cm^{{3}}}}]$", fontsize=24)
plt.xlabel("$r \\ [\\mathrm{{\\mu m}}]$", fontsize=24)
plt.legend(loc="upper left", fontsize=16).set_draggable(True)
plt.ylim(ymax=45)
ticks = np.linspace(0,R/1e-4,11)
lticks = [f"{t:g}" for t in ticks]
plt.xticks(ticks, lticks)
plt.grid()
plt.tight_layout()
plt.savefig(f"Test3_u.pdf", bbox_inches='tight')

plt.show()
ff = h5py.File('SN.h5', 'r+')
del ff['test3']['xs']
del ff['test3']['T4']
del ff['test3']['u']
ff['test3']['xs'] = xs
ff['test3']['T4'] = 10* rad_T
ff['test3']['u'] = e * 10**16 #convert GJ to kelvin
ff.close()

quit()
