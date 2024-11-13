import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import h5py
matplotlib.style.use('classic')
matplotlib.rcParams.update({
    'font.size': 16,        # Default font size
})

sigma_sb = 5.670374419e-5
clight = 2.99792458e10
arad = 4. * sigma_sb / clight 
ev_kelvin = 1.160451812e4
hev_kelvin = 100. * ev_kelvin
kev_kelvin = 1000. * ev_kelvin
sn = h5py.File('SN_test4.h5', 'r+')
xs = sn['test4']['xs']
mat_T = sn['test4']['T4']
u = sn['test4']['u']
a = 0.0137225

diff = np.loadtxt("test4_diff.txt")
mc = np.loadtxt("test4_mc.txt")

# analytical solution
R = 10.
delta=0.462367118894146
xsi_rt = np.vectorize(lambda r,t: r/(R/10.)/(t/(-1e-9))**delta)
Vxsi = np.vectorize(lambda x: 0.0624652*x**-3.83607+0.399907*x**-2.1566 if x>=1. else 0.)
Wxsi = np.vectorize(lambda x: (x-1)**1.102 * (0.184553+0.150513*x+4.39412e-05*x*x)  if x>=2. else (x-1)**1.14124 * (0.225081+0.127014*x+0.00162581*x*x) if x>1. else 0.)
Trt_fit = np.vectorize(lambda r,t: 5.52153922031134*(t/-1e-9)**0.242705*Wxsi(xsi_rt(r,t))**0.25)

# ------- plot surface and bath temperatures
t_init = -(10**(1./delta)) * 1e-9
times = np.linspace(t_init, -1e-9, 1000)
xsiR = xsi_rt(R,times)
Lambda = xsiR*Vxsi(xsiR)

Ts = Trt_fit(R,times)
Tbath = Ts * (1.+0.083391*(times/-1e-9)**(-0.537633)*Lambda)**0.25

plt.plot(times/1e-9, Ts, "r", label="surface")
plt.plot(times/1e-9, Tbath, "--b", label="bath")
plt.xlabel("$t$ [ns]")
plt.ylabel("$T(t)$ [HeV]")
plt.grid()
plt.legend(loc="best")
plt.show()

# ------- plot simulation profiles
r_anal = np.linspace(R*1e-10, R, 1000)
t1 = -9.470688883217099e-08
t2 = -2.7126998146008884e-08
t3 = -1e-9

rho0 = 1.
omega = -1.
beta, mu = 4., 1.
f = 1.25*arad
urt = np.vectorize(lambda r,t: 1e-13*f*(Trt_fit(r,t)*hev_kelvin)**beta*(rho0*r**-omega)**(1.-mu))


plt.figure("T")
plt.plot(mc[:,0], mc[:,1]*0.1, color="k", lw=2.5,  label="Transport IMC")
plt.plot(mc[:,0], mc[:,3]*0.1, color="k", lw=2.5, )
plt.plot(mc[:,0], mc[:,5]*0.1, color="k", lw=2.5, )

plt.plot(diff[:,0], diff[:,1]*0.1, c="lime", ls="-", lw=2, label="Diffusion Simulation")
plt.plot(diff[:,0], diff[:,3]*0.1, c="lime", ls="-", lw=2, )
plt.plot(diff[:,0], diff[:,5]*0.1, c="lime", ls="-", lw=2, )

plt.plot(r_anal, Trt_fit(r_anal, t3)*0.1, c="r", ls="--", lw=2, label="Diffusion Analytic")
plt.plot(r_anal, Trt_fit(r_anal, t2)*0.1, c="r", ls="--", lw=2)
plt.plot(r_anal, Trt_fit(r_anal, t1)*0.1, c="r", ls="--", lw=2)

plt.plot(xs[0,:], mat_T[0,:], c="b", ls="--", lw=2, label=r"$S_8$ transport")
plt.plot(xs[1,:], mat_T[1,:], c="b", ls="--", lw=2)
plt.plot(xs[2,:], mat_T[2,:], c="b", ls="--", lw=2)

plt.ylabel("$T \\ [\\mathrm{{KeV}}]$", fontsize=24)
plt.xlabel("$r \\ [\\mathrm{{cm}}]$", fontsize=24)
plt.title("$\\mathrm{{Test \\ 4}}$", fontsize=22)
plt.legend(loc="upper left", fontsize=16).set_draggable(True)
plt.ylim(ymax=1.2)
ticks = np.linspace(0,R,11)
lticks = [f"{t:g}" for t in ticks]
plt.xticks(ticks, lticks)
plt.grid()
plt.tight_layout()
plt.savefig(f"Test4_T.pdf", bbox_inches='tight')


plt.figure("u")
plt.plot(mc[:,0], mc[:,2], color="k", lw=2.5,  label="Transport IMC")
plt.plot(mc[:,0], mc[:,4], color="k", lw=2.5, )
plt.plot(mc[:,0], mc[:,6], color="k", lw=2.5, )

plt.plot(diff[:,0], diff[:,2], c="lime", ls="-", lw=2, label="Diffusion Simulation")
plt.plot(diff[:,0], diff[:,4], c="lime", ls="-", lw=2, )
plt.plot(diff[:,0], diff[:,6], c="lime", ls="-", lw=2, )

plt.plot(r_anal, urt(r_anal, t3), c="r", ls="--", lw=2, label="Diffusion Analytic")
plt.plot(r_anal, urt(r_anal, t2), c="r", ls="--", lw=2)
plt.plot(r_anal, urt(r_anal, t1), c="r", ls="--", lw=2)

plt.plot(xs[0,:], 5*u[0,:]/1e13 * a, c="b", ls="--", lw=2, label=r"$S_8$ transport")
plt.plot(xs[1,:], 5*u[1,:]/1e13 * a, c="b", ls="--", lw=2)
plt.plot(xs[2,:], 5*u[2,:]/1e13 * a, c="b", ls="--", lw=2)

plt.ylabel("$u \\ [10^{{13}} \\ \\mathrm{{erg/cm^{{3}}}}]$", fontsize=24)
plt.xlabel("$r \\ [\\mathrm{{cm}}]$", fontsize=24)
plt.legend(loc="upper left", fontsize=16).set_draggable(True)
plt.ylim(ymax=35)
ticks = np.linspace(0,R,11)
lticks = [f"{t:g}" for t in ticks]
plt.xticks(ticks, lticks)
plt.grid()
plt.tight_layout()
plt.savefig(f"Test4_u.pdf", bbox_inches='tight')

plt.show()

quit()