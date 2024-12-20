import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import h5py
matplotlib.style.use('classic')
matplotlib.rcParams.update({
    'font.size': 13.2,        # Default font size
})
a = 0.0137225
diff = np.loadtxt("test2_diff.txt")
mc = np.loadtxt("test2_mc.txt")
sn = h5py.File('SN.h5', 'r+')
xs = sn['test2']['xs']
mat_T = sn['test2']['T4']
u = sn['test2']['u']

# analytical solution
R = 0.05
delta=0.5176512540260865
xsi_rt = np.vectorize(lambda r,t: r/(R/10.)/(t/(-1e-9))**delta)
Vxsi = np.vectorize(lambda x: 0.261997*x**-3.2404+0.255751*x**-1.87993 if x>=1. else 0.)
Wxsi = np.vectorize(lambda x: (x-1)**0.34005 * (1.02052-0.000712269*x+0.000172561*x*x) if x>=2. else (x-1)**0.397741 * (1.24353-0.175717*x+0.0318612*x*x) if x>1. else 0.)


Trt_fit = np.vectorize(lambda r,t: 0.8098923983397368*(t/-1e-9)**0.100238*Wxsi(xsi_rt(r,t))**0.5)

# ------- plot surface and bath temperatures
t_init = -(10**(1./delta)) * 1e-9
times = np.linspace(t_init, -1e-9, 1000)
xsiR = xsi_rt(R,times)
Lambda = xsiR**1.2*Vxsi(xsiR)*Wxsi(xsiR)**-1

Ts = Trt_fit(R,times)
Tbath = Ts * (1.+0.385372*(times/-1e-9)**(-0.579294)*Lambda)**0.25

plt.plot(times/1e-9, Ts, "r", label="surface")
plt.plot(times/1e-9, Tbath, "--b", label="bath")
plt.xlabel("$t$ [ns]")
plt.ylabel("$T(t)$ [HeV]")
plt.grid()
plt.legend(loc="best")
plt.show()

# ------- plot simulation profiles
r_anal = np.linspace(R*1e-10, R, 1000)
t1 = -58.251607e-9
t2 = -19.068532e-9
t3 = -1e-9

rho0 = 1.
omega = -0.5
beta, mu = 2., 0.6
f = 3e13
urt = np.vectorize(lambda r,t: 1e-13*f*Trt_fit(r,t)**beta*(rho0*r**-omega)**(1.-mu))


plt.figure("T")
plt.plot(mc[:,0], mc[:,1], color="k", lw=2.5,  label="Transport IMC")
plt.plot(mc[:,0], mc[:,3], color="k", lw=2.5, )
plt.plot(mc[:,0], mc[:,5], color="k", lw=2.5, )

plt.plot(diff[:,0], diff[:,1], c="lime", ls="-", lw=2, label="Diffusion Simulation")
plt.plot(diff[:,0], diff[:,3], c="lime", ls="-", lw=2, )
plt.plot(diff[:,0], diff[:,5], c="lime", ls="-", lw=2, )

plt.plot(r_anal, Trt_fit(r_anal, t3), c="r", ls="--", lw=2, label="Diffusion Analytic")
plt.plot(r_anal, Trt_fit(r_anal, t2), c="r", ls="--", lw=2)
plt.plot(r_anal, Trt_fit(r_anal, t1), c="r", ls="--", lw=2)

plt.plot(xs[0,:], mat_T[0,:], c="b", ls="--", lw=2, label=r"$S_8$ transport")
plt.plot(xs[1,:], mat_T[1,:], c="b", ls="--", lw=2)
plt.plot(xs[2,:], mat_T[2,:], c="b", ls="--", lw=2)

plt.ylabel("$T \\ [\\mathrm{{HeV}}]$", fontsize=24)
plt.xlabel("$r \\ [\\mathrm{{cm}}]$", fontsize=24)
plt.title("$\\mathrm{{Test \\ 2}}$", fontsize=22)
plt.legend(loc="upper left", fontsize=16).set_draggable(True)
plt.ylim(ymax=1.4)

ticks = np.linspace(0,R,11)
lticks = [f"{t:g}" for t in ticks]
plt.xticks(ticks, lticks)

plt.grid()
plt.tight_layout()
plt.savefig(f"Test2_T.pdf", bbox_inches='tight')


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

plt.plot(xs[0,:], u[0,:]*a, c="b", ls="--", lw=2, label=r"$S_8$ transport")
plt.plot(xs[1,:], u[1,:]*a, c="b", ls="--", lw=2)
plt.plot(xs[2,:], u[2,:]*a, c="b", ls="--", lw=2)


plt.ylabel("$u \\ [10^{{13}} \\ \\mathrm{{erg/cm^{{3}}}}]$", fontsize=24)
plt.xlabel("$r \\ [\\mathrm{{cm}}]$", fontsize=24)
plt.legend(loc="upper left", fontsize=16).set_draggable(True)
plt.ylim(ymax=2.5)
ticks = np.linspace(0,R,11)
lticks = [f"{t:g}" for t in ticks]
plt.xticks(ticks, lticks)
plt.grid()
plt.tight_layout()
plt.savefig(f"Test2_u.pdf", bbox_inches='tight')

plt.show()

quit()
