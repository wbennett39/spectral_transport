"""Inverse bow shock with the correct downstream entropy/shear-layer asymptote.

The curved shock deposits K=K(psi). Far downstream the flow approaches a
parallel inviscid shear/entropy layer: p=p_far and direction u/v=m_far are
common to all streamlines, while rho, q, and Mach vary with psi.

This script uses the branch-safe M^2 thermodynamic variable and imposes that
asymptotic layer at finite YMAX.  It is intended to test convergence before a
semi-infinite spectral formulation.
"""
import time
import numpy as np
import torch
from scipy.optimize import root, least_squares
from scipy.integrate import quad
from scipy.interpolate import interp1d, RegularGridInterpolator

torch.set_default_dtype(torch.float64)
G=1.4; MI=2.0; BETA=np.deg2rad(70.0); A=1/np.tan(BETA)
H0=1/(G-1)+0.5*MI*MI


def shock_np_from_sp(ss):
    Z=MI*MI/(1+ss*ss); R=(G+1)*Z/((G-1)*Z+2)
    P=1+2*G/(G+1)*(Z-1); p=P/G
    u=MI*(ss*ss+1/R)/(1+ss*ss)
    v=MI*ss*(1-1/R)/(1+ss*ss)
    K=p/R**G; a2=G*p/R; mu=(u*u+v*v)/a2
    return R,p,u,v,K,mu

RF,PF,UF,VF,KF,MUF=shock_np_from_sp(A)
MF=UF/VF
CPINF=(A-MF)/MI

def K_np(psi):
    t=np.tanh(psi/MI); ss=A*t
    return shock_np_from_sp(ss)[4]

def asym_state_np(psi):
    K=K_np(np.asarray(psi))
    rho=(PF/K)**(1/G)
    a2=G*PF/rho
    q2=2*(H0-G/(G-1)*PF/rho)
    mu=q2/a2
    return rho,a2,q2,mu

def cprime_np(psi):
    rho,a2,q2,mu=asym_state_np(psi)
    v=np.sqrt(q2)/np.sqrt(1+MF*MF)
    return -1/(rho*v)

# Match C(psi)=x-m_f y to the far straight-shock asymptote.
_INT,_=quad(lambda s: float(cprime_np(s)-CPINF),0,np.inf,epsabs=2e-12,epsrel=2e-12,limit=300)
C0=-A*np.log(2)-_INT
WOFF=C0+A*np.log(2)
WFAR_SLOPE=MF-A


def far_profile(xi,YMAX,nfine=12000):
    """Return psi(xi,YMAX), mu(xi,YMAX), and W asymptote.

    C(psi) is tabulated monotonically from psi=0 to MI*YMAX.  Linear x mapping
    at the top means C_target=C_shock+xi(C_body-C_shock).
    """
    pmax=MI*YMAX
    ps=np.linspace(0,pmax,nfine)
    cp=cprime_np(ps)
    # cumulative trapezoid without scipy version dependence
    C=np.empty_like(ps); C[0]=C0
    C[1:]=C0+np.cumsum(0.5*(cp[1:]+cp[:-1])*np.diff(ps))
    Cshock=A*np.log(np.cosh(YMAX))-MF*YMAX
    # C(pmax) and exact shock C differ exponentially; pin endpoint to exact shock
    correction=Cshock-C[-1]
    # distribute only in the exponentially-small high-psi tail to preserve C0
    ramp=(ps/pmax)**8 if pmax>0 else ps
    C=C+correction*ramp
    target=Cshock+xi*(C0-Cshock)
    # C decreases with psi; invert on reversed arrays
    psi_top=np.interp(target,C[::-1],ps[::-1])
    mu_top=asym_state_np(psi_top)[3]
    Wtop=(MF*YMAX+C0)-A*np.log(np.cosh(YMAX))
    return psi_top,mu_top,Wtop,Cshock,C[-1]


def run(NX=9,NY=17,YMAX=4.0,previous=None,maxfev=500,save=True):
    xi_np=np.linspace(0,1,NX); y_np=np.linspace(0,YMAX,NY)
    dxi=xi_np[1]-xi_np[0]; dy=y_np[1]-y_np[0]
    xi=torch.tensor(xi_np); y=torch.tensor(y_np); XI=xi[:,None]

    def sp(y): return A*torch.tanh(y)
    def shock(y):
        ss=sp(y); Z=MI*MI/(1+ss*ss); R=(G+1)*Z/((G-1)*Z+2)
        P=1+2*G/(G+1)*(Z-1); rho=R; pres=P/G
        u=MI*(ss*ss+1/R)/(1+ss*ss); v=MI*ss*(1-1/R)/(1+ss*ss)
        a2=G*pres/rho; mu=(u*u+v*v)/a2
        return rho,pres,u,v,mu
    rho_s,p_s,u_s,v_s,mu_s=shock(y)

    def KKP(psi):
        zz=psi/MI; t=torch.tanh(zz); den=1+A*A*t*t; Z=MI*MI/den
        R=(G+1)*Z/((G-1)*Z+2); P=1+2*G/(G+1)*(Z-1)
        K=(P/G)/R**G
        dZ=-2*MI*A*A*t*(1-t*t)/(den*den)
        dln=2*G/(2*G*Z-(G-1))-2*G/(Z*((G-1)*Z+2))
        return K,K*dln*dZ
    def rho_from_mu(psi,mu):
        K,_=KKP(psi); a2=2*(G-1)*H0/(2+(G-1)*mu)
        rho=(a2/(G*K))**(1/(G-1)); return rho,a2
    def dx(q):
        o=torch.empty_like(q); o[1:-1]=(q[2:]-q[:-2])/(2*dxi); o[0]=(-3*q[0]+4*q[1]-q[2])/(2*dxi); o[-1]=(3*q[-1]-4*q[-2]+q[-3])/(2*dxi); return o
    def dyop(q):
        o=torch.empty_like(q); o[:,1:-1]=(q[:,2:]-q[:,:-2])/(2*dy); o[:,0]=(-3*q[:,0]+4*q[:,1]-q[:,2])/(2*dy); o[:,-1]=(3*q[:,-1]-4*q[:,-2]+q[:,-3])/(2*dy); return o

    psi_top_np,mu_top_np,Wtop,_,_=far_profile(xi_np,YMAX)
    psi_top=torch.tensor(psi_top_np); mu_top=torch.tensor(mu_top_np)
    psi_bc=torch.zeros((NX,NY)); psi_bc[0]=MI*y; psi_bc[-1]=0; psi_bc[:,0]=0; psi_bc[:,-1]=psi_top
    npsi=(NX-2)*(NY-2)
    mumask=np.zeros((NX,NY),bool); mumask[1:,:-1]=True; mumask[-1,0]=False
    mask_t=torch.tensor(mumask)

    def unpack(z):
        psi=psi_bc.clone(); psi[1:-1,1:-1]=z[:npsi].reshape(NX-2,NY-2)
        W=torch.exp(z[npsi:npsi+NY]); rv=z[npsi+NY:]
        mu=torch.empty((NX,NY)); mu[0]=mu_s; mu[:,-1]=mu_top; mu[-1,0]=0.0; mu[mask_t]=torch.sigmoid(rv)
        rho,a2=rho_from_mu(psi,mu); return psi,W,mu,rho,a2

    def res(z):
        psi,W,mu,rho,a2=unpack(z)
        Wp=torch.cat([((-3*W[0]+4*W[1]-W[2])/(2*dy))[None],(W[2:]-W[:-2])/(2*dy),((3*W[-1]-4*W[-2]+W[-3])/(2*dy))[None]])
        gg=sp(y)[None,:]+XI*Wp[None,:]
        px=dx(psi); py=dyop(psi); gx=px/W[None,:]; gy=py-gg*px/W[None,:]
        q2m=gx*gx+gy*gy; K,Kp=KKP(psi)
        Fxi=((1+gg*gg)*px/W[None,:]-gg*py)/rho; Fy=(W[None,:]*py-gg*px)/rho
        pde=dx(Fxi)+dyop(Fy)+W[None,:]*rho**G/(G-1)*Kp
        rp=(pde[1:-1,1:-1]/(1+W[None,1:-1])).reshape(-1)
        comp=(q2m/(rho*rho*a2)-mu)[mask_t]
        rs=(px[0,1:-1]+W[1:-1]*rho_s[1:-1]*v_s[1:-1])/MI
        rsym=(Wp[0]/(1+W[0]))[None]
        rfar=((W[-1]-Wtop)/(1+Wtop))[None]
        return torch.cat([rp,comp,rs,rsym,rfar])

    if previous is None:
        W0=0.55+WFAR_SLOPE*y_np*np.tanh(y_np/.5)+WOFF*np.tanh(y_np/.5)
        # blend simple shock/body interpolation to correct far profile
        base=MI*y_np[None,:]*(1-xi_np[:,None])
        blend=(y_np/YMAX)**4
        psi0=base*(1-blend[None,:])+psi_top_np[:,None]*blend[None,:]
        # fix orientation: previous line broadcasts wrongly for top profile; build by columns
        psi0=base.copy()
        for j,b in enumerate(blend): psi0[:,j]=(1-b)*base[:,j]+b*psi_top_np
        mu0=(1-xi_np[:,None])*mu_s.detach().numpy()[None,:]+xi_np[:,None]*(y_np[None,:]/YMAX)*mu_top_np[:,None]
        for j,b in enumerate(blend): mu0[:,j]=(1-b)*mu0[:,j]+b*mu_top_np
    else:
        p=np.load(previous); xo=p['xi']; yo=p['y']; po=p['psi']; wo=p['W']; mo=p['mu']
        psi0=np.empty((NX,NY)); mu0=np.empty((NX,NY)); W0=np.empty(NY)
        Ipsi=RegularGridInterpolator((xo,yo),po,bounds_error=False,fill_value=None)
        Imu=RegularGridInterpolator((xo,yo),mo,bounds_error=False,fill_value=None)
        for j,yj in enumerate(y_np):
            if yj <= yo[-1]+1e-12:
                pts=np.column_stack([xi_np,np.full(NX,yj)])
                psi0[:,j]=Ipsi(pts); mu0[:,j]=Imu(pts); W0[j]=np.interp(yj,yo,wo)
            else:
                psj,muj,Wj,_,_=far_profile(xi_np,float(yj))
                psi0[:,j]=psj; mu0[:,j]=muj; W0[j]=Wj
    psi0[0]=MI*y_np; psi0[-1]=0; psi0[:,0]=0; psi0[:,-1]=psi_top_np
    mu0[0]=mu_s.detach().numpy(); mu0[:,-1]=mu_top_np; mu0[-1,0]=0
    W0[-1]=Wtop
    sig=np.clip(mu0[mumask],1e-6,1-1e-6); rv=np.log(sig/(1-sig))
    z0=np.concatenate([psi0[1:-1,1:-1].reshape(-1),np.log(np.maximum(W0,1e-8)),rv])

    def fun(z):
        with torch.no_grad(): return res(torch.tensor(z)).numpy()
    def jac(z):
        zz=torch.tensor(z,requires_grad=True); return torch.autograd.functional.jacobian(res,zz,vectorize=True).detach().numpy()
    r0=fun(z0); t=time.time(); sol=least_squares(fun,z0,jac=jac,method='trf',x_scale='jac',xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=maxfev,verbose=0); elapsed=time.time()-t
    rr=fun(sol.x); psi,W,mu,rho,a2=unpack(torch.tensor(sol.x)); psi=psi.detach().numpy(); W=W.detach().numpy(); mu=mu.detach().numpy(); rho=rho.detach().numpy()
    body=A*np.log(np.cosh(y_np))+W; b1=np.gradient(body,dy,edge_order=2); b2=np.gradient(b1,dy,edge_order=2)
    st=dict(NX=NX,NY=NY,YMAX=YMAX,success=bool(sol.success),res=float(np.max(np.abs(rr))),rms=float(np.sqrt(np.mean(rr*rr))),delta=float(W[0]),Mmax=float(np.sqrt(mu.max())),Mbodyfar=float(np.sqrt(mu_top_np[-1])),Mshockfar=float(np.sqrt(mu_top_np[0])),Wmin=float(W.min()),bprime0=float(b1[0]),bprimefar=float(b1[-1]),b2nose=float(b2[0]),Wtop=float(W[-1]),Wtop_exact=float(Wtop),elapsed=elapsed,nfev=int(sol.nfev),njev=int(getattr(sol,'njev',-1)),initial=float(np.max(np.abs(r0))))
    print(st,flush=True)
    out=f'inverse_bow_entropyfar_{NX}x{NY}_Y{YMAX:g}.npz'
    if save: np.savez(out,xi=xi_np,y=y_np,psi=psi,W=W,mu=mu,rho=rho,body=body,res=rr,psi_top=psi_top_np,mu_top=mu_top_np)
    return st,out

if __name__=='__main__':
    print('far constants',dict(pfar=PF,m=MF,C0=C0,Woffset=WOFF,W_slope=WFAR_SLOPE,Mbody=np.sqrt(asym_state_np(0.0)[3]),Mshock=np.sqrt(MUF)))
    run(9,17,4.0,previous='inverse_bow_solution_branch_safe.npz')
