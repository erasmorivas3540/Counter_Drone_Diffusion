import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf
def h(a,t,d,R):  # their closed form single-sensor ray exposure
    return np.sqrt(np.pi/2)*np.exp(-a**2*np.sin(t)**2/2)*(erf((R-a*np.cos(t))/np.sqrt(2))-erf((d-a*np.cos(t))/np.sqrt(2)))
def V_layout(s,d,R,nth=7200):
    th=np.linspace(0,2*np.pi,nth,endpoint=False); a=np.abs(s); ph=np.angle(s)
    return min_I(th,a,ph,d,R)
def min_I(th,a,ph,d,R): return np.min(sum(h(a[j],th-ph[j],d,R) for j in range(len(a))))
def ring(K,a): return a*np.exp(2j*np.pi*np.arange(K)/K)
def best_ring(K,d,R,center=False):
    f=lambda a:-V_layout(np.r_[ring(K-1 if center else K,a),[0] if center else []],d,R)
    grid=np.linspace(0.01,R,300); i=np.argmin([f(a) for a in grid])
    res=minimize_scalar(f,bounds=(grid[max(i-1,0)],grid[min(i+1,299)]),method='bounded',options={'xatol':1e-7})
    return res.x,-res.fun
d,R=2.0,6.5
# (a) closed form h vs quadrature, and K=6 ring
print('K6 ring', best_ring(6,d,R))
# (b) local max test: random small perturbations of the hexagon
a0,V0=best_ring(6,d,R); s0=ring(6,a0); rng=np.random.default_rng(0); worst=-1
for eps in (1e-2,3e-3):
    gains=[]
    for _ in range(400):
        p=rng.normal(size=6)+1j*rng.normal(size=6); p-=1j*np.angle(np.exp(1j*0))*0
        p=p/np.linalg.norm(np.r_[p.real,p.imag])*eps
        # remove pure rotation component
        rot=1j*s0/np.linalg.norm(np.r_[s0.real,s0.imag]); p=p-np.real(np.vdot(rot,p))*rot
        gains.append(V_layout(s0+p,d,R,14400)-V0)
    print(f'eps={eps}: max change in V over 400 random directions = {max(gains):+.2e} (negative => local max)')
# (c) Q2 table spot checks
for K in (3,5,7,9):
    print(f'K={K}: single ring {best_ring(K,d,R)}  center+ring {best_ring(K,d,R,center=True)}')
# (d) collapse condition and the d=1 case
for dd,RR in ((1,2.0),(1,5.5),(1,6.5)):
    r=np.linspace(dd,RR,20001); A2=np.trapezoid((r**2-2)*np.exp(-r**2/2),r); A0=np.trapezoid(np.exp(-r**2/2),r)
    a,v=best_ring(3,dd,RR); print(f'd={dd},R={RR}: A2={A2:+.6f}; K=3 best ring a={a:.4f} V={v:.6f} vs center {3*A0:.6f}')
