import numpy as np
from scipy.optimize import minimize
d,R,Da,K=2.0,6.5,1.0,4
xg,wg=np.polynomial.legendre.leggauss(120); r=d+(R-d)*(xg+1)/2; w=wg*(R-d)/2
TH=np.linspace(0,2*np.pi,1440,endpoint=False); X=(r[:,None]*np.exp(1j*TH))  # (nr, nth)
def expo(p):
    s=p[:K]+1j*p[K:]
    return Da*np.einsum('i,ijk->j',w,np.exp(-np.abs(X[:,:,None]-s)**2/2))
def jac_expo(p):
    s=p[:K]+1j*p[K:]; diff=X[:,:,None]-s; g=np.exp(-np.abs(diff)**2/2)
    gx=np.einsum('i,ijk->jk',w,g*diff.real)*Da; gy=np.einsum('i,ijk->jk',w,g*diff.imag)*Da
    return np.hstack([gx,gy])
ring=1.822662*np.exp(2j*np.pi*np.arange(K)/K); ring_val=expo(np.r_[ring.real,ring.imag]).min()
print('ring min exposure', ring_val)
rng=np.random.default_rng(0); best=[]
starts=[np.r_[ring.real,ring.imag]+rng.normal(0,0.3,2*K) for _ in range(5)]
for _ in range(40):
    rad=rng.uniform(0,R,K); ang=rng.uniform(0,2*np.pi,K); s=rad*np.exp(1j*ang); starts.append(np.r_[s.real,s.imag])
for p0 in starts:
    z0=np.r_[p0,expo(p0).min()]
    cons={'type':'ineq','fun':lambda z: expo(z[:-1])-z[-1],'jac':lambda z: np.hstack([jac_expo(z[:-1]),-np.ones((TH.size,1))])}
    res=minimize(lambda z:-z[-1],z0,jac=lambda z:np.r_[np.zeros(2*K),-1.0],constraints=[cons],method='SLSQP',
                 bounds=[(-R,R)]*(2*K)+[(None,None)],options={'maxiter':500,'ftol':1e-12})
    v=expo(res.x[:-1]).min(); best.append((v,res.x[:-1]))
best.sort(key=lambda t:-t[0])
vals=np.array([b[0] for b in best]); print('best',vals[:5],'median',np.median(vals),'n beating ring by >1e-4:',int((vals>ring_val+1e-4).sum()),'of',len(vals))
s=best[0][1][:K]+1j*best[0][1][K:]; print('best layout radii',np.round(np.sort(np.abs(s)),3),'angles',np.round(np.sort(np.degrees(np.angle(s))%360),1))
pass
