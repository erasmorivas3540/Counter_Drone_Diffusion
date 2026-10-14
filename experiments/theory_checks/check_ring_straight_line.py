import numpy as np
from scipy.special import i0e, ive
from scipy.optimize import minimize_scalar
d,R,Da,K=2.0,6.5,1.0,6
xg,wg=np.polynomial.legendre.leggauss(200); r=d+(R-d)*(xg+1)/2; w=wg*(R-d)/2
def B(n,a): return np.sum(w*ive(n,r*a)*np.exp(-(r-a)**2/2))
G=lambda a:B(0,a)
def I_ring(a,th):  # direct: exposure along ray theta for regular ring
    s=a*np.exp(2j*np.pi*np.arange(K)/K); x=r[:,None]*np.exp(1j*th)
    return Da*np.sum(w[:,None]*np.exp(-np.abs(x[:,:,None]-s)**2/2).sum(2),0)
def M(a): return I_ring(a,np.linspace(0,2*np.pi/K,721)).min()
aG=minimize_scalar(lambda a:-G(a),bounds=(0.1,6.5),method='bounded',options={'xatol':1e-7}).x
aM=minimize_scalar(lambda a:-M(a),bounds=(0.5,6.5),method='bounded',options={'xatol':1e-7}).x
for name,a in [('argmax G',aG),('argmax M6',aM)]:
    th=np.linspace(0,2*np.pi/K,721); I=I_ring(a,th)
    series=K*Da*(G(a)+2*sum((-1)**q*B(q*K,a) for q in range(1,8)))
    print(f'{name}: a={a:.5f} 6G={K*Da*G(a):.5f} M6={I.min():.5f} (series {series:.5f}) exp(-M)={np.exp(-I.min()):.5f} argmin theta/(pi/K)={th[I.argmin()]/(np.pi/K):.3f}')
# monotone on (0, pi/K)?
for a in (0.5,1.5,aM,4.0,6.0):
    th=np.linspace(0,np.pi/K,400); I=I_ring(a,th); print(f'a={a:.2f} decreasing on (0,pi/K): {np.all(np.diff(I)<1e-13)}')
