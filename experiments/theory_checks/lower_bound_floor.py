import numpy as np
from scipy.special import i0e
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize, minimize_scalar
d,R,Da,K=2.0,6.5,1.0,6
N=3000; r=np.linspace(d,R,N+1); h=r[1]-r[0]
def kbar(a): return Da*sum(i0e(r*aj)*np.exp(-(r-aj)**2/2) for aj in np.atleast_1d(a))
def u_R(kap,D):
    # D(u''+u'/r) - u' - kap u = 0 ; u(d)=1, u'(R)=0 ; unknowns r[1..N]; conservative central
    ri=r[1:]; P=D*(ri+h/2)/(ri*h*h)-1/(2*h)*(-1)*0; 
    dp=D*(ri+h/2)/(ri*h*h); dm=D*(ri-h/2)/(ri*h*h)
    # drift -u' : central  -> coefficients: +1/(2h) on u_{i-1}, -1/(2h) on u_{i+1}
    P=dp-1/(2*h); M=dm+1/(2*h); diag=-(dp+dm)-kap[1:]
    lower=M[1:].copy(); upper=P[:-1].copy()
    lower[-1]+=P[-1]  # ghost: u_{N+1}=u_{N-1}
    A=diags([lower,diag,upper],[-1,0,1],format='csc'); b=np.zeros(N); b[0]=-M[0]
    return spsolve(A,b)[-1]
for pe,ring_u,a_ring in [(0.5,0.44400,4.0266),(2,0.31006,3.2537),(20,0.27095,2.5979)]:
    D=1/pe
    eq=minimize_scalar(lambda a:u_R(kbar(np.full(K,a)),D),bounds=(0,6.5),method='bounded',options={'xatol':1e-6})
    rng=np.random.default_rng(1); best=eq.fun; bx=np.full(K,eq.x)
    for t in range(12):
        x0=bx if t==0 else rng.uniform(0,6.5,K)
        res=minimize(lambda a:u_R(kbar(a),D),x0,method='L-BFGS-B',bounds=[(0,6.5)]*K)
        if res.fun<best: best,bx=res.fun,res.x
    print(f'Pe={pe}: ring F={ring_u:.4f} | Jensen lower bound (all radii equal) {eq.fun:.4f} at a={eq.x:.3f} | free radii {best:.4f} radii {np.round(np.sort(bx),2)} | ring within {ring_u-best:.4f} ({100*(ring_u-best)/ring_u:.1f}%) of any layout')
