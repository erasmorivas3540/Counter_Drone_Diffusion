import numpy as np, sys, time
from scipy.optimize import minimize_scalar
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cdd import Scenario, ring_sensors, solve
d,R=2.0,6.5
def F(a,D,nr,nth=128):
    scn=Scenario(r0=d,R=R,v=1.0,sigma=np.sqrt(2*D),lam0=1.0,rho=1.0,sensors=ring_sensors(6,a),kernel='gaussian',outer='reflect')
    sol=solve(scn,nr=nr,ntheta=nth); j=np.argmin(np.abs(sol.theta-np.pi/6)); return sol.u[-1,j]   # midpoint (proved worst angle)
for D,nr,lo,hi in [(0.005,2400,2.3,2.8),(0.01,1600,2.3,2.8),(0.02,1200,2.3,2.9),(0.05,800,2.4,3.0),(8.0,400,3.5,6.4),(32.0,400,3.5,6.4)]:
    t=time.time(); res=minimize_scalar(lambda a:F(a,D,nr),bounds=(lo,hi),method='bounded',options={'xatol':2e-3})
    pred=2.452775+3.317216*D if D<1 else None
    print(f'D={D}: a*={res.x:.3f} F={res.fun:.5f}' + (f'  first-order prediction {pred:.3f}' if pred else '') + f'  ({time.time()-t:.0f}s)',flush=True)
