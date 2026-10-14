import numpy as np
from scipy.optimize import minimize
d,R,K=2.0,6.5,6
xg,wg=np.polynomial.legendre.leggauss(200); r=d+(R-d)*(xg+1)/2; w=wg*(R-d)/2
a_star=2.452779
mids=(2*np.arange(K)+1)*np.pi/K
P=r[:,None]*np.exp(1j*mids)            # points on the K midpoint rays
def W(y): z=y[0]+1j*y[1]; return np.sum(w[:,None]*np.exp(-np.abs(P-z)**2/2))   # sum_m k_y(theta_m), Da=1
ring=a_star*np.exp(2j*np.pi*np.arange(K)/K)
th=np.linspace(0,2*np.pi,7201); X=r[:,None]*np.exp(1j*th)
M=np.min(np.sum(w[:,None]*np.exp(-np.abs(X[:,:,None]-ring)**2/2).sum(2),0))
print('ring min exposure M_6 =',M, '  W(ring sensor) =',W([a_star,0]))
# global max of W over the plane: dense grid then polish
g=np.linspace(-8,8,801); best=(-1,None)
for x in g:
    vals=[W([x,y]) for y in g]; j=int(np.argmax(vals))
    if vals[j]>best[0]: best=(vals[j],(x,g[j]))
res=minimize(lambda y:-W(y),best[1],method='Nelder-Mead',options={'xatol':1e-10,'fatol':1e-14})
print('grid max',best, ' polished max W =',-res.fun,'at',res.x,'radius',np.hypot(*res.x),'angle deg',np.degrees(np.arctan2(res.x[1],res.x[0])))
# second-best local maxima
