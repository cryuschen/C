"""Illustrative spatial selectivity and stable neural-mass forward simulation.

These geometries, couplings and gains are assumptions, NOT inferred anatomy.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import root
from scipy.integrate import solve_ivp
from scipy.special import expit
from scipy.signal import fftconvolve
from .data import save_json, save_csv


def spatial_encoding():
    axis=np.linspace(-1,1,129);xx,yy=np.meshgrid(axis,axis)
    right=((xx>=-.6)&(xx<=.6)&(np.abs(yy)<=.5*(.6-xx)/1.2)).astype(float)
    left=right[:,::-1].copy()
    missing=right.copy();missing[:,axis>.15]=0
    scrambled=np.roll(right,43,axis=0);scrambled[:64]=np.roll(scrambled[:64],40,axis=1)
    images=np.stack([left,right,missing,scrambled])
    dog=np.stack([gaussian_filter(a,.8)-gaussian_filter(a,2.) for a in images])
    # Eight local edge-normal orientations. Pooling tolerates <= 2-pixel shifts.
    gx=np.stack([gaussian_filter(a,1.,order=(0,1)) for a in dog])
    gy=np.stack([gaussian_filter(a,1.,order=(1,0)) for a in dog])
    angles=np.arange(8)*np.pi/8
    response=np.stack([np.abs(gx*np.cos(a)+gy*np.sin(a)) for a in angles],axis=1)
    response=maximum_filter(response,size=(1,1,5,5))
    # Sample the three edges, each at 25%, 50%, 75% of its length.
    vertices=np.array([[-.6,-.5],[-.6,.5],[.6,0]])
    features=[];points=[]
    for mirror in (-1,1):
        v=vertices.copy();v[:,0]*=mirror; samples=[];points_group=[]
        for a,b in zip(v,np.roll(v,-1,axis=0)):
            delta=b-a;normal=np.array([-delta[1],delta[0]])
            theta=np.arctan2(normal[1],normal[0])%np.pi
            orient=int(np.argmin(np.abs(np.angle(np.exp(2j*(angles-theta))))))
            for weight in (.25,.5,.75):
                pos=a+weight*(b-a); ix=int(np.argmin(abs(axis-pos[0])));iy=int(np.argmin(abs(axis-pos[1])))
                samples.append(response[:,orient,iy,ix]);points_group.append([ix,iy,orient])
        features.append(np.exp(np.log(np.maximum(np.stack(samples),1e-9)).mean(0)))
        points.append(points_group)
    f=np.stack(features,axis=1)
    f/=np.max(f[:2],axis=0)
    common=np.mean(response,axis=(1,2,3));common/=max(common[:2])
    inputs=np.column_stack([f,common]);eta=(f[:,1]-f[:,0])/(f.sum(1)+1e-12)
    return dict(images=images,dog=dog,responses=response,inputs=inputs,eta=eta,
                points=np.array(points),names=['left','right','missing_tip','scrambled'])


def neural_forward(inputs):
    W=np.ones((3,3))-np.eye(3);W/=W.sum(1,keepdims=True)
    def derivative(t,state,feature):
        g,e,i=state.reshape(3,3)
        drive=3*feature if 0<=t<.203125 else np.zeros(3)
        return np.concatenate([(-g+drive)/.02,
            (-e+expit(e-i+.1*W@e+g-2))/.02,
            (-i+expit(e-i-2))/.04])
    equilibrium=root(lambda v:derivative(-1,v,np.zeros(3)),np.r_[np.zeros(3),np.full(6,.12)])
    if not equilibrium.success:raise RuntimeError('No simulation equilibrium')
    state=equilibrium.x
    eps=1e-6
    jac=np.column_stack([(derivative(-1,state+eps*np.eye(9)[j],np.zeros(3))-
                          derivative(-1,state-eps*np.eye(9)[j],np.zeros(3)))/(2*eps) for j in range(9)])
    eig=np.linalg.eigvals(jac)
    if max(eig.real)>=0:raise RuntimeError('Illustrative resting state is unstable')
    t=np.arange(-250,1501)/1000
    source=np.array([[-.4,0,.3],[.4,0,.3],[0,-.3,.3]])
    sensors=np.array([[0,.5,1],[-.6,.5,1],[.6,.5,1]])
    ref=np.array([0,-1,1])
    def lead(e):
        d=e[None,:]-source
        return d[:,2]/np.linalg.norm(d,axis=1)**3
    L=np.stack([lead(e)-lead(ref) for e in sensors]);L/=np.linalg.norm(L)
    values=[];q=[];eeg=[]
    ht=np.arange(2001)/1000
    he=ht/.025**2*np.exp(-ht/.025);hi=ht/.075**2*np.exp(-ht/.075)
    for feature in inputs:
        sol=solve_ivp(lambda tt,xx:derivative(tt,xx,feature),(-.25,1.5),state,t_eval=t,
                      max_step=.001,rtol=1e-7,atol=1e-9)
        if not sol.success:raise RuntimeError(sol.message)
        de=sol.y[3:6]-state[3:6,None];di=sol.y[6:9]-state[6:9,None]
        qq=np.stack([fftconvolve(a,he)[:len(t)]*.001-.7*fftconvolve(b,hi)[:len(t)]*.001 for a,b in zip(de,di)])
        values.append(sol.y);q.append(qq);eeg.append(L@qq)
    return dict(t=t,state=np.stack(values),dipoles=np.stack(q),eeg=np.stack(eeg),
                lead=L,sources=source,sensors=sensors,reference=ref,equilibrium=state,eigenvalues=eig)


def run_mechanism(out):
    dest=Path(out)/'mechanism';dest.mkdir(parents=True,exist_ok=True)
    spatial=spatial_encoding();dyn=neural_forward(spatial['inputs'])
    np.savez_compressed(dest/'spatial_encoding.npz',**spatial)
    np.savez_compressed(dest/'neural_forward.npz',**dyn)
    save_csv([dict(stimulus=n,f_left=f[0],f_right=f[1],f_common=f[2],eta=float(e))
              for n,f,e in zip(spatial['names'],spatial['inputs'],spatial['eta'])],dest/'selectivity.csv')
    save_json(dict(status='illustrative_assumptions_not_fitted_to_EEG',grid_size=129,
        LGN_tau_ms=20,E_tau_ms=20,I_tau_ms=40,w_EE=1,w_EI=1,w_IE=1,w_II=1,
        spatial_coupling=.1,background=-2,pulse_seconds=.203125,synapse_E_ms=25,synapse_I_ms=75,
        inhibitory_dipole_weight=.7,max_equilibrium_eigenvalue_real=float(max(dyn['eigenvalues'].real)),
        geometry_units='arbitrary_normalized_coordinates',sources=dyn['sources'],sensors=dyn['sensors'],
        lead=dyn['lead'],reference=dyn['reference'],
        source='https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2014.00080/full'),dest/'assumptions.json')
    return spatial,dyn
