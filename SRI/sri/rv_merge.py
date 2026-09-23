"""Radial-velocity conflict veto, with spatial/tangential compatibility evaluated first."""
import numpy as np
from sri_preprocess import coordinates as co
from sri_preprocess.dynamic_rv import compute_rv_evidence
from .uncertainty import UncertaintyError

def gated_components(nodes,stars,prep,threshold=3.):
    pairs={tuple(sorted((r.left,r.right))):r._asdict() for r in prep['strict_pairs'].itertuples(index=False)}
    summaries={};audit=[]
    for n in nodes.itertuples():
        g=stars[stars.id_node==n.id_node];ev=compute_rv_evidence(g.radial_velocity,g.radial_velocity_error)
        B=co.icrs_velocity_to_galactic(np.full(3,n.RA),np.full(3,n.DEC),[1,0,0],[0,1,0],[0,0,1]).T
        summaries[n.id_node]=(ev.n_valid,B@np.array([n.Vra,n.Vdec,ev.center_median]),B@np.diag([n.e_Vra**2,n.e_Vdec**2,ev.sigma_sys**2])@B.T,B[:,2])
    allowed={}
    for key in pairs:
        a,b=[summaries[k] for k in key];z=np.nan;status='insufficient_RV'
        if a[0]>=3 and b[0]>=3:
            direction=a[3]+b[3];direction/=np.linalg.norm(direction)
            variance=float(direction@(a[2]+b[2])@direction)
            if not np.isfinite(variance) or variance<=0:raise UncertaintyError(f'RV_merge covariance invalid: {key}')
            z=abs(float((a[1]-b[1])@direction))/np.sqrt(variance);status='assessed'
        allowed[key]=not np.isfinite(z) or z<=threshold
        audit.append(dict(left=key[0],right=key[1],Z_RV=z,RV_status=status,allowed=allowed[key]))
    groups=[(x,) for x in sorted(nodes.id_node)]
    while True:
        options=[]
        for i,left in enumerate(groups):
            for j in range(i+1,len(groups)):
                right=groups[j];keys=[tuple(sorted((a,b))) for a in left for b in right]
                rx=min(pairs[k]['R_X'] for k in keys);vt=max(pairs[k]['R_V'] for k in keys)
                if rx<=1 and vt<=1 and all(allowed[k] for k in keys):options.append((max(rx,vt),vt,rx,left,right,i,j))
        if not options:break
        _,_,_,left,right,i,j=min(options);groups[i]=tuple(sorted(left+right));del groups[j];groups.sort()
    return [list(g) for g in groups],audit
