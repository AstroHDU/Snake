import numpy as np
from scipy.spatial import cKDTree
from . import numerics
def covariance_root(s):
    e,q=np.linalg.eigh(.5*(s+s.T))
    tol=len(e)*np.finfo(float).eps*max(float(np.max(np.abs(e))),0.)
    if e.min() < -tol:raise numerics.UndefinedComponent('member covariance is not numerically PSD')
    return (q*np.sqrt(np.maximum(e,0.)))@q.T

def transform_clouds(a,b):
    from .fixed_components import transform_clouds as fixed_transform
    return fixed_transform(a,b)

def member_pair(a,b):
    """Return both role responses from precisely the same common distance."""
    a,b=transform_clouds(a,b)
    da=cKDTree(b).query(a,k=1)[0]/np.sqrt(a.shape[1])
    db=cKDTree(a).query(b,k=1)[0]/np.sqrt(a.shape[1])
    distance=float(.5*(np.median(da)+np.median(db)))
    return dict(distance=distance,spatial=float(1/np.hypot(1,distance)),velocity=float(np.exp(-distance)))

def spatial_member_edge(a,b):
    return member_pair(a,b)['spatial']

def velocity_member_edge(a,b):
    return member_pair(a,b)['velocity']

def member_tree(spatial_clouds,velocity_clouds,edges):
    if not edges:raise numerics.UndefinedComponent('No fixed member edges')
    spatial=[spatial_member_edge(spatial_clouds[a],spatial_clouds[b]) for a,b in edges]
    velocity=[velocity_member_edge(velocity_clouds[a],velocity_clouds[b]) for a,b in edges]
    return dict(spatial_mem_node=float(np.mean(spatial)),velocity_mem=float(np.mean(velocity)),member_edges=len(edges))

def harmonic(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return np.divide(2*a*b,a+b,out=np.zeros(np.broadcast(a,b).shape),where=a+b>0)
