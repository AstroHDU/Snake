"""Robust member-cloud scatter and shared distance normalization."""
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.stats import chi2
from . import members,geometry

@lru_cache(None)
def calibration(d):
 q=chi2.ppf(.5,d)
 return d/(d*chi2.cdf(q,d+2)+q*chi2.sf(q,d))

def _scatter_uncached(x):
 x=np.asarray(x,float);n,d=x.shape
 if n<2 or not np.isfinite(x).all():raise ValueError('invalid member cloud')
 def accumulate(length):
  cap=np.median(length)
  if cap<=0:raise ValueError('zero pair-median scale')
  out=np.zeros((d,d));offset=0
  for i in range(n-1):
   delta=x[i+1:]-x[i];ell=length[offset:offset+len(delta)];offset+=len(delta);w=np.ones(len(delta));k=ell>cap;w[k]=(cap/ell[k])**2;out+=delta.T@(delta*w[:,None])
  return calibration(d)*out/(n*(n-1))
 pilot=accumulate(pdist(x));ev,U=np.linalg.eigh(pilot)
 if ev.min()<=d*np.finfo(float).eps*max(ev.max(),0):raise ValueError('singular robust pilot; no physical floor')
 w=(U/np.sqrt(ev))@U.T
 return accumulate(pdist((x-np.median(x,axis=0))@w))

@lru_cache(maxsize=512)
def _cached_scatter(shape,blob):
 return _scatter_uncached(np.frombuffer(blob,dtype=np.float64).reshape(shape))

def scatter(x):
 x=np.ascontiguousarray(x,dtype=np.float64)
 return _cached_scatter(x.shape,x.tobytes())

def inverse_H(a,b):
 H=members.covariance_root(scatter(a))+members.covariance_root(scatter(b));ev,U=np.linalg.eigh(H)
 if ev.min()<=len(ev)*np.finfo(float).eps*max(ev.max(),0):raise ValueError('singular shared H')
 return (U/ev)@U.T

def transform_clouds(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);w=inverse_H(a,b);origin=(a.mean(0)+b.mean(0))/2
 return (a-origin)@w,(b-origin)@w

def translated_edge(a,b,centers,return_distance=False):
 ca=np.median(a,axis=0);cb=np.median(b,axis=0);a=a-ca;b=b-cb;w=inverse_H(a,b);aa=a@w;bb=b@w;shift=(centers[:,0]-centers[:,1])@w
 da=cKDTree(bb).query(aa[None]+shift[:,None])[0];db=cKDTree(aa).query(bb[None]-shift[:,None])[0];D=(np.median(da,axis=1)+np.median(db,axis=1))/(2*np.sqrt(a.shape[1]))
 return D if return_distance else np.exp(-D)
