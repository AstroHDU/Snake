from __future__ import annotations
import inspect
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from .dynamic_rv import standardized_pairwise_distance as ORIGINAL_DISTANCE
_default_rcond=inspect.signature(np.linalg.pinv).parameters["rcond"].default
PINV_RCOND=float(1e-15 if _default_rcond is None else _default_rcond)
class UndefinedComponent(ValueError):
    """Preserve the input row with an explicit error; do not substitute a score."""

def graph_mst(node_ids,pairs):
    names=list(node_ids);n=len(names);index={k:i for i,k in enumerate(names)}
    if len(index)!=n:raise UndefinedComponent('duplicate_entity_ID')
    columns=['left','right','primary_merge_cost','support_merge_cost','log_cost','log_gap_to_next']
    if n<2:return pd.DataFrame(columns=columns)
    edges=[];lookup={}
    for r in pairs.itertuples(index=False):
        if r.left not in index or r.right not in index:raise UndefinedComponent('unknown_entity_ID')
        i,j=index[r.left],index[r.right]
        if i==j:raise UndefinedComponent('self_edge_in_pair_table')
        value=float(r.primary_merge_cost)
        if not np.isfinite(value):continue
        if value<0:raise UndefinedComponent('negative_merge_cost')
        key=tuple(sorted([i,j]))
        if key in lookup:raise UndefinedComponent('duplicate_pair_edge')
        lookup[key]=r;edges.append((value,key[0],key[1]))
    if not edges:raise UndefinedComponent('disconnected_finite_pair_graph')
    chosen=[]
    if any(value==0 for value,_,_ in edges):
        # Explicit edges avoid scipy's zero-as-absence representation. Only
        # zero-cost cases use this tie-deterministic Kruskal branch.
        parent=list(range(n))
        def find(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        for cost,i,j in sorted(edges):
            a,b=find(i),find(j)
            if a!=b:parent[a]=b;chosen.append((i,j,cost))
            if len(chosen)==n-1:break
    else:
        rr=[];cc=[];vv=[]
        for value,i,j in edges:rr.extend([i,j]);cc.extend([j,i]);vv.extend([value,value])
        matrix=csr_matrix((vv,(rr,cc)),shape=(n,n))
        tree=minimum_spanning_tree(matrix).tocoo()
        chosen=[(int(i),int(j),float(v)) for i,j,v in zip(tree.row,tree.col,tree.data)]
    if len(chosen)!=n-1:raise UndefinedComponent('disconnected_finite_pair_graph')
    rows=[]
    for i,j,cost in chosen:
        source=lookup[tuple(sorted([i,j]))]
        rows.append(dict(left=names[i],right=names[j],primary_merge_cost=cost,support_merge_cost=float(source.support_merge_cost)))
    result=pd.DataFrame(rows).sort_values('primary_merge_cost').reset_index(drop=True)
    result['log_cost']=np.log(np.maximum(result.primary_merge_cost,np.finfo(float).tiny))
    result['log_gap_to_next']=result.log_cost.shift(-1)-result.log_cost
    return result

def standardized_pairwise_distance(points,covariances):
    x=np.asarray(points,float);cov=np.asarray(covariances,float)
    if x.ndim!=2 or cov.shape!=(len(x),x.shape[1],x.shape[1]):raise ValueError('points/covariances shape mismatch')
    valid=np.isfinite(x).all(axis=1)&np.isfinite(cov).all(axis=(1,2))
    ids=np.flatnonzero(valid)
    for i in ids:
        eig=np.linalg.eigvalsh((cov[i]+cov[i].T)/2);scale=float(np.max(np.abs(eig)))
        if eig.min() < -len(eig)*np.finfo(float).eps*scale:raise UndefinedComponent('non_PSD_entity_covariance')
    for j,i in enumerate(ids):
        for k in ids[j+1:]:
            pair=(cov[i]+cov[k]);eig=np.linalg.eigvalsh((pair+pair.T)/2)
            if eig.max()<=0 or eig.min()<=PINV_RCOND*eig.max():
                raise UndefinedComponent('singular_or_numerically_unresolved_pair_covariance')
    # Preserve the exact established numerical solver for the admitted domain.
    return ORIGINAL_DISTANCE(points,covariances)
