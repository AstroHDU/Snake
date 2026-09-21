"""Deterministic spatial / 3D / RV-limited tangential partition diagnostic.
This diagnostic does not change the full-system SRI or delete input members.
"""
from __future__ import annotations
import json
import numpy as np
from sri_preprocess.members import member_pair
from sri_preprocess.geometry import mst_edges_from_distance


def cloud_matrices(clouds):
    n=len(clouds);D=np.zeros((n,n));W=D.copy()
    for i in range(n):
        for j in range(i+1,n):
            D[i,j]=D[j,i]=member_pair(clouds[i],clouds[j])['distance']
            # member_pair already uses the shared robust H metric and /sqrt(d).
            # Its unit H-ellipsoid radius therefore has this dimensionless scale.
            # This is a metric scale, not a centre-error or confidence interval.
            dim=np.asarray(clouds[i]).shape[1]
            W[i,j]=W[j,i]=1.0/np.sqrt(dim)
    return D,W


def split_tree(n,edges):
    adjacency=[[] for _ in range(n)]
    for i,j,_ in edges[1:]:adjacency[i].append(j);adjacency[j].append(i)
    parts=[];seen=set()
    for k in range(n):
        if k in seen:continue
        todo=[k];seen.add(k);part=[]
        while todo:
            i=todo.pop();part.append(i)
            for j in adjacency[i]:
                if j not in seen:seen.add(j);todo.append(j)
        parts.append(tuple(sorted(part)))
    return parts


def point_tree(D):
    # Kruskal including zero-length edges, deterministic under equal distances.
    n=len(D);parent=list(range(n));edges=[]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for d,i,j in sorted((float(D[i,j]),i,j) for i in range(n) for j in range(i+1,n)):
        a,b=root(i),root(j)
        if a!=b:parent[a]=b;edges.append((i,j,d))
    return sorted(edges,key=lambda t:-t[2])


def partition(D,W=None,prominence=None,limited=None):
    """One longest-edge split per step; limited restricts 2D cuts to RV-poor sides."""
    if prominence is None:
        from .config import SRIConfig
        prominence=SRIConfig().outlier_prominence
    leaves=[];tests=[]
    def rec(idx):
        idx=np.asarray(idx,int);n=len(idx)
        if n<3:leaves.append(tuple(idx));return
        d=D[np.ix_(idx,idx)]
        tree=(point_tree(d) if W is None else sorted([(i,j,float(d[i,j])) for i,j in mst_edges_from_distance(d)],key=lambda t:(-t[2],t[0],t[1])))
        L=tree[0][2];ref=tree[1][2]
        if W is not None:
            widths=[W[idx[i],idx[j]] for i,j,_ in tree];ref=max(ref,widths[0],float(np.median(widths[1:])))
        ratio=L/ref if ref>0 else np.inf if L>0 else 1.
        parts=[tuple(idx[list(p)]) for p in split_tree(n,tree)]
        allowed=limited is None or any(all(limited[i] for i in p) for p in parts)
        cut=ratio>prominence and allowed
        tests.append(dict(component=idx.tolist(),left=int(idx[tree[0][0]]),right=int(idx[tree[0][1]]),gap=L,reference=ref,ratio=ratio,accepted=bool(cut),limited_side_condition=bool(allowed)))
        if cut:
            for part in parts:rec(part)
        else:leaves.append(tuple(idx))
    rec(range(len(D)));return tuple(sorted(leaves)),tests


def choose_core(parts,counts):
    if len(parts)<2:return None
    sizes=[(len(p),sum(int(counts[i]) for i in p)) for p in parts]
    best=max(sizes);winners=[p for p,s in zip(parts,sizes) if s==best]
    return winners[0] if len(winners)==1 else None


def review_state(s,cfg,return_core_labels=False):
    labels=np.asarray(s.labels);n=len(labels)
    good=np.asarray(s.valid,bool)&(np.asarray(s.N_RV)>=cfg.min_rv_sources)&np.isfinite(s.uvw).all(1)
    row=dict(KPD_mark=None,candidate=False,can_evaluate=n>=3,Nentity=n,M_RV=int(good.sum()),
        space_split=False,velocity3d_split=False,tangent2d_split=False,partition='',spatial_partition='',
        unassessed_entities=';'.join(labels[~good]),core_entities='',core_selection_status='not_nominated',cuts_json='[]')
    if n<3:return (row,None) if return_core_labels else row
    X,xt=partition(*cloud_matrices([s.spatial[k] for k in labels]),prominence=cfg.outlier_prominence)
    final=[];assignments=[];traces=[dict(channel='space',**t) for t in xt];ambiguous=False
    for block in X:
        idx=np.array(block,int);limited=~good[idx]
        pieces=[tuple(range(len(idx)))]
        if limited.any() and len(idx)>=3:
            td,tw=cloud_matrices([s.free[labels[i]] for i in idx])
            pieces,tt=partition(td,tw,cfg.outlier_prominence,limited=limited)
            row['tangent2d_split']|=len(pieces)>1
            for t in tt:
                traces.append(dict(channel='tangent_missing_rv',**{**t,'component':idx[t['component']].tolist(),'left':int(idx[t['left']]),'right':int(idx[t['right']])}))
        for local in pieces:
            group=idx[list(local)];known=group[good[group]];unknown=group[~good[group]]
            if not len(known):final.append(tuple(group));continue
            v=np.asarray(s.uvw)[known];vd=np.linalg.norm(v[:,None]-v[None,:],axis=-1)
            vp,vt=partition(vd,prominence=cfg.outlier_prominence)
            groups=[list(known[list(p)]) for p in vp];row['velocity3d_split']|=len(vp)>1
            for t in vt:
                traces.append(dict(channel='velocity3d',**{**t,'component':known[t['component']].tolist(),'left':int(known[t['left']]),'right':int(known[t['right']])}))
            if len(groups)==1:groups[0].extend(unknown.tolist())
            elif len(unknown):
                # No measured radial information is invented for membership.
                pooled=[np.concatenate([s.free[labels[i]] for i in part],axis=0) for part in groups]
                for i in unknown:
                    distances=np.asarray([member_pair(s.free[labels[i]],cloud)['distance'] for cloud in pooled])
                    winners=np.flatnonzero(distances==distances.min())
                    assignments.append(dict(entity=str(labels[i]),distances=distances.tolist(),reference_groups=[[str(labels[j]) for j in part if good[j]] for part in groups[:len(pooled)]],unique=len(winners)==1))
                    if len(winners)==1:groups[int(winners[0])].append(int(i))
                    else:ambiguous=True;groups.append([int(i)])
            final.extend(tuple(sorted(p)) for p in groups)
    row['space_split']=len(X)>1
    candidate=bool(row['space_split'] or row['velocity3d_split'] or row['tangent2d_split'])
    fmt=lambda pp:' | '.join(';'.join(labels[list(p)]) for p in sorted(pp,key=lambda p:(-len(p),p)))
    counts=[len(s.spatial[k]) for k in labels]
    core=choose_core(final,counts) if candidate and not ambiguous else None
    status=('not_nominated' if not candidate else 'rv_assignment_undetermined' if ambiguous else 'tied_entity_and_star_counts' if core is None else 'core_too_small' if len(core)<2 else 'selected')
    row.update(KPD_mark=int(candidate),candidate=candidate,partition=fmt(final),spatial_partition=fmt(X),core_selection_status=status,
        core_entities=';'.join(labels[list(core)]) if core is not None else '',N_partition=len(final),
        cuts_json=json.dumps(traces,ensure_ascii=False),partition_complete=not ambiguous,assignments_json=json.dumps(assignments,ensure_ascii=False),partition_member_counts_json=json.dumps([{'entities':[str(labels[i]) for i in part],'member_count':sum(counts[i] for i in part)} for part in final],ensure_ascii=False))
    assert sorted(i for p in final for i in p)==list(range(n))
    core_labels=tuple(labels[list(core)]) if core is not None else None
    return (row,core_labels) if return_core_labels else row