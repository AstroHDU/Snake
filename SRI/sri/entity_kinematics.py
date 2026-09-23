"""Member-velocity transport used only by Vmem.
A systemic motion supplies the radial component of a modeled transport vector.
It is not an observed stellar RV. The vector and diagnostic covariance here
must never supply Vcen or Cross, which use measured_kinematics independently.
The input source RV column is not overwritten.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from .observables import derive_node_table
from sri_preprocess import coordinates as co
from sri_preprocess.dynamic_rv import compute_rv_evidence


def estimate_entity_sources(sources, label):
    if sources.empty or not sources.source_id.astype(str).is_unique:
        raise ValueError('Entity needs nonempty, unique original sources')
    g=sources.copy().sort_values('source_id',kind='stable')
    # Reuse the existing scalar source-summary conventions on ONE entity.
    # The original base-node membership is not used after this relabelling.
    g['id_node']=str(label)
    row=derive_node_table(g).iloc[0].copy()
    evidence=compute_rv_evidence(g.radial_velocity,g.radial_velocity_error)
    row['RV_information']=evidence.information
    row['RV_intrinsic_scatter']=evidence.intrinsic_scatter
    row['RV_observed_scatter']=evidence.observed_scatter
    row['N_RV']=evidence.n_valid
    row['f_RV']=evidence.n_valid/len(g)
    radial=co.icrs_radial_unit(g.ra.to_numpy(float),g.dec.to_numpy(float))
    direction=radial.mean(axis=0);norm=float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm<=1e-12:
        raise ValueError('Entity has no identifiable mean sightline')
    direction/=norm
    ra=float(np.degrees(np.arctan2(direction[1],direction[0]))%360)
    dec=float(np.degrees(np.arcsin(np.clip(direction[2],-1,1))))
    row['RA']=ra;row['DEC']=dec
    # Columns of B rotate [Vra, Vdec, RV] in this exact entity basis to UVW.
    B=co.icrs_velocity_to_galactic(np.full(3,ra),np.full(3,dec),
        np.array([1.,0,0]),np.array([0.,1,0]),np.array([0.,0,1])).T
    q=np.array([row['Vra'],row['Vdec'],row['RV']],float)
    error=np.array([row['e_Vra'],row['e_Vdec'],row['e_RV']],float)
    valid=bool(np.isfinite(q).all() and np.isfinite(error[2]) and error[2]>0)
    from sri_preprocess.uncertainty_mode import ENABLED
    if not ENABLED.get():valid=bool(np.isfinite(q).all())
    Cq=np.diag(error**2)
    # Retain the member-UVW coordinate-median estimator. A single entity
    # systemic vector supplies transport, never the constituent node vectors.
    a=co.icrs_velocity_to_galactic(g.ra,g.dec,np.ones(len(g)),np.zeros(len(g)),np.zeros(len(g)))
    b=co.icrs_velocity_to_galactic(g.ra,g.dec,np.zeros(len(g)),np.ones(len(g)),np.zeros(len(g)))
    ng=radial@co.ICRS_TO_GALACTIC.T
    member_jacobian=ng[:,:,None]*(ng@B)[:,None,:]
    member_jacobian[:,:,0]+=a
    member_jacobian[:,:,1]+=b
    transported=(a*g.Vra.to_numpy(float)[:,None]+b*g.Vdec.to_numpy(float)[:,None]
        +ng*((ng@(B@q))[:,None])) if valid else np.full((len(g),3),np.nan)
    centre=np.median(transported,axis=0) if valid else np.full(3,np.nan)
    J=np.full((3,3),np.nan)
    if valid:
        for axis in range(3):
            values=transported[:,axis];order=np.sort(values)
            lo=order[(len(g)-1)//2];hi=order[len(g)//2]
            J[axis]=.5*(member_jacobian[values==lo,axis,:].mean(0)
                         +member_jacobian[values==hi,axis,:].mean(0))
    cov=J@Cq@J.T if valid and np.isfinite(error).all() else np.full((3,3),np.nan)
    seed=int(hashlib.sha256('|'.join(g.source_id.astype(str)).encode()).hexdigest()[:8],16)
    return dict(row=row,uvw=centre,covariance=cov,basis=B,center_jacobian=J,
        transported_members=transported,member_jacobian=member_jacobian,local_center=q,
        local_covariance=Cq,valid=valid,N_RV=int(row['N_RV']),seed=seed,
        covariance_status='computed' if np.isfinite(cov).all() else 'unavailable_source_errors')


def prepare_entity_kinematics(entity_members, labels, compute_covariance=False,
                              draws=4096, seed=20260609):
    estimates=[estimate_entity_sources(entity_members[k],k) for k in labels]
    rows=pd.DataFrame([e['row'] for e in estimates])
    blocks=[]
    for label in labels:
        g=entity_members[label].copy();g['id_node']=label;blocks.append(g)
    frame=pd.concat(blocks,ignore_index=True)
    # One vote per final entity, not per original base node.
    t1,t2=co.common_tangent_basis(rows,frame,weight_mode='node_equal')
    free={};projected={};cov_mc=[]
    for label,e in zip(labels,estimates):
        g=entity_members[label]
        ra=g.ra.to_numpy(float);dec=g.dec.to_numpy(float)
        tangent=co.icrs_velocity_to_galactic(ra,dec,g.Vra.to_numpy(float),
                                            g.Vdec.to_numpy(float),np.zeros(len(g)))
        free[label]=np.column_stack([tangent@t1,tangent@t2])
        if e['valid']:
            # Model projection for transport only, not measured stellar RV.
            radial=co.icrs_radial_unit(ra,dec)
            vv=e['transported_members']
            projected[label]=np.column_stack([vv@t1,vv@t2])
        if compute_covariance and np.isfinite(e['covariance']).all():
            rng=np.random.default_rng(np.random.SeedSequence([seed,e['seed']]))
            q=rng.normal(size=(draws,3))*np.sqrt(np.diag(e['local_covariance']))
            samples=[]
            for lo in range(0,draws,64):
                y=e['transported_members'][None]+np.einsum('sk,njk->snj',q[lo:lo+64],e['member_jacobian'])
                samples.append(np.median(y,axis=1))
            cov_mc.append(np.cov(np.concatenate(samples),rowvar=False,ddof=1))
        else:cov_mc.append(np.full((3,3),np.nan))
    return dict(estimates=estimates,entity_table=rows,t1=t1,t2=t2,
        free=free,projected=projected,uvw=np.array([e['uvw'] for e in estimates]),
        cov_analytic=np.array([e['covariance'] for e in estimates]),cov=np.array(cov_mc),
        valid=np.array([e['valid'] for e in estimates]),
        complete=np.array([e['valid'] for e in estimates]),
        N_RV=np.array([e['N_RV'] for e in estimates],int))


def diagnostic_table(state, scope='full'):
    if getattr(state,'measured_kinematics',None) is not None and len(state.measured_kinematics):
        table=state.measured_kinematics.copy();table['scope']=scope
        return table
    table=state.entity_kinematics.copy().rename(columns={'id_node':'entity'})
    table['scope']=scope
    for j,key in enumerate(('U','V','W')):table[key]=state.uvw[:,j]
    for i in range(3):
        for j in range(3):table[f'cov_{i}{j}']=state.cov_analytic[:,i,j]
    table['velocity_estimator']='entity_RV_transported_member_UVW_median'
    table['covariance_model']='entity_scalar_errors_median_jacobian'
    return table
