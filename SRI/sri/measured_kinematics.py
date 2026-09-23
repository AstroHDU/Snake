"""Independent entity UVW estimation, separate from Vmem transport.

The default rotates valid component medians at the entity mean sightline.
Systemic component errors I^-1+s_int^2/N use each component's valid sources.
The covariance rotation assumes independent components; it is not an exact
sampling covariance of the median estimator. Missing required errors fail.
Only the component-median estimator is exposed by this package.
"""
import hashlib
import numpy as np
import pandas as pd
from sri_preprocess import coordinates as co
from sri_preprocess.dynamic_rv import compute_rv_evidence
from sri_preprocess.uncertainty_mode import scoped


def source_arrays(sources,complete_only_errors=False,use_uncertainty=True):
    g=sources.sort_values('source_id',kind='stable').copy()
    if not g.source_id.astype(str).is_unique:raise ValueError('Duplicate source in entity')
    def col(k):return pd.to_numeric(g[k],errors='coerce').to_numpy(float) if k in g else np.full(len(g),np.nan)
    ra,dec=col('ra'),col('dec');vra,vdec=col('Vra'),col('Vdec');rv,erv=col('radial_velocity'),col('radial_velocity_error')
    sky=np.isfinite(ra)&np.isfinite(dec)
    astrometry=sky&np.isfinite(col('parallax'))&(col('parallax')>0)
    masks=[astrometry&np.isfinite(vra),astrometry&np.isfinite(vdec),sky&np.isfinite(rv)&np.isfinite(erv)&(erv>0)]
    if not use_uncertainty:masks[2]=sky&np.isfinite(rv)
    complete=masks[0]&masks[1]&masks[2]
    # Transform source-local velocity bases into Galactic Cartesian axes.
    a=co.icrs_velocity_to_galactic(ra,dec,np.ones(len(g)),np.zeros(len(g)),np.zeros(len(g)))
    b=co.icrs_velocity_to_galactic(ra,dec,np.zeros(len(g)),np.ones(len(g)),np.zeros(len(g)))
    n=co.icrs_radial_unit(ra,dec)@co.ICRS_TO_GALACTIC.T
    errors=[];fallback=0
    for key,values in [('e_Vra_source',vra),('e_Vdec_source',vdec)]:
        e=col(key);valid=sky&np.isfinite(values);missing=valid&~(np.isfinite(e)&(e>0))
        # Missing measurement errors remain missing; no dispersion replacement.
        errors.append(e)
    info=compute_rv_evidence(rv[masks[2]],erv[masks[2]])
    return g,np.stack([a,b,n],axis=1),np.column_stack([vra,vdec,rv]),np.column_stack([*errors,erv]),masks,info,fallback


@scoped
def estimate_measured_entity(sources,method='component_median',draws=4096,use_uncertainty=True):
    if method!='component_median':raise ValueError('Only component_median is supported')
    if use_uncertainty:
        from .uncertainty import validate_sources
        validate_sources(sources)
    g,H,y,e,masks,rvinfo,fallback=source_arrays(sources,use_uncertainty=use_uncertainty)
    complete=masks[0]&masks[1]&masks[2]
    result=dict(method=method,N_RV=int(masks[2].sum()),N_3D=int(complete.sum()),
        N_tan1=int(masks[0].sum()),N_tan2=int(masks[1].sum()),
        rv_information=rvinfo.information,rv_sigma_sys=rvinfo.sigma_sys,
        rv_intrinsic_scatter=rvinfo.intrinsic_scatter,tangent_error_fallback_sources=fallback,
        uvw=np.full(3,np.nan),covariance=np.full((3,3),np.nan),valid=False,
        source_ids=g.loc[complete,'source_id'].astype(str).tolist(),
        covariance_status='unavailable',chi2=np.nan,ndof=0)
    # Keep explicit observed RV support even if tangent geometry alone has rank3.
    if not masks[2].any():return result
    if method=='component_median':
        if not all(mask.any() for mask in masks):return result
        direction=co.icrs_radial_unit(g.ra,g.dec).mean(0);direction/=np.linalg.norm(direction)
        ra=np.degrees(np.arctan2(direction[1],direction[0]))%360;dec=np.degrees(np.arcsin(direction[2]))
        q=np.array([np.median(y[masks[k],k]) for k in range(3)])
        centre=co.icrs_velocity_to_galactic([ra],[dec],[q[0]],[q[1]],[q[2]])[0]
        result.update(uvw=centre,valid=bool(np.isfinite(centre).all()),
            source_ids=g.loc[np.logical_or.reduce(masks),'source_id'].astype(str).tolist())
    if use_uncertainty:
        from .uncertainty import UncertaintyError
        infos=[]
        for k,name in enumerate(('Vra','Vdec','RV')):
            bad=masks[k]&(~np.isfinite(e[:,k])|(e[:,k]<=0))
            if bad.any():
                ids=g.loc[bad,'source_id'].astype(str).tolist()
                raise UncertaintyError(f'invalid_derived_error: component={name},sources={ids}')
            infos.append(compute_rv_evidence(y[masks[k],k],e[masks[k],k]))
        direction=co.icrs_radial_unit(g.ra,g.dec).mean(0);norm=np.linalg.norm(direction)
        if not np.isfinite(norm) or norm<=0:raise UncertaintyError('undefined_entity_mean_sightline')
        direction/=norm
        ra=np.degrees(np.arctan2(direction[1],direction[0]))%360;dec=np.degrees(np.arcsin(direction[2]))
        B=co.icrs_velocity_to_galactic(np.full(3,ra),np.full(3,dec),[1,0,0],[0,1,0],[0,0,1]).T
        sigma=np.array([x.sigma_sys for x in infos])
        if not np.isfinite(sigma).all() or (sigma<=0).any():raise UncertaintyError('invalid_entity_systemic_sigma')
        result.update(covariance=B@np.diag(sigma**2)@B.T,covariance_status='systemic_component_rotation',
            sigma_tan1=sigma[0],sigma_tan2=sigma[1],sigma_RV=sigma[2],
            error_source_ids=';'.join(g.loc[np.logical_or.reduce(masks),'source_id'].astype(str)))
    else:result['covariance_status']='disabled_explicitly'
    return result


def prepare_measured_kinematics(entity_members,labels,method='component_median',use_uncertainty=True):
    estimates=[estimate_measured_entity(entity_members[k],method,use_uncertainty=use_uncertainty) for k in labels]
    table=[]
    for lab,e in zip(labels,estimates):
        row={k:v for k,v in e.items() if k not in ('uvw','covariance','source_uvw','source_ids')}
        row.update(entity=lab,source_ids=';'.join(e['source_ids']))
        for i,key in enumerate(('U','V','W')):row[key]=e['uvw'][i]
        for i in range(3):
            for j in range(3):row[f'cov_{i}{j}']=e['covariance'][i,j]
        table.append(row)
    return dict(uvw=np.array([x['uvw'] for x in estimates]),
        covariance=np.array([x['covariance'] for x in estimates]),
        valid=np.array([x['valid'] for x in estimates]),
        N_RV=np.array([x['N_RV'] for x in estimates]),
        table=pd.DataFrame(table))
