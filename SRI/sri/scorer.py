from __future__ import annotations
import hashlib,json
import numpy as np
from .config import SRIConfig
from .core import cen,vcen_local,cross_loss_measurement,combine

try:
    from sri_preprocess.members import member_pair
    from sri_preprocess.geometry import mst_edges_from_distance
except Exception:
    member_pair=None; mst_edges_from_distance=None


def _masks(s,cfg):
    n=len(s.labels); count=np.asarray(s.N_RV)
    if count.shape!=(n,): raise ValueError('state.N_RV must align with labels')
    valid=np.asarray(s.valid,bool)&np.isfinite(np.asarray(s.uvw,float)).all(1)&(count>=cfg.min_rv_sources)
    # Vmem eligibility cannot depend on a different 3D estimator's fit/status.
    vm_valid=getattr(s,'vmem_valid',None)
    vm_count=getattr(s,'vmem_N_RV',None)
    if vm_valid is None:vm_valid=np.asarray(s.complete,bool)
    if vm_count is None:vm_count=count
    complete=np.asarray(s.complete,bool)&np.asarray(vm_valid,bool)&(np.asarray(vm_count)>=cfg.min_rv_sources)
    return valid,complete


def member_evidence(s,complete,scale=2.0):
    if member_pair is None: raise RuntimeError('sri_preprocess package unavailable')
    labels=list(s.labels); n=len(labels)
    if n<2: raise ValueError('at least two scoring entities required')
    DV=np.zeros((n,n)); DX=np.zeros((n,n)); rows=[]
    for i in range(n):
        for j in range(i+1,n):
            aware=bool(complete[i] and complete[j])
            cloud=s.projected if aware else s.free
            dv=member_pair(cloud[labels[i]],cloud[labels[j]])['distance']
            dx=member_pair(s.spatial[labels[i]],s.spatial[labels[j]])['distance']
            DV[i,j]=DV[j,i]=dv; DX[i,j]=DX[j,i]=dx
            rows.append({'i':i,'j':j,'left':labels[i],'right':labels[j],'uses_RV':aware,'V_distance':float(dv),'S_distance':float(dx)})
    vt=mst_edges_from_distance(DV); xt=mst_edges_from_distance(DX)
    vset={tuple(sorted(e)) for e in vt}; xset={tuple(sorted(e)) for e in xt}
    for r in rows:
        e=(r['i'],r['j']); r['on_velocity_MST']=e in vset; r['on_spatial_MST']=e in xset
    vm=float(np.mean([np.exp(-DV[i,j]) for i,j in vt]))
    def _adjacency_response(d):
        d=float(d)
        return float(1/np.hypot(1,d/scale))
    sm_node=float(np.mean([_adjacency_response(DX[i,j]) for i,j in xt]))
    return {'V_mem':vm,'S_mem_node':sm_node,'pairs':rows,'velocity_tree':vt,'spatial_tree':xt}


def score_state(s,cfg:SRIConfig=SRIConfig(),dataset=''):
    if vars(s).get('spatial_response_model')!='adjacency_root_distance':
        raise ValueError('Prepared spatial response is obsolete; rebuild from source observations')
    if vars(s).get('background_support_model')!='disjoint_midrank':
        raise ValueError('Prepared background support is obsolete; rebuild from source observations')
    method=vars(s).get('velocity_center_method')
    if method not in ('component_median','measured_median','joint_observations'):
        raise ValueError('Legacy model-UVW state rejected; rebuild independent 3D inputs from sources')
    if method!=cfg.velocity_center_method:
        raise ValueError('Prepared 3D estimator differs from configuration; rebuild state')
    cfg.validate(); valid,complete=_masks(s,cfg); M=int(valid.sum()); N=len(s.labels)
    if vars(s).get('use_uncertainty') is not cfg.use_uncertainty:
        raise ValueError('Prepared uncertainty mode differs or is missing; rebuild state')
    if cfg.use_uncertainty and vars(s).get('uncertainty_model')!='systemic_components':
        raise ValueError('Prepared covariance is not the required systemic error model; rebuild from sources')
    if vars(s).get('spatial_mem_scale') != cfg.spatial_mem_scale:raise ValueError('Prepared spatial scale differs; rebuild state')
    member=member_evidence(s,complete,cfg.spatial_mem_scale)
    # A current prepared state must contain the selected log-response result.
    # Native/root responses are audit controls, never silent scoring fallbacks.
    if 'S_mem_adjacency' not in s.result:
        raise ValueError('Current Smem adjacency response missing; rebuild state from sources')
    try:
        sm=float(s.result['S_mem_adjacency'])
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid current Smem adjacency response') from exc
    if not np.isfinite(sm) or not 0 <= sm <= 1:
        raise ValueError('Invalid current Smem adjacency response')
    sc=cen(np.asarray(s.xyz,float),[s.spatial[k] for k in s.labels],cfg.spatial_cen_tested_edge_floor,width_model=cfg.spatial_cen_width_model)
    cov_all=getattr(s,'cov_analytic',None)
    if cov_all is None:
        cov_all=np.asarray(getattr(s,'cov',np.full((N,3,3),np.nan)),float)
    else:
        cov_all=np.asarray(cov_all,float)
    cov_valid=cov_all[valid] if cov_all.shape==(N,3,3) else np.full((M,3,3),np.nan)
    if cfg.use_uncertainty:
        from .uncertainty import validate_covariances
        validate_covariances(cov_valid,np.asarray(s.labels)[valid])
    else:
        # Explicit nominal calculation, never a missing-input fallback.
        cov_valid=np.zeros((M,3,3))
    vc=vcen_local(np.asarray(s.uvw,float)[valid],cov_valid)
    cr=cross_loss_measurement(
        np.asarray(s.xyz,float)[valid],np.asarray(s.uvw,float)[valid],cov_valid,
        h=cfg.h_cross_kms,ridge=cfg.affine_ridge,
        cap_single_largest=cfg.cross_cap_single_largest,
        aggregation=cfg.cross_aggregation,
    )
    identifiable=np.asarray(cr.get('prediction_identifiable',[]),bool)
    out=combine(sm,sc['value'] if M>=3 else np.nan,member['V_mem'],vc['value'] if M>=3 else np.nan,cr['D_cross_kms'],M,cfg.h_cross_kms)
    grade='Gold' if out['SRI']>=cfg.gold_threshold else 'Silver' if out['SRI']>=cfg.silver_threshold else 'Bronze'
    row={
        'dataset':dataset,'rng_id':int(getattr(s,'rng_id',getattr(s,'snake',-1))),'Snake':int(getattr(s,'snake',-1)),
        'Nentity':N,'M_RV':M,'S_cen_active':M>=3,'spatial_cen_width_model':cfg.spatial_cen_width_model,'group':out['group'],'S_mem':sm,'S_cen':sc['value'] if M>=3 else np.nan,
        'velocity_center_method':getattr(s,'velocity_center_method','legacy'),
        'use_uncertainty':cfg.use_uncertainty,
        'cross_aggregation':cfg.cross_aggregation,
        'cross_cap_applied':cfg.cross_aggregation=='legacy_rms' and cfg.cross_cap_single_largest,
        'cross_identifiable_folds':int(identifiable.sum()),
        'cross_prediction_folds':int(len(identifiable)),
        'cross_geometry_status':('not_active' if M<4 else 'identified_predictions' if identifiable.all() else 'regularization_dependent_predictions'),
        'rv_gate_status':getattr(s,'prepared',{}).get('rv_gate_status','legacy'),
        'spatial_response_model':'adjacency_root',
        'background_support_model':'rotation_midrank_support',
        'uncertainty_model':'component_systemic_errors',
        'V_mem':member['V_mem'],'V_cen':vc['value'] if M>=3 else np.nan,'S':out['S'],'V':out['V'],'R0':out['R0'],
        'D_cross_kms':cr['D_cross_kms'],'Cross_quality':out['Cross'],'SRI':out['SRI'],'grade':grade,
        'rv_coverage':M/N,'ineligible_3d':';'.join(np.asarray(s.labels)[~valid]),
        'rv_free_pairs':sum(not r['uses_RV'] for r in member['pairs']),
        'affine_ridge':cfg.affine_ridge,'h_cross_kms':cfg.h_cross_kms,'min_rv_sources':cfg.min_rv_sources,
        'max_prediction_amplification':float(np.nanmax(cr['amplification'])) if M>=4 else np.nan,
        'raw_cross_loo_rms_kms':cr.get('raw_rms_kms',np.nan),
        'cross_all_folds_identifiable':bool(np.all(cr['prediction_identifiable'])) if M>=4 else False,
    }
    return row,{'valid':valid,'complete':complete,'member':member,'spatial_cen':sc,'velocity_cen':vc,'cross':cr}


def state_identity(s):
    h=hashlib.sha256()
    h.update(str(getattr(s,'velocity_center_method','legacy')).encode())
    h.update(str(vars(s).get('use_uncertainty','missing')).encode())
    h.update(str(vars(s).get('uncertainty_model','missing')).encode())
    h.update(str(vars(s).get('spatial_response_model','missing')).encode())
    h.update(str(vars(s).get('spatial_mem_scale','missing')).encode())
    h.update(str(vars(s).get('background_support_model','missing')).encode())
    h.update(json.dumps(list(s.labels),ensure_ascii=False).encode())
    arrays=[s.xyz,s.uvw,s.cov,np.asarray(s.N_RV)]
    ca=getattr(s,'cov_analytic',None)
    if ca is not None: arrays.append(ca)
    for a in arrays: h.update(np.asarray(a).tobytes())
    return h.hexdigest()
