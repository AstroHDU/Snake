from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
import pandas as pd
from sri_preprocess import coordinates as co, entities as en, dynamic_rv as rv, members as mem, background as bg
from .observables import derive_member_observables, derive_node_table
from sri_preprocess.uncertainty_mode import scoped

SEED=20260609
RV_DRAWS=4096
BLOCK=64

@dataclass
class PreparedState:
    snake:int; rng_id:int; labels:list; complete:np.ndarray; valid:np.ndarray
    xyz:np.ndarray; uvw:np.ndarray; cov:np.ndarray; free:dict; projected:dict
    spatial:dict; t1:np.ndarray; t2:np.ndarray; result:dict=field(default_factory=dict)
    N_RV:np.ndarray=field(default_factory=lambda:np.empty(0,int))
    components:list=field(default_factory=list); prepared:dict=field(default_factory=dict)
    cov_analytic:np.ndarray|None=None
    entity_kinematics:pd.DataFrame=field(default_factory=pd.DataFrame)
    vmem_valid:np.ndarray|None=None
    vmem_N_RV:np.ndarray|None=None
    measured_kinematics:pd.DataFrame=field(default_factory=pd.DataFrame)
    velocity_center_method:str='component_median'
    use_uncertainty:bool=True
    uncertainty_model:str='systemic_components'
    spatial_response_model:str='adjacency_root_distance'
    background_support_model:str='disjoint_midrank'
    spatial_mem_scale:float=2.0


def _read(path):
    return pd.read_csv(path,dtype={'source_id':str,'id_node':str,'id_part':str},float_precision='round_trip')


def load_catalogue(input_dir: str|Path, node_metadata: str|Path|None=None):
    """Load basic member observables and derive every SRI working coordinate.

    Public raw input no longer requires pre-computed X/Y/Z, Vra/Vdec, or a
    node-level kinematic table.  If those columns happen to be present they are
    ignored for scoring and may be compared through the audit helper.

    Required file:
      * Snake_Member_Catalogue.csv

    Optional files:
      * Snake_Complex_Catalogue.csv (otherwise synthesized from Snake IDs)
      * Snake_ID_Mapping.csv (old_Snake is used only for reproducible rng_id)
      * Snake_Node_Catalogue.csv / --node-metadata are optional audit inputs
        but their derived kinematic columns are not trusted by this raw path.
    """
    p=Path(input_dir)
    mem_path=p/'Snake_Member_Catalogue.csv'
    if not mem_path.exists():
        raise FileNotFoundError(mem_path)
    raw=_read(mem_path)
    if not raw.source_id.is_unique:
        raise ValueError('source_id must be unique')
    stars=derive_member_observables(raw)
    nodes=derive_node_table(stars)

    # Optional original->current ID mapping affects deterministic RNG identity
    # only; it is not a physical input.
    map_path=p/'Snake_ID_Mapping.csv'
    if map_path.exists():
        mapping=pd.read_csv(map_path,float_precision='round_trip')
        if {'Snake','old_Snake'} <= set(mapping.columns):
            mp=dict(zip(pd.to_numeric(mapping.Snake,errors='coerce'),pd.to_numeric(mapping.old_Snake,errors='coerce')))
            nodes['rng_id']=nodes.Snake.map(mp).fillna(nodes.Snake).astype(int)
            stars['rng_id']=stars.Snake.map(mp).fillna(stars.Snake).astype(int)
        else:
            stars['rng_id']=stars.Snake.astype(int)
    else:
        stars['rng_id']=stars.Snake.astype(int)
    nodes=nodes.sort_values(['rng_id','id_node'],kind='stable').reset_index(drop=True)
    nodes['rv_rng_index']=np.arange(len(nodes))
    stars=stars.sort_values(['rng_id','source_id'],kind='stable').reset_index(drop=True)

    comp_path=p/'Snake_Complex_Catalogue.csv'
    if comp_path.exists():
        complexes=pd.read_csv(comp_path,float_precision='round_trip')
        if 'Snake' not in complexes:
            raise ValueError('complex table missing Snake')
        complexes['Snake']=pd.to_numeric(complexes.Snake,errors='raise').astype(int)
        missing=set(nodes.Snake)-set(complexes.Snake)
        if missing:
            raise ValueError(f'complex table missing Snake IDs: {sorted(missing)[:10]}')
    else:
        complexes=pd.DataFrame({'Snake':sorted(nodes.Snake.unique().astype(int))})

    # Legacy node inputs are accepted only as optional audit/reference material.
    # They never override quantities reconstructed from the member observations.
    legacy_node=p/'Snake_Node_Catalogue.csv'
    if node_metadata is not None or legacy_node.exists():
        aux_path=Path(node_metadata) if node_metadata is not None else legacy_node
        try:
            aux=_read(aux_path)
            if {'Snake','id_node'} <= set(aux.columns):
                expected=set(zip(nodes.Snake.astype(int),nodes.id_node.astype(str)))
                got=set(zip(pd.to_numeric(aux.Snake,errors='coerce').astype('Int64'),aux.id_node.astype(str)))
                if not expected <= got:
                    raise ValueError('node audit table does not cover all derived nodes')
        except Exception as exc:
            raise ValueError(f'node audit failed for {aux_path}: {exc}') from exc
    return nodes,stars,complexes




def _spatial_response(distance, mode, scale=2.0):
    d=float(distance)
    if mode=='native':
        return float(1/np.hypot(1,d))
    if mode=='adjacency':
        return float(1/np.hypot(1,d/scale))
    raise ValueError(mode)


def _companion_spatial(spatial_clouds, spatial_edges, members, components, labels, rng_id, scale):
    """Bridge/background correction for native and adjacency spatial responses.

    The same source assignment and the same line-of-sight-preserving null
    rotations are used for both responses.  SRI scoring uses
    the adjacency response; native values are retained only as an audit comparison.
    """
    def edge_distance(clouds,i,j):
        return mem.member_pair(clouds[labels[i]],clouds[labels[j]])['distance']
    def score(clouds,mode):
        return float(np.mean([_spatial_response(edge_distance(clouds,i,j),mode,scale) for i,j in spatial_edges]))

    base={mode:score(spatial_clouds,mode) for mode in ('native','adjacency')}
    clouds={name:pd.DataFrame(x,columns=['X','Y','Z']) for name,x in spatial_clouds.items()}
    node_rows=members.loc[members.id_node.notna()]
    parts=en.entity_parts(components,node_rows)
    back=members.loc[members.id_node.isna()].copy()
    if back.empty:
        return {mode:dict(value=base[mode],base=base[mode],gain=0.,q=np.nan) for mode in base},0

    def assigned_score(assigned,mode):
        return float(np.mean([_spatial_response(mem.member_pair(assigned[labels[i]],assigned[labels[j]])['distance'],mode,scale) for i,j in spatial_edges]))

    assigned=bg.assign_background_xyz(clouds,parts,back)
    obs={mode:assigned_score(assigned,mode) for mode in base}
    gain={mode:obs[mode]-base[mode] for mode in base}
    centers={str(part):np.median(bg.xyz_values(g),axis=0) for part,g in members.groupby('id_part')}
    rng=np.random.default_rng(SEED+rng_id); gains={mode:[] for mode in base}
    for _ in range(bg.N_BACKGROUND_ROTATIONS):
        rot=bg.rotate_background_by_part(back,centers,rng,mode='los')
        ar=bg.assign_background_xyz(clouds,parts,rot)
        for mode in base:
            gains[mode].append(assigned_score(ar,mode)-base[mode])
    out={}
    for mode in base:
        q=bg.midrank_percentile(gain[mode],np.asarray(gains[mode]))
        value=bg.support_only_corrected_continuity(base[mode],gain[mode],2*q-1)
        out[mode]=dict(value=float(value),base=float(base[mode]),gain=float(gain[mode]),q=float(q))
    return out,len(back)


def _prepare_fixed_components(snake,nodes,frame,components):
    """Build a prepared structure from an explicit, fixed entity partition.

    This is used only for retained-core rescoring after the review diagnostic.
    It preserves exactly the entities selected from the full-system review; the
    contraction step is not rerun, so a 3-entity core remains a 3-entity core.
    """
    components=[list(map(str,c)) for c in components]
    flat=[n for c in components for n in c]
    node_ids=sorted(nodes['id_node'].astype(str).tolist())
    if len(flat)!=len(set(flat)) or set(flat)!=set(node_ids):
        raise ValueError('fixed components must cover each retained base node exactly once')
    node_rows=frame.loc[frame.id_node.notna()].copy()
    node_rows['id_node']=node_rows['id_node'].astype(str)
    node_members={node_id:node_rows.loc[node_rows.id_node.eq(node_id)].copy() for node_id in node_ids}
    if any(g.empty for g in node_members.values()):
        raise ValueError('fixed core is missing member rows for a retained node')
    entity_members=en.aggregate_members(components,node_members)
    return {
        'node_ids':node_ids,
        'node_rows':node_rows,
        'background':frame.loc[frame.id_node.isna()].copy(),
        'components':components,
        'natural_components':components,
        'natural_entity_count':len(components),
        'entity_members':entity_members,
        'entity_to_parts':en.entity_parts(components,node_rows),
        'relation_status':'review_core_fixed',
        'singleton_fallback':False,
        'merge_rule_id':'review-core-fixed-partition',
        'strict_pairs':pd.DataFrame(),
        'strict_trace':[],
    }


@scoped
def build_state(nodes,stars,draws=None,compute_covariance=True,components_override=None,velocity_center_method='component_median',use_uncertainty=True,spatial_mem_scale=None):
    if spatial_mem_scale is None:
        from .config import SRIConfig
        spatial_mem_scale=SRIConfig().spatial_mem_scale
    if not (0 < spatial_mem_scale < float('inf')):raise ValueError('invalid spatial_mem_scale')
    # Source reconstruction may not reuse stale model-transport columns, even
    # when an old metadata tag claims the same coordinate convention.
    stars=stars.drop(columns=[c for c in stars.columns if c.endswith('_transport')],errors='ignore').copy()
    if use_uncertainty:
        from .uncertainty import validate_sources
        validate_sources(stars)
    else:
        rng_id=int(nodes.rng_id.iloc[0])
        nodes=derive_node_table(stars);nodes['rng_id']=rng_id
    snake=int(nodes.Snake.iloc[0]);rng_id=int(nodes.rng_id.iloc[0]);allvalid=co.rv_complete_rows(nodes)
    mode='cartesian' if allvalid else 'valid_rv_centroid'
    frame=co.add_member_score_columns(nodes,stars,mode);co.configure_entity_velocity_basis(mode)
    prep=(en.prepare_structure(snake,nodes,frame) if components_override is None
          else _prepare_fixed_components(snake,nodes,frame,components_override))
    if components_override is None and use_uncertainty:
        from .rv_merge import gated_components
        natural,audit=gated_components(nodes,stars,prep)
        components=natural if len(natural)>1 else [[x] for x in sorted(nodes.id_node)]
        prep=_prepare_fixed_components(snake,nodes,frame,components)
        prep.update(natural_components=natural,natural_entity_count=len(natural),singleton_fallback=len(natural)==1,
            relation_status='RV_conflict_gated',merge_rule_id='local-contact-velocity-body-rv3',rv_gate_status='active',rv_gate_audit=audit)
    else:
        prep['rv_gate_status']='fixed_partition' if components_override is not None else 'disabled_with_uncertainty'
    labels=list(prep['entity_members']);components=prep['components']
    group=en.aggregate_group(nodes,components);assert labels==group.id_node.tolist()
    from .entity_kinematics import prepare_entity_kinematics
    if draws is not None:
        raise ValueError('Node-level draws are incompatible with entity-source estimation')
    kin=prepare_entity_kinematics(prep['entity_members'],labels,
        compute_covariance=False,draws=RV_DRAWS,seed=SEED)
    # This estimator consumes only source observations, never kin's model UVW.
    from .measured_kinematics import prepare_measured_kinematics
    measured=prepare_measured_kinematics(prep['entity_members'],labels,velocity_center_method,use_uncertainty=use_uncertainty)
    if use_uncertainty:
        from .uncertainty import validate_covariances
        valid=measured['valid']
        validate_covariances(measured['covariance'][valid],np.asarray(labels)[valid])
    t1,t2=kin['t1'],kin['t2'];free=kin['free'];projected=kin['projected']
    spatial={label:prep['entity_members'][label][['X','Y','Z']].to_numpy(float) for label in labels}
    from sri_preprocess.geometry import mst_edges_from_distance
    dx=np.zeros((len(labels),len(labels)))
    for i,a in enumerate(labels):
        for j in range(i+1,len(labels)):
            dx[i,j]=dx[j,i]=mem.member_pair(spatial[a],spatial[labels[j]])['distance']
    xt=mst_edges_from_distance(dx)
    sm,nc=_companion_spatial(spatial,xt,stars,components,labels,rng_id,spatial_mem_scale)
    result={
        'S_mem_adjacency':sm['adjacency']['value'],'S_mem_native':sm['native']['value'],
        'S_mem_node_adjacency':sm['adjacency']['base'],'S_mem_node_native':sm['native']['base'],
        'S_mem_node':sm['adjacency']['base'],
        'companion_gain_adjacency':sm['adjacency']['gain'],'companion_q_adjacency':sm['adjacency']['q'],
        'companion_gain_native':sm['native']['gain'],'companion_q_native':sm['native']['q'],
        'companion_gain':sm['adjacency']['gain'],'companion_q':sm['adjacency']['q'],'N_companion':nc,
    }
    return PreparedState(snake,rng_id,labels,kin['complete'],measured['valid'],
        group[['X','Y','Z']].to_numpy(float),measured['uvw'],
        measured['covariance'].copy() if compute_covariance else np.full_like(measured['covariance'],np.nan),
        free,projected,spatial,t1,t2,result,measured['N_RV'],components,prep,
        measured['covariance'],kin['entity_table'],kin['valid'],kin['N_RV'],
        measured['table'],velocity_center_method,use_uncertainty,spatial_mem_scale=spatial_mem_scale)


def build_core_state(full_state,nodes,stars,core_labels,draws=None,compute_covariance=True):
    """Rebuild and rescore only the selected retained core.

    ``core_labels`` must be entity labels from ``full_state``.  The selected
    entities are preserved exactly, while source-level coordinates, tangent
    basis, member continuity, bridge/background support, RV summaries and all
    other score inputs are recomputed from only the retained core.  Smaller
    review-partition components are excluded entirely.
    """
    core_labels=tuple(map(str,core_labels))
    if len(core_labels)<2:
        raise ValueError('retained core requires at least two entities')
    label_to_component={str(label):list(map(str,component)) for label,component in zip(full_state.labels,full_state.components)}
    missing=[label for label in core_labels if label not in label_to_component]
    if missing:
        raise ValueError(f'core labels not present in full state: {missing}')
    components=[label_to_component[label] for label in core_labels]
    node_ids={node for component in components for node in component}
    core_nodes=nodes.loc[nodes.id_node.astype(str).isin(node_ids)].copy()
    if set(core_nodes.id_node.astype(str))!=node_ids:
        raise ValueError('retained core node table is incomplete')
    # Keep only source rows that belong to retained core nodes plus bridge rows
    # whose support scope is compatible with the retained node parts.  In the
    # group-wide bridge mode all rows share one synthetic scope, so all group
    # bridge candidates remain available to the retained core.
    core_stars=en.subset_members(stars,core_nodes)
    return build_state(
        core_nodes,core_stars,draws=draws,compute_covariance=compute_covariance,
        components_override=components,
        velocity_center_method=full_state.velocity_center_method,
        use_uncertainty=full_state.use_uncertainty,
        spatial_mem_scale=full_state.spatial_mem_scale,
    )


def prepare_all(input_dir,node_metadata=None,system_ids=None,limit=None,use_uncertainty=True):
    nodes,stars,complexes=load_catalogue(input_dir,node_metadata)
    ids=sorted(nodes.Snake.unique().astype(int))
    if system_ids is not None:ids=[i for i in ids if i in set(system_ids)]
    if limit is not None:ids=ids[:limit]
    ng={int(k):g for k,g in nodes.groupby('Snake')};mg={int(k):g for k,g in stars.groupby('Snake')}
    out=[]
    for sid in ids:
        nd=ng[sid]; st=mg[sid]; out.append(('',build_state(nd,st,use_uncertainty=use_uncertainty)))
    return out
