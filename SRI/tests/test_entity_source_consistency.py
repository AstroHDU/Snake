from pathlib import Path
import numpy as np
from sri.entity_kinematics import estimate_entity_sources,prepare_entity_kinematics
from sri.raw_prepare import load_catalogue,build_state
from sri_preprocess.dynamic_rv import compute_rv_evidence

ROOT=Path(__file__).resolve().parents[1]

def sources():
    nodes,stars,_=load_catalogue(ROOT/'examples/raw_basic')
    return nodes,stars.loc[stars.id_node.notna()].copy()

def test_estimator_ignores_node_labels_and_source_order():
    _,g=sources();a=estimate_entity_sources(g,'entity')
    z=g.iloc[::-1].copy();z['id_node']=['split_'+str(i%17) for i in range(len(z))]
    b=estimate_entity_sources(z,'other_entity_name')
    np.testing.assert_allclose(a['uvw'],b['uvw'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(a['covariance'],b['covariance'],atol=1e-12,rtol=0)

def test_pooled_entity_RV_information_formula_unchanged():
    _,g=sources();a=estimate_entity_sources(g,'entity')
    ev=compute_rv_evidence(g.radial_velocity,g.radial_velocity_error)
    assert a['N_RV']==ev.n_valid
    assert a['row']['RV']==ev.center_median
    np.testing.assert_allclose(a['local_covariance'][2,2],
        1/ev.information+ev.intrinsic_scatter**2/ev.n_valid,rtol=1e-13)

def test_center_and_covariance_share_exact_estimator():
    _,g=sources();a=estimate_entity_sources(g,'entity');J=a['center_jacobian']
    np.testing.assert_allclose(a['uvw'],np.median(a['transported_members'],axis=0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(a['covariance'],J@a['local_covariance']@J.T,rtol=0,atol=1e-12)
    # Numerical response of the actual median estimator, not a weighted mean.
    eps=1e-5
    D=np.column_stack([(np.median(a['transported_members']+eps*a['member_jacobian'][:,:,j],axis=0)-np.median(a['transported_members']-eps*a['member_jacobian'][:,:,j],axis=0))/(2*eps) for j in range(3)])
    np.testing.assert_allclose(D@a['local_covariance']@D.T,a['covariance'],rtol=1e-8,atol=1e-9)

def test_fixed_entity_estimates_ignore_node_summary_errors():
    nodes,g=sources();parts=[[x] for x in nodes.id_node]
    a=build_state(nodes,g,compute_covariance=False,components_override=parts)
    changed=nodes.copy();changed['e_RV']*=100;changed['e_Vra']*=50;changed['e_Vdec']*=50;changed['RV']+=100
    b=build_state(changed,g,compute_covariance=False,components_override=parts)
    np.testing.assert_allclose(a.uvw,b.uvw,rtol=0,atol=1e-12)
    np.testing.assert_allclose(a.cov_analytic,b.cov_analytic,rtol=0,atol=1e-12)
    for label in a.labels:np.testing.assert_allclose(a.projected[label],b.projected[label],rtol=0,atol=1e-12)

def test_missing_RV_is_not_filled_and_MC_checks_same_model():
    _,g=sources();no=g.copy();no['radial_velocity']=np.nan;no['radial_velocity_error']=np.nan
    a=estimate_entity_sources(no,'entity');assert not a['valid'] and a['N_RV']==0
    assert np.isnan(a['uvw']).all() and no.radial_velocity.isna().all()
    # Aligned sightlines give a genuinely linear median response, for which
    # the finite Gaussian audit must reproduce analytic propagation.
    aligned=g.copy();aligned['ra']=float(g.ra.iloc[0]);aligned['dec']=float(g.dec.iloc[0])
    k=prepare_entity_kinematics({'entity':aligned},['entity'],compute_covariance=True,draws=12000)
    np.testing.assert_allclose(np.diag(k['cov'][0]),np.diag(k['cov_analytic'][0]),rtol=.035)
