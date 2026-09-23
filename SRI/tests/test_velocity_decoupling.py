from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from sri.measured_kinematics import estimate_measured_entity
from sri.raw_prepare import load_catalogue,build_state
from sri.scorer import score_state,_masks
from sri.config import SRIConfig
from sri_preprocess import coordinates as co

ROOT=Path(__file__).resolve().parents[1]

def fixture():
    nodes,g,_=load_catalogue(ROOT/'examples/raw_basic')
    return nodes,g[g.id_node.notna()].copy()


def test_systemic_formula_is_scoring_error_not_MC():
    from sri_preprocess.dynamic_rv import compute_rv_evidence
    _,g=fixture();a=estimate_measured_entity(g,draws=16)
    b=estimate_measured_entity(g,draws=4096)
    np.testing.assert_array_equal(a['covariance'],b['covariance'])
    for value,error,name in [('Vra','e_Vra_source','sigma_tan1'),('Vdec','e_Vdec_source','sigma_tan2'),('radial_velocity','radial_velocity_error','sigma_RV')]:
        expected=compute_rv_evidence(g[value],g[error]).sigma_sys
        np.testing.assert_allclose(a[name],expected,rtol=0,atol=1e-12)
    assert a['covariance_status']=='systemic_component_rotation'

def test_measurement_only_cache_cannot_be_used():
    n,g=fixture();s=build_state(n,g,compute_covariance=False)
    del s.__dict__['uncertainty_model']
    with pytest.raises(ValueError,match='required systemic error model'):score_state(s)


def test_no_RV_has_no_three_dimensional_representative():
    _,g=fixture();g['radial_velocity']=np.nan
    for method in ['component_median']:
        r=estimate_measured_entity(g,method)
        assert not r['valid'] and np.isnan(r['uvw']).all()


def test_vmem_model_cannot_leak_into_3D_inputs():
    from sri.entity_kinematics import prepare_entity_kinematics as original
    nodes,g=fixture();a=build_state(nodes,g,compute_covariance=False)
    def poisoned(*args,**kwargs):
        d=original(*args,**kwargs)
        d['uvw'][:]=123456.;d['cov_analytic'][:]=999999.
        for j,cloud in enumerate(d['projected'].values()):cloud[:]+=(j+1)*1e4
        d['t1']=np.array([9.,8,7]);d['t2']=np.array([6.,5,4])
        return d
    with patch('sri.entity_kinematics.prepare_entity_kinematics',side_effect=poisoned):
        b=build_state(nodes,g,compute_covariance=False)
    np.testing.assert_array_equal(a.uvw,b.uvw)
    np.testing.assert_array_equal(a.cov_analytic,b.cov_analytic)
    ra,_=score_state(a);rb,_=score_state(b)
    assert abs(ra['V_mem']-rb['V_mem'])>.01
    assert ra['V_cen']==rb['V_cen'] and ra['Cross_quality']==rb['Cross_quality']

def test_Vmem_RV_gate_does_not_follow_3D_fit_failure():
    nodes,g=fixture();a=build_state(nodes,g,compute_covariance=False)
    _,before=_masks(a,SRIConfig());a.valid[:]=False;a.uvw[:]=np.nan
    _,after=_masks(a,SRIConfig());np.testing.assert_array_equal(before,after)

def test_reject_legacy_or_mislabelled_prepared_3D():
    nodes,g=fixture();a=build_state(nodes,g,compute_covariance=False)
    with pytest.raises(ValueError,match='differs from configuration'):
        score_state(a,SRIConfig(velocity_center_method='joint_observations'))
    a.__dict__.pop('velocity_center_method')
    with pytest.raises(ValueError,match='Legacy model-UVW'):
        score_state(a)


def test_default_entity_estimator_is_component_median():
    _,g=fixture()
    a=estimate_measured_entity(g)
    b=estimate_measured_entity(g,'component_median')
    np.testing.assert_array_equal(a['uvw'],b['uvw'])
