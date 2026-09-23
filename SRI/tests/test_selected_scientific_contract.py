import json
from pathlib import Path
import numpy as np
import pytest
from sri import SRIConfig,score_state
from sri.core import cen,vcen_local,combine
from sri.kpd import partition
from sri.raw_prepare import build_state,build_core_state
from test_velocity_decoupling import fixture

def test_single_configuration_source():
    values=json.loads((Path(__file__).resolve().parents[1]/'config/defaults.json').read_text())
    assert SRIConfig().to_dict()==values
    assert values['outlier_prominence']==3

def test_kpd_strict_three_boundary():
    x=np.array([0.,1.,4.]);d=abs(x[:,None]-x[None,:])
    assert len(partition(d)[0])==1
    assert len(partition(d,prominence=2)[0])==2

def test_cen_nominal_same_q75_response():
    x=np.array([[0.,0,0],[1.,0,0],[5.,0,0],[7.,0,0]])
    # Physical MST lengths4,2,1; Q75 remaining=1.75.
    expected=np.sqrt(1.75/4)
    assert cen(x)['value']==pytest.approx(expected)
    assert vcen_local(x,np.zeros((4,3,3)))['value']==pytest.approx(expected)

def test_velocity_finite_error_does_not_erase_large_gap():
    x=np.array([[0.,0,0],[1.,0,0],[11.,0,0]])
    value=vcen_local(x,np.tile(np.eye(3)*1e12,(3,1,1)))['value']
    assert value<=1/np.sqrt(2)+1e-12
    assert value>vcen_local(x,np.zeros((3,3,3)))['value']

def test_geometric_channels_product_cross():
    row=combine(.64,.81,.81,.64,5,4,5)
    assert row['S']==pytest.approx(.72)
    assert row['V']==pytest.approx(.72)
    assert row['SRI']==pytest.approx(.72/np.sqrt(2))

def test_custom_spatial_scale_survives_core_rebuild():
    n,g=fixture();cfg=SRIConfig(spatial_mem_scale=3.)
    s=build_state(n,g,compute_covariance=False,spatial_mem_scale=3.)
    c=build_core_state(s,n,g,s.labels[:2],compute_covariance=False)
    assert c.spatial_mem_scale==3.
    score_state(c,cfg)
    with pytest.raises(ValueError,match='spatial scale'):score_state(c,SRIConfig())


@pytest.mark.parametrize('kwargs',[{'cross_aggregation':'legacy_rms'},{'cross_cap_single_largest':True},{'spatial_cen_width_model':'projected_mad'},{'velocity_center_method':'joint_observations'},{'h_cross_kms':float('nan')},{'outlier_prominence':float('inf')}])
def test_unpublished_methods_and_invalid_scales_are_rejected(kwargs):
    with pytest.raises(ValueError):SRIConfig(**kwargs).validate()
