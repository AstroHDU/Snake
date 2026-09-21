import numpy as np
from sri.core import _tangent_radial_parts,_sqrt_resolved_measurement_loss

def test_zero_covariance_preserves_full_residual_norm():
    for n in [np.array([1.,1.,1.]),np.array([1.,0.,0.]),np.array([.2,-.8,.5])]:
        z=np.array([1.,2.,3.]);p=_tangent_radial_parts(z,np.zeros((3,3)),n,5.)
        np.testing.assert_allclose(p['tan2']+p['rad2'],z@z,atol=1e-12)
        loss,_=_sqrt_resolved_measurement_loss(z,np.zeros((3,3)),n,5,5.)
        np.testing.assert_allclose(loss,z@z,atol=1e-12)

def test_radial_only_covariance_does_not_make_fake_tangent_residual():
    n=np.array([1.,2.,3.]);n/=np.linalg.norm(n);z=7*n
    p=_tangent_radial_parts(z,3*np.outer(n,n),n,5.)
    np.testing.assert_allclose(p['tan2'],0,atol=1e-12)
    np.testing.assert_allclose(p['rad2'],49,atol=1e-12)

def test_component_candidate_uses_all_tangent_data_same_systemic_errors():
    from test_velocity_decoupling import fixture
    from sri.measured_kinematics import estimate_measured_entity
    _,g=fixture();a=estimate_measured_entity(g,method='component_median');b=estimate_measured_entity(g,method='measured_median')
    np.testing.assert_allclose(a['covariance'],b['covariance'])
    assert a['N_tan1']>=a['N_RV']
