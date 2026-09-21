from copy import deepcopy
import numpy as np
import pytest
from test_velocity_decoupling import fixture
from sri.raw_prepare import build_state
from sri.scorer import score_state


def test_missing_log_cannot_silently_use_native_or_member_fallback():
    n,g=fixture();state=build_state(n,g,compute_covariance=False)
    assert 'S_mem_native' in state.result
    del state.result['S_mem_adjacency']
    with pytest.raises(ValueError,match='Smem adjacency response missing'):
        score_state(state)


@pytest.mark.parametrize('bad',[None,np.nan,np.inf,-.01,1.01])
def test_invalid_current_log_response_is_rejected(bad):
    n,g=fixture();state=build_state(n,g,compute_covariance=False)
    state.result['S_mem_adjacency']=bad
    with pytest.raises(ValueError,match='Invalid current Smem'):
        score_state(state)


def test_native_control_cannot_change_current_score():
    n,g=fixture();state=build_state(n,g,compute_covariance=False)
    before,_=score_state(state);other=deepcopy(state);other.result['S_mem_native']=0.
    after,_=score_state(other)
    assert before['S_mem']==after['S_mem'] and before['SRI']==after['SRI']
