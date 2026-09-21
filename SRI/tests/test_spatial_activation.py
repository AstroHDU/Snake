import numpy as np
from sri.core import combine
from sri import SRIConfig

def test_rv_count_controls_cen_activation():
    r=combine(.4,np.nan,.6,np.nan,np.nan,2,5)
    assert r['S']==.4 and r['V']==.6 and np.isnan(r['Cross'])

def test_shared_width_is_default():
    assert SRIConfig().spatial_cen_width_model=='smem_scatter'
