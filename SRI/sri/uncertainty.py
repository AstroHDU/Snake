"""Strict input/centre-error validation. No scatter or zero-error fallback."""
import numpy as np
import pandas as pd

class UncertaintyError(ValueError):
    pass

def validate_sources(stars):
    issues=[]
    rv=pd.to_numeric(stars.radial_velocity,errors='coerce').to_numpy(float)
    for name in ('parallax_error','pmra_error','pmdec_error','radial_velocity_error'):
        needed=np.isfinite(rv) if name=='radial_velocity_error' else np.ones(len(stars),bool)
        v=pd.to_numeric(stars[name],errors='coerce').to_numpy(float) if name in stars else np.full(len(stars),np.nan)
        bad=needed & (~np.isfinite(v) | (v<=0))
        for k in np.flatnonzero(bad):
            row=stars.iloc[k]
            issues.append(f"node={row.get('id_node')},source={row.get('source_id')},field={name},value={v[k]}")
    # Correlations are optional under the existing independent-error convention;
    # an explicitly supplied invalid correlation is never clipped/replaced.
    for name in ('parallax_pmra_corr','parallax_pmdec_corr'):
        if name not in stars:continue
        v=pd.to_numeric(stars[name],errors='coerce').to_numpy(float)
        for k in np.flatnonzero(~np.isfinite(v)|(np.abs(v)>1)):
            row=stars.iloc[k];issues.append(f"node={row.get('id_node')},source={row.get('source_id')},field={name},value={v[k]}")
    if issues:raise UncertaintyError('invalid_required_source_errors: '+'; '.join(issues))

def validate_covariances(cov, labels):
    c=np.asarray(cov,float)
    if c.shape!=(len(labels),3,3):raise UncertaintyError('invalid_covariance_shape')
    for label,m in zip(labels,c):
        if not np.isfinite(m).all():raise UncertaintyError(f'entity={label}: nonfinite_covariance')
        if not np.allclose(m,m.T,rtol=1e-8,atol=1e-12):raise UncertaintyError(f'entity={label}: asymmetric_covariance')
        if np.any(np.diag(m)<=0) or np.linalg.eigvalsh(m).min() < -1e-10:
            raise UncertaintyError(f'entity={label}: invalid_covariance_variance')
