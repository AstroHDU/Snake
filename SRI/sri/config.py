from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path

# All user-adjustable scientific defaults are loaded from this one file.
_DEFAULTS = json.loads((Path(__file__).resolve().parents[1]/'config/defaults.json').read_text(encoding='utf-8'))

@dataclass(frozen=True)
class SRIConfig:
    h_cross_kms: float = _DEFAULTS["h_cross_kms"]
    outlier_prominence: float = _DEFAULTS["outlier_prominence"]
    min_rv_sources: int = _DEFAULTS["min_rv_sources"]
    gold_threshold: float = _DEFAULTS["gold_threshold"]
    silver_threshold: float = _DEFAULTS["silver_threshold"]
    affine_ridge: float = _DEFAULTS["affine_ridge"]
    cross_cap_single_largest: bool = _DEFAULTS["cross_cap_single_largest"]
    cross_aggregation: str = _DEFAULTS["cross_aggregation"]
    spatial_cen_tested_edge_floor: bool = _DEFAULTS["spatial_cen_tested_edge_floor"]
    spatial_cen_width_model: str = _DEFAULTS["spatial_cen_width_model"]
    velocity_center_method: str = _DEFAULTS["velocity_center_method"]
    use_uncertainty: bool = _DEFAULTS["use_uncertainty"]
    spatial_mem_scale: float = _DEFAULTS["spatial_mem_scale"]
    def validate(self) -> None:
        if self.spatial_cen_width_model not in ('projected_mad','smem_scatter'):
            raise ValueError('Unknown spatial width model')
        if self.cross_aggregation not in ('mean_residual','legacy_rms'):
            raise ValueError('Unknown Cross aggregation')
        if not isinstance(self.use_uncertainty,bool):
            raise ValueError('use_uncertainty must be boolean')
        if self.velocity_center_method not in ('component_median','measured_median','joint_observations'):
            raise ValueError('Unknown independent 3D velocity estimator')
        if not (0 < self.spatial_mem_scale < float('inf')):raise ValueError('spatial_mem_scale must be finite and positive')
        if self.h_cross_kms <= 0:
            raise ValueError('h_cross_kms must be positive')
        if self.outlier_prominence <= 1:
            raise ValueError('outlier_prominence must exceed 1')
        if self.min_rv_sources < 1:
            raise ValueError('min_rv_sources must be positive')
        if not (0 < self.silver_threshold < self.gold_threshold < 1):
            raise ValueError('require 0 < silver < gold < 1')
        if self.affine_ridge <= 0:
            raise ValueError('affine_ridge must be positive')
    @classmethod
    def from_json(cls, path: str | Path) -> 'SRIConfig':
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        obj=cls(**data); obj.validate(); return obj

    def to_dict(self):
        return asdict(self)
