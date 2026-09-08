"""
Kimodo SO(3) Kinematic Blending & Deadband Filter Package
Production-grade open-source contribution package for localai-org/kimodo.cpp and nv-tlabs/kimodo.
"""

from .blend import (
    fast_attack_trapezoid_envelope,
    fast_attack_trapezoid_scalar,
    quat_from_euler_xyz,
    quat_to_euler_xyz,
    quat_geodesic_angle,
    quat_slerp,
    quat_slerp_batch,
    AngularDeadbandFilter,
    BilateralClappingConstraint,
    blend_skeleton_trajectories,
)

__all__ = [
    "fast_attack_trapezoid_envelope",
    "fast_attack_trapezoid_scalar",
    "quat_from_euler_xyz",
    "quat_to_euler_xyz",
    "quat_geodesic_angle",
    "quat_slerp",
    "quat_slerp_batch",
    "AngularDeadbandFilter",
    "BilateralClappingConstraint",
    "blend_skeleton_trajectories",
]

__version__ = "1.0.0"
