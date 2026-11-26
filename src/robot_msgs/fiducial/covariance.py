"""The one covariance convention every fiducial message uses, and the conversion to ROS's.

**Convention (local tangent, right perturbation).** A 6×6 pose covariance in
`robot_msgs.fiducial` is over ``[δp, δθ]`` (m, rad), both expressed in the pose's *own* (child)
axes::

    p_true = p_est + R_est · δp          ->  δp = R_estᵀ (p_true - p_est)
    R_true = R_est · Exp(δθ)             ->  δθ = rotvec(R_estᵀ R_true)

WHY this one, not ROS's: it is the tangent the ArUco Kalman tracker and the bundle adjustment
already estimate in (2026-11-26, both asked for it), it does not depend on where the parent frame
sits, and a marker's uncertainty is naturally "along its own face / about its own normal".

**ROS's convention** (geometry_msgs/PoseWithCovariance, vision_msgs) is ``[x y z rx ry rz]`` with
the translation and the small rotation about the *parent's fixed axes*. To first order the two are
related by ``J = blockdiag(R, R)``: `local_to_parent_covariance` / `parent_to_local_covariance`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_msgs._core import Array, frozen_array
from robot_msgs.geometry import Pose


def checked_covariance(owner: str, value: Any) -> Array | None:
    """`value` as a frozen (6, 6) symmetric finite covariance, or `None`; else a ValueError."""
    if value is None:
        return None
    covariance = frozen_array(value)
    if covariance.shape != (6, 6):
        raise ValueError(f"{owner}.covariance must be (6, 6), got {covariance.shape}")
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{owner}.covariance must be finite")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-9 * scale):
        raise ValueError(f"{owner}.covariance must be symmetric")
    return covariance


def _jacobian(pose: Pose) -> np.ndarray:
    rotation = pose.orientation.to_matrix()
    jacobian = np.zeros((6, 6))
    jacobian[:3, :3] = rotation
    jacobian[3:, 3:] = rotation
    return jacobian


def local_to_parent_covariance(pose: Pose, covariance: Array) -> Array:
    """A local-tangent covariance (this package's) → ROS's ``[x y z rx ry rz]`` about the parent's
    fixed axes, to first order: ``J Σ Jᵀ`` with ``J = blockdiag(R, R)``."""
    jacobian = _jacobian(pose)
    return frozen_array(jacobian @ np.asarray(covariance, dtype=float) @ jacobian.T)


def parent_to_local_covariance(pose: Pose, covariance: Array) -> Array:
    """The inverse of `local_to_parent_covariance`."""
    jacobian = _jacobian(pose)
    return frozen_array(jacobian.T @ np.asarray(covariance, dtype=float) @ jacobian)
