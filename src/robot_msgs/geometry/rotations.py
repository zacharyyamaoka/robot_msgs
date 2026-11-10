"""Batch rotation math on plain arrays. Every function broadcasts over leading axes.

Conventions, fixed for the whole package:
- quaternions are ``(x, y, z, w)`` — ROS, Pinocchio and scipy order. MuJoCo and viser use
  ``(w, x, y, z)``: convert explicitly with `xyzw_to_wxyz` / `wxyz_to_xyzw`, never by guessing.
- Hamilton product; `quat_multiply(a, b)` applies ``b`` first, then ``a`` (like matrices).
- Euler angles are ROS roll-pitch-yaw: fixed axes X, then Y, then Z, i.e. ``R = Rz·Ry·Rx``.

WHY owned here instead of importing bam_ws's tf_transformations fork: this package is the
bottom of the stack and must install with numpy alone; that fork pulls transforms3d and pytest
at runtime. The tests check every function against scipy as an independent oracle.
"""

from __future__ import annotations

import numpy as np

_NLERP_DOT = 0.9995


def _xyzw(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    return q[..., 0], q[..., 1], q[..., 2], q[..., 3]


def quat_identity(shape: tuple[int, ...] = ()) -> np.ndarray:
    q = np.zeros((*shape, 4))
    q[..., 3] = 1.0
    return q


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("cannot normalise a zero-length quaternion")
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.array(q, dtype=np.float64, copy=True)
    q[..., :3] *= -1.0
    return q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = _xyzw(a)
    bx, by, bz, bw = _xyzw(b)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v (..., 3)`` by unit quaternions ``q (..., 4)``."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, w = q[..., :3], q[..., 3:4]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _xyzw(quat_normalize(q))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def quat_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Shepperd's method, vectorised: pick the numerically largest component per rotation."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) rotation matrices, got {m.shape}")
    flat = m.reshape(-1, 3, 3)
    trace = np.trace(flat, axis1=1, axis2=2)
    decision = np.concatenate([np.diagonal(flat, axis1=1, axis2=2), trace[:, None]], axis=1)
    choice = np.argmax(decision, axis=1)
    q = np.empty((flat.shape[0], 4))
    for i in range(3):
        rows = choice == i
        j, k = (i + 1) % 3, (i + 2) % 3
        r = flat[rows]
        q[rows, i] = 1 - trace[rows] + 2 * r[:, i, i]
        q[rows, j] = r[:, j, i] + r[:, i, j]
        q[rows, k] = r[:, k, i] + r[:, i, k]
        q[rows, 3] = r[:, k, j] - r[:, j, k]
    rows = choice == 3
    r = flat[rows]
    q[rows, 0] = r[:, 2, 1] - r[:, 1, 2]
    q[rows, 1] = r[:, 0, 2] - r[:, 2, 0]
    q[rows, 2] = r[:, 1, 0] - r[:, 0, 1]
    q[rows, 3] = 1 + trace[rows]
    return quat_normalize(q).reshape(*m.shape[:-2], 4)


def quat_from_euler(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """ROS RPY (fixed X, then Y, then Z) to quaternions."""
    hr, hp, hy = (np.asarray(a, dtype=np.float64) / 2.0 for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = np.cos(hr), np.sin(hr), np.cos(hp), np.sin(hp), np.cos(hy), np.sin(hy)
    return np.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        axis=-1,
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternions to ROS RPY, stacked as ``(..., 3)`` = roll, pitch, yaw."""
    x, y, z, w = _xyzw(quat_normalize(q))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def euler_to_matrix(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """ROS RPY straight to ``(..., 3, 3)`` rotation matrices, ``R = Rz(yaw)·Ry(pitch)·Rx(roll)``.

    WHY direct (not through a quaternion): a URDF ``rpy`` and a scene file's angles are printed
    verbatim; the quaternion detour costs the last digits, which shows up in every file written.
    """
    roll, pitch, yaw = (np.asarray(a, dtype=np.float64) for a in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rows = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    return np.stack([np.stack(np.broadcast_arrays(*row), axis=-1) for row in rows], axis=-2)


def matrix_to_euler(matrix: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrices to ROS RPY, stacked ``(..., 3)`` = roll, pitch, yaw.

    Pitch is in [-π/2, π/2]. At gimbal lock (|cos pitch| < 1e-9) only roll ∓ yaw is defined, and
    the convention is tf's: yaw = 0 and the whole turn goes into roll.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) rotation matrices, got {m.shape}")
    r00, r10, r20 = m[..., 0, 0], m[..., 1, 0], m[..., 2, 0]
    r11, r12, r21, r22 = m[..., 1, 1], m[..., 1, 2], m[..., 2, 1], m[..., 2, 2]
    cos_pitch = np.sqrt(r00 * r00 + r10 * r10)
    gimbal = cos_pitch < 1e-9
    roll = np.where(gimbal, np.arctan2(-r12, r11), np.arctan2(r21, r22))
    pitch = np.arctan2(-r20, cos_pitch)
    yaw = np.where(gimbal, 0.0, np.arctan2(r10, r00))
    return np.stack([roll, pitch, yaw], axis=-1)


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vectors (axis × angle, radians) to quaternions."""
    v = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = angle / 2.0
    # sin(θ/2)/θ, with its Taylor series near zero so tiny rotations stay accurate
    small = angle < 1e-8
    scale = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / np.where(small, 1.0, angle))
    return np.concatenate([v * scale, np.cos(half)], axis=-1)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    q = np.where(q[..., 3:4] < 0, -q, q)  # shortest rotation: angle in [0, π]
    sin_half = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, q[..., 3:4])
    small = sin_half < 1e-8
    scale = np.where(small, 2.0 + angle**2 / 12.0, angle / np.where(small, 1.0, sin_half))
    return q[..., :3] * scale


def quat_from_axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("rotation axis must be non-zero")
    return quat_from_rotvec(axis / norm * np.asarray(angle, dtype=np.float64)[..., None])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Spherical interpolation along the shorter arc; ``alpha`` 0 gives q0, 1 gives q1."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    alpha = np.asarray(alpha, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    near = dot > _NLERP_DOT
    safe = np.where(near, 1.0, sin_theta)
    w0 = np.where(near, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / safe)
    w1 = np.where(near, alpha, np.sin(alpha * theta) / safe)
    return quat_normalize(w0 * q0 + w1 * q1)


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Angle in radians of the rotation taking q0 to q1, in [0, π]."""
    dot = np.abs(np.sum(quat_normalize(q0) * quat_normalize(q1), axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def pose_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """``(..., 3)`` positions and ``(..., 4)`` xyzw quaternions to ``(..., 4, 4)`` transforms."""
    position = np.asarray(position, dtype=np.float64)
    rotation = quat_to_matrix(quaternion)
    out = np.zeros((*rotation.shape[:-2], 4, 4))
    out[..., :3, :3] = rotation
    out[..., :3, 3] = position
    out[..., 3, 3] = 1.0
    return out


def matrix_to_pose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape[-2:] != (4, 4):
        raise ValueError(f"expected (..., 4, 4) transforms, got {m.shape}")
    return m[..., :3, 3].copy(), quat_from_matrix(m[..., :3, :3])
