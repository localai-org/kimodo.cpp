"""
Kimodo SO(3) Kinematic Blending & Deadband Filtering Library.
Vectorized NumPy SIMD implementations matching C++20 kimodo_blend.hpp.
"""

from typing import Union, Tuple, Optional
import math
import numpy as np


def fast_attack_trapezoid_scalar(
    progress: float,
    attack_frac: float = 0.15,
    release_frac: float = 0.20
) -> float:
    """
    Computes scalar instantaneous motion envelope factor E(p) in [0, 1].
    Uses C1-continuous Hermite cubic smoothstep S(s) = s^2 * (3 - 2s).
    """
    p = max(0.0, min(1.0, float(progress)))
    att = max(0.001, min(0.999, float(attack_frac)))
    rel = max(0.001, min(1.0 - att, float(release_frac)))

    if p < att:
        s = p / att
        return float(s * s * (3.0 - 2.0 * s))
    if p <= (1.0 - rel):
        return 1.0
    s = (1.0 - p) / rel
    return float(s * s * (3.0 - 2.0 * s))


def fast_attack_trapezoid_envelope(
    num_frames: int,
    attack_frac: float = 0.15,
    release_frac: float = 0.20
) -> np.ndarray:
    """
    Generates vectorized C1-smooth trapezoidal envelope array of shape [F].
    - Attack phase [0, attack_frac): smooth cubic ramp from 0.0 to 1.0.
    - Hold phase [attack_frac, 1.0 - release_frac]: flat 1.0 plateau.
    - Release phase (1.0 - release_frac, 1.0]: smooth cubic decay from 1.0 to 0.0.
    """
    if num_frames <= 0:
        return np.empty((0,), dtype=np.float32)
    if num_frames == 1:
        return np.array([1.0], dtype=np.float32)

    p = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    env = np.ones(num_frames, dtype=np.float32)

    att = float(np.clip(attack_frac, 0.001, 0.999))
    rel = float(np.clip(release_frac, 0.001, 1.0 - att))

    # Attack ramp
    mask_att = p < att
    if np.any(mask_att):
        s_att = p[mask_att] / att
        env[mask_att] = s_att * s_att * (3.0 - 2.0 * s_att)

    # Release ramp
    mask_rel = p > (1.0 - rel)
    if np.any(mask_rel):
        s_rel = (1.0 - p[mask_rel]) / rel
        env[mask_rel] = s_rel * s_rel * (3.0 - 2.0 * s_rel)

    return env


def quat_from_euler_xyz(
    rx: Union[float, np.ndarray],
    ry: Union[float, np.ndarray],
    rz: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Constructs unit quaternion(s) from intrinsic Euler XYZ rotation angles in radians.
    R = R_x(rx) * R_y(ry) * R_z(rz)
    Returns array of shape [..., 4] (x, y, z, w).
    """
    rx = np.asarray(rx, dtype=np.float32)
    ry = np.asarray(ry, dtype=np.float32)
    rz = np.asarray(rz, dtype=np.float32)

    hx = rx * 0.5
    hy = ry * 0.5
    hz = rz * 0.5

    cx, sx = np.cos(hx), np.sin(hx)
    cy, sy = np.cos(hy), np.sin(hy)
    cz, sz = np.cos(hz), np.sin(hz)

    qx = sx * cy * cz + cx * sy * sz
    qy = cx * sy * cz - sx * cy * sz
    qz = cx * cy * sz + sx * sy * cz
    qw = cx * cy * cz - sx * sy * sz

    q = np.stack([qx, qy, qz, qw], axis=-1)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(norm > 1e-12, q / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def quat_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """
    Extracts Euler XYZ angles in radians from quaternion array [..., 4].
    R = R_x(rx) * R_y(ry) * R_z(rz)
    Returns array of shape [..., 3] (rx, ry, rz).
    """
    q = np.asarray(q, dtype=np.float32)
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    # R02 = 2*(x*z + w*y) = sin(ry)
    sin_ry = 2.0 * (x * z + w * y)
    sin_ry_clipped = np.clip(sin_ry, -1.0, 1.0)
    ry = np.arcsin(sin_ry_clipped)

    # Check for gimbal lock (|sin_ry| >= 1.0 - 1e-6)
    gimbal_lock = np.abs(sin_ry) >= (1.0 - 1e-6)

    # Standard non-gimbal extraction:
    # -R12 = 2*(w*x - y*z), R22 = 1 - 2*(x*x + y*y)
    rx_standard = np.arctan2(2.0 * (w * x - y * z), 1.0 - 2.0 * (x * x + y * y))
    # -R01 = 2*(w*z - x*y), R00 = 1 - 2*(y*y + z*z)
    rz_standard = np.arctan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y * y + z * z))

    # Gimbal lock fallback (ry = +/- pi/2): set rz = 0, rx = atan2(2*(x*y + w*z), 1 - 2*(x*x + z*z))
    rx_gimbal = np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z))
    rz_gimbal = np.zeros_like(rx_gimbal)

    rx = np.where(gimbal_lock, rx_gimbal, rx_standard)
    rz = np.where(gimbal_lock, rz_gimbal, rz_standard)

    return np.stack([rx, ry, rz], axis=-1)


def quat_geodesic_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Computes exact geodesic angular distance (in radians) on SO(3) between q1 and q2.
    Uses relative quaternion dq = q1^{-1} * q2 and 2*atan2(||v||, |w|) to avoid acos cancellation.
    """
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    xd = w1 * x2 - x1 * w2 - y1 * z2 + z1 * y2
    yd = w1 * y2 + x1 * z2 - y1 * w2 - z1 * x2
    zd = w1 * z2 - x1 * y2 + y1 * x2 - z1 * w2
    wd = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2

    v_norm = np.sqrt(xd * xd + yd * yd + zd * zd)
    return 2.0 * np.arctan2(v_norm, np.abs(wd))


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Scalar Spherical Linear Interpolation (SLERP) on SO(3) along the shortest geodesic path.
    """
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    t = float(t)

    dot = float(np.sum(q1 * q2))
    q2_adj = q2.copy()

    if dot < 0.0:
        dot = -dot
        q2_adj = -q2_adj

    if dot > 0.9995:
        r = q1 + t * (q2_adj - q1)
        norm = float(np.linalg.norm(r))
        return r / max(norm, 1e-12)

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    r = s0 * q1 + s1 * q2_adj
    norm = float(np.linalg.norm(r))
    return r / max(norm, 1e-12)


def quat_slerp_batch(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Vectorized SO(3) SLERP between batch quaternions q0 [F, J, 4] and q1 [F, J, 4] by t [F].
    Supports arbitrary broadcasting shapes [..., 4].
    """
    q0 = np.asarray(q0, dtype=np.float32)
    q1 = np.asarray(q1, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)

    # Compute dot product along quaternion dimension
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1_adj = np.where(dot < 0.0, -q1, q1)
    dot_adj = np.abs(dot)

    # Prepare interpolation factor broadcasting
    while t.ndim < q0.ndim:
        t = t[..., None]

    theta_0 = np.arccos(np.clip(dot_adj, -1.0, 1.0))
    theta = theta_0 * t

    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    # Use NLERP for near-parallel quaternions to avoid division by zero
    safe_mask = sin_theta_0 > 1e-4
    s0 = np.where(safe_mask, np.cos(theta) - dot_adj * sin_theta / np.maximum(sin_theta_0, 1e-8), 1.0 - t)
    s1 = np.where(safe_mask, sin_theta / np.maximum(sin_theta_0, 1e-8), t)

    out = s0 * q0 + s1 * q1_adj
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


class AngularDeadbandFilter:
    """
    Geodesic Angular Deadband Filter for joint orientation streams.
    Eliminates 60Hz floating-point micro-jitter below epsilon (default 0.0005 rad = 0.0286 deg)
    and applies continuous-time exponential damping alpha = 1 - exp(-lambda * delta_sec) above epsilon.
    """
    def __init__(self, epsilon: float = 0.0005, lambda_damping: float = 18.0):
        self.epsilon = float(epsilon)
        self.lambda_damping = float(lambda_damping)

    def is_in_deadband(self, current: np.ndarray, target: np.ndarray) -> bool:
        """Checks if angular delta is within the deadband tolerance."""
        delta_rad = float(quat_geodesic_angle(current, target))
        return delta_rad < self.epsilon

    def filter(self, current: np.ndarray, target: np.ndarray, delta_sec: float) -> np.ndarray:
        """
        Filters a single joint quaternion candidate.
        """
        current = np.asarray(current, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        delta_sec = max(0.0, min(float(delta_sec), 0.033))

        delta_rad = float(quat_geodesic_angle(current, target))
        if delta_rad < self.epsilon:
            # Jitter suppression: drop update and lock to current state
            return current.copy()

        alpha = 1.0 - math.exp(-self.lambda_damping * delta_sec)
        return quat_slerp(current, target, alpha)

    def filter_trajectory(
        self,
        trajectory: np.ndarray,
        delta_sec: float = 1.0 / 60.0
    ) -> np.ndarray:
        """
        Sequentially filters an entire joint rotation trajectory [F, J, 4].
        """
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.shape[0] <= 1:
            return traj.copy()

        out = np.zeros_like(traj)
        out[0] = traj[0]

        num_frames = traj.shape[0]
        num_joints = traj.shape[1] if traj.ndim >= 3 else 1

        for f in range(1, num_frames):
            if traj.ndim >= 3:
                for j in range(num_joints):
                    out[f, j] = self.filter(out[f - 1, j], traj[f, j], delta_sec)
            else:
                out[f] = self.filter(out[f - 1], traj[f], delta_sec)

        return out


class BilateralClappingConstraint:
    """
    Bilateral Clapping & Contact Clearance Constraint Validator.
    """
    TARGET_WRIST_SEPARATION_M = 0.170  # 17.0 cm clearance
    HALF_WRIST_SEPARATION_M = 0.085
    MIN_PALM_OPPOSITION_DOT = -0.90
    MIN_THUMB_UPWARD_DOT = 0.90

    @classmethod
    def rotate_vector(cls, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotates 3D vector v by unit quaternion q."""
        qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])

        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)

        return np.array([
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx)
        ], dtype=np.float32)

    @classmethod
    def validate(
        cls,
        left_wrist_pos: np.ndarray,
        right_wrist_pos: np.ndarray,
        left_wrist_rot: np.ndarray,
        right_wrist_rot: np.ndarray,
        separation_tol_m: float = 0.03
    ) -> dict:
        """
        Validates bilateral clapping kinematics.
        """
        lw_pos = np.asarray(left_wrist_pos, dtype=np.float32)
        rw_pos = np.asarray(right_wrist_pos, dtype=np.float32)

        # 1. Separation distance
        actual_sep = float(np.linalg.norm(rw_pos - lw_pos))
        sep_ok = abs(actual_sep - cls.TARGET_WRIST_SEPARATION_M) <= separation_tol_m

        # 2. Palm normal alignment (left palm +X, right palm -X)
        left_palm_w = cls.rotate_vector(left_wrist_rot, np.array([1.0, 0.0, 0.0]))
        right_palm_w = cls.rotate_vector(right_wrist_rot, np.array([-1.0, 0.0, 0.0]))
        palm_dot = float(np.dot(left_palm_w, right_palm_w))
        palm_ok = palm_dot <= cls.MIN_PALM_OPPOSITION_DOT

        # 3. Upward thumb alignment (local +Y)
        left_thumb_w = cls.rotate_vector(left_wrist_rot, np.array([0.0, 1.0, 0.0]))
        right_thumb_w = cls.rotate_vector(right_wrist_rot, np.array([0.0, 1.0, 0.0]))
        left_thumb_dot = float(left_thumb_w[1])
        right_thumb_dot = float(right_thumb_w[1])
        thumb_ok = (left_thumb_dot >= cls.MIN_THUMB_UPWARD_DOT) and (right_thumb_dot >= cls.MIN_THUMB_UPWARD_DOT)

        # 4. No coronal X-crossing
        no_x_crossing = bool(lw_pos[0] < rw_pos[0])

        is_valid = sep_ok and palm_ok and thumb_ok and no_x_crossing

        return {
            "is_valid": is_valid,
            "actual_separation_m": actual_sep,
            "palm_opposition_dot": palm_dot,
            "left_thumb_dot": left_thumb_dot,
            "right_thumb_dot": right_thumb_dot,
            "no_x_crossover": no_x_crossing,
        }


def blend_skeleton_trajectories(
    rest_pose: np.ndarray,
    target_gesture: np.ndarray,
    duration_sec: float = 3.0,
    fps: int = 60,
    attack_frac: float = 0.15,
    release_frac: float = 0.20
) -> np.ndarray:
    """
    Synthesizes full multi-joint trajectory by continuous SO(3) SLERP blending.
    Guarantees that at t=0 and t=T, the pose strictly matches rest_pose (zero rest pose collapse).

    Args:
        rest_pose: [J, 4] resting joint quaternions (e.g. resting A-pose).
        target_gesture: [J, 4] peak target gesture joint quaternions (e.g. clapping, akimbo).
        duration_sec: Total gesture duration in seconds.
        fps: Frames per second.
        attack_frac: Attack envelope ratio (default 15%).
        release_frac: Release envelope ratio (default 20%).

    Returns:
        blended_traj: [F, J, 4] interpolated motion trajectory.
    """
    num_frames = int(round(duration_sec * fps))
    env = fast_attack_trapezoid_envelope(num_frames, attack_frac, release_frac)

    # Broadcast rest_pose and target_gesture to [F, J, 4]
    rest_batch = np.broadcast_to(rest_pose[None, :, :], (num_frames, rest_pose.shape[0], 4))
    target_batch = np.broadcast_to(target_gesture[None, :, :], (num_frames, target_gesture.shape[0], 4))

    return quat_slerp_batch(rest_batch, target_batch, env)
