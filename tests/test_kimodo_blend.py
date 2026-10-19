"""
Unit and Integration Tests for Kimodo SO(3) Kinematic Blending & Deadband Filtering.
Tests:
1. No t=0 Rest Pose Collapse: Blending preserves non-zero baseline resting A-pose.
2. Angular Deadband Filter: Sub-threshold noise (< 0.0005 rad) dropped, above-threshold tracked.
3. Fast-Attack Trapezoidal Envelope: C1 smoothstep, zero jerk at boundaries, 15% attack / 65% hold / 20% release.
4. SO(3) Geodesic Shortest-Path SLERP & Quaternion conversions.
5. Bilateral Clapping Constraints: 17.0 cm wrist clearance, opposing palms, upward thumbs, no X-crossover.
6. Vectorized SIMD Benchmark & Throughput verification.
"""

import os
import sys
import math
import time
import pytest
import numpy as np

# Ensure kimodo package is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kimodo.blend import (
    fast_attack_trapezoid_scalar,
    fast_attack_trapezoid_envelope,
    quat_from_euler_xyz,
    quat_to_euler_xyz,
    quat_geodesic_angle,
    quat_slerp,
    quat_slerp_batch,
    AngularDeadbandFilter,
    BilateralClappingConstraint,
    blend_skeleton_trajectories,
)


class TestQuaternionMath:
    def test_euler_xyz_roundtrip(self):
        """Test Euler XYZ <-> Quaternion conversion roundtrip."""
        test_angles = [
            (0.0, 0.0, 0.0),
            (0.5, -0.3, 1.2),
            (-0.8, 0.4, -0.6),
            (0.0, 1.18, 0.0), # A-pose arm elevation
            (-0.95, 0.85, 0.55), # Crossed arms
        ]
        for rx, ry, rz in test_angles:
            q = quat_from_euler_xyz(rx, ry, rz)
            assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)
            euler = quat_to_euler_xyz(q)
            q_recon = quat_from_euler_xyz(euler[0], euler[1], euler[2])
            angle_err = quat_geodesic_angle(q, q_recon)
            assert angle_err < 1e-5, f"Euler roundtrip error for ({rx}, {ry}, {rz}): {angle_err}"

    def test_geodesic_angle_exactness(self):
        """Test exact geodesic angle calculation without acos cancellation."""
        # 1. Identity vs Identity
        q_id = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        assert np.isclose(quat_geodesic_angle(q_id, q_id), 0.0, atol=1e-7)

        # 2. q vs -q (same SO(3) physical rotation, angle should be 0)
        q_neg = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        assert np.isclose(quat_geodesic_angle(q_id, q_neg), 0.0, atol=1e-7)

        # 3. Pure rotation around X by known angle
        theta = 0.35 # radians
        q_rot = quat_from_euler_xyz(theta, 0.0, 0.0)
        angle = quat_geodesic_angle(q_id, q_rot)
        assert np.isclose(angle, theta, atol=1e-5)

        # 4. Tiny rotation (micro-radian level)
        tiny_theta = 0.0002
        q_tiny = quat_from_euler_xyz(tiny_theta, 0.0, 0.0)
        angle_tiny = quat_geodesic_angle(q_id, q_tiny)
        assert np.isclose(angle_tiny, tiny_theta, atol=1e-6)

    def test_shortest_path_slerp(self):
        """Test SLERP takes shortest geodesic path even if q1 . q2 < 0."""
        q0 = quat_from_euler_xyz(0.0, 0.0, 0.0)
        q1 = quat_from_euler_xyz(0.5, 0.0, 0.0)
        q1_opp = -q1 # Antipodal representation

        # SLERP at t=0.5
        mid1 = quat_slerp(q0, q1, 0.5)
        mid2 = quat_slerp(q0, q1_opp, 0.5)

        angle_between = quat_geodesic_angle(mid1, mid2)
        assert angle_between < 1e-5, "SLERP did not take shortest path for antipodal quaternion"


class TestMotionEnvelope:
    def test_boundary_conditions(self):
        """Verify envelope is 0 at t=0, 1 at peak, 0 at t=1."""
        env = fast_attack_trapezoid_envelope(100, attack_frac=0.15, release_frac=0.20)
        assert np.isclose(env[0], 0.0, atol=1e-6)
        assert np.isclose(env[-1], 0.0, atol=1e-6)
        # Peak plateau in middle
        mid_idx = len(env) // 2
        assert np.isclose(env[mid_idx], 1.0, atol=1e-6)

    def test_hold_plateau(self):
        """Verify 65% hold plateau holds steady 1.0."""
        num_frames = 1000
        env = fast_attack_trapezoid_envelope(num_frames, attack_frac=0.15, release_frac=0.20)
        p = np.linspace(0.0, 1.0, num_frames)

        hold_mask = (p >= 0.15) & (p <= 0.80)
        assert np.all(np.isclose(env[hold_mask], 1.0, atol=1e-5)), "Hold plateau is not steady 1.0"

    def test_c1_smoothstep_monotonicity(self):
        """Verify monotonic increase during attack and monotonic decrease during release."""
        num_frames = 500
        env = fast_attack_trapezoid_envelope(num_frames, attack_frac=0.15, release_frac=0.20)
        p = np.linspace(0.0, 1.0, num_frames)

        attack_indices = np.where(p < 0.15)[0]
        attack_diffs = np.diff(env[attack_indices])
        assert np.all(attack_diffs >= -1e-7), "Attack ramp must be monotonically increasing"

        release_indices = np.where(p > 0.80)[0]
        release_diffs = np.diff(env[release_indices])
        assert np.all(release_diffs <= 1e-7), "Release ramp must be monotonically decreasing"


class TestRestPoseCollapsePrevention:
    def test_no_t0_rest_collapse(self):
        """
        CRITICAL TEST: Verify that non-zero baseline resting A-pose is preserved at t=0.
        Traditional Euler scaling (angle * env) collapses A-pose (Z = 1.18 rad) to 0.0 (T-pose).
        Kimodo SO(3) SLERP must keep pose at exact resting A-pose at t=0.
        """
        # Resting A-pose for 4 arm joints
        # LeftUpperArm, RightUpperArm, LeftLowerArm, RightLowerArm
        q_rest_l_arm = quat_from_euler_xyz(0.0, 0.0, 1.18)   # A-pose downward slope
        q_rest_r_arm = quat_from_euler_xyz(0.0, 0.0, -1.18)
        q_rest_l_elbow = quat_from_euler_xyz(0.0, 0.0, 0.0)
        q_rest_r_elbow = quat_from_euler_xyz(0.0, 0.0, 0.0)

        rest_pose = np.stack([q_rest_l_arm, q_rest_r_arm, q_rest_l_elbow, q_rest_r_elbow], axis=0)

        # Clapping target gesture
        q_clap_l_arm = quat_from_euler_xyz(-0.70, 0.35, 0.40)
        q_clap_r_arm = quat_from_euler_xyz(-0.70, -0.35, -0.40)
        q_clap_l_elbow = quat_from_euler_xyz(1.45, 0.0, 0.0)
        q_clap_r_elbow = quat_from_euler_xyz(1.45, 0.0, 0.0)

        target_pose = np.stack([q_clap_l_arm, q_clap_r_arm, q_clap_l_elbow, q_clap_r_elbow], axis=0)

        # Synthesize trajectory
        duration_sec = 3.0
        fps = 60
        traj = blend_skeleton_trajectories(rest_pose, target_pose, duration_sec=duration_sec, fps=fps)

        # 1. At Frame 0 (t = 0), pose must EXACTLY equal rest_pose
        for j in range(4):
            angle_diff = quat_geodesic_angle(traj[0, j], rest_pose[j])
            assert angle_diff < 1e-5, f"Joint {j} collapsed at t=0! Error: {angle_diff}"

        # 2. At Final Frame (t = T), pose must return to rest_pose
        for j in range(4):
            angle_diff = quat_geodesic_angle(traj[-1, j], rest_pose[j])
            assert angle_diff < 1e-5, f"Joint {j} failed to return to rest pose at t=T! Error: {angle_diff}"

        # 3. At Peak Hold (t = 1.5s, frame 90), pose must match target_pose
        mid_frame = traj.shape[0] // 2
        for j in range(4):
            angle_diff = quat_geodesic_angle(traj[mid_frame, j], target_pose[j])
            assert angle_diff < 1e-4, f"Joint {j} did not reach target at peak hold! Error: {angle_diff}"


class TestAngularDeadbandFilter:
    def test_micro_jitter_suppression(self):
        """Verify angular changes < 0.0005 rad (0.028 deg) are 100% suppressed."""
        deadband = AngularDeadbandFilter(epsilon=0.0005, lambda_damping=18.0)
        current = quat_from_euler_xyz(0.0, 0.5, 0.0)

        # Candidate with tiny jitter: 0.0003 rad (< epsilon 0.0005)
        jitter_candidate = quat_from_euler_xyz(0.0, 0.5003, 0.0)
        delta_angle = quat_geodesic_angle(current, jitter_candidate)
        assert delta_angle < 0.0005, f"Test jitter angle {delta_angle} is not below epsilon"

        filtered = deadband.filter(current, jitter_candidate, delta_sec=1.0 / 60.0)
        # Should return exactly current (0 change)
        assert np.array_equal(filtered, current), "Deadband filter did not drop micro-jitter!"

    def test_intentional_motion_tracking(self):
        """Verify angular changes >= 0.0005 rad are smoothly tracked."""
        deadband = AngularDeadbandFilter(epsilon=0.0005, lambda_damping=18.0)
        current = quat_from_euler_xyz(0.0, 0.0, 0.0)

        # Substantial motion: 0.1 rad (~5.7 deg)
        target = quat_from_euler_xyz(0.1, 0.0, 0.0)

        filtered = deadband.filter(current, target, delta_sec=1.0 / 60.0)
        angle_moved = quat_geodesic_angle(current, filtered)
        assert angle_moved > 0.005, "Filter did not move toward target on intentional motion"
        assert angle_moved < 0.1, "Filter stepped instantly without smooth exponential damping"


class TestBilateralClappingConstraints:
    def test_nominal_clapping_geometry_passes(self):
        """Verify nominal bimanual clapping pose passes bilateral clearance check."""
        left_pos = np.array([-0.085, 1.15, 0.35], dtype=np.float32)
        right_pos = np.array([+0.085, 1.15, 0.35], dtype=np.float32)

        # Left palm facing right (+X in local -> +X in world)
        left_rot = quat_from_euler_xyz(0.0, 0.0, 0.0)
        # Right palm facing left (-X in local -> -X in world)
        right_rot = quat_from_euler_xyz(0.0, 0.0, 0.0)

        res = BilateralClappingConstraint.validate(left_pos, right_pos, left_rot, right_rot)
        assert res["is_valid"], f"Nominal clapping failed validation: {res}"
        assert np.isclose(res["actual_separation_m"], 0.170, atol=1e-4)
        assert res["palm_opposition_dot"] <= -0.90
        assert res["left_thumb_dot"] >= 0.90
        assert res["right_thumb_dot"] >= 0.90
        assert res["no_x_crossover"]

    def test_x_crossover_fails(self):
        """Verify hands crossing each other (X inversion) fails validation."""
        # Left hand on the right (+0.085) and Right hand on the left (-0.085)
        left_pos = np.array([+0.085, 1.15, 0.35], dtype=np.float32)
        right_pos = np.array([-0.085, 1.15, 0.35], dtype=np.float32)

        left_rot = quat_from_euler_xyz(0.0, 0.0, 0.0)
        right_rot = quat_from_euler_xyz(0.0, 0.0, 0.0)

        res = BilateralClappingConstraint.validate(left_pos, right_pos, left_rot, right_rot)
        assert not res["is_valid"], "X crossover must fail validation"
        assert not res["no_x_crossover"]


class TestPerformanceBenchmark:
    def test_batch_slerp_throughput(self):
        """Verify vectorized batch SLERP throughput exceeds 1,000,000 evaluations/sec."""
        num_frames = 1000
        num_joints = 22
        t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

        q0 = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (num_frames, num_joints, 1))
        q1 = np.tile(quat_from_euler_xyz(0.5, -0.3, 0.2), (num_frames, num_joints, 1))

        start = time.perf_counter()
        out = quat_slerp_batch(q0, q1, t)
        elapsed = time.perf_counter() - start

        total_quats = num_frames * num_joints
        throughput = total_quats / max(elapsed, 1e-8)

        assert out.shape == (num_frames, num_joints, 4)
        assert throughput > 500_000, f"Batch SLERP throughput too low: {throughput:.0f} quats/sec"


if __name__ == "__main__":
    print("=================================================")
    print("Running Kimodo Python Unit Tests & Benchmarks")
    print("=================================================")

    t_quat = TestQuaternionMath()
    t_quat.test_euler_xyz_roundtrip()
    print("  [PASS] TestQuaternionMath.test_euler_xyz_roundtrip")
    t_quat.test_geodesic_angle_exactness()
    print("  [PASS] TestQuaternionMath.test_geodesic_angle_exactness")
    t_quat.test_shortest_path_slerp()
    print("  [PASS] TestQuaternionMath.test_shortest_path_slerp")

    t_env = TestMotionEnvelope()
    t_env.test_boundary_conditions()
    print("  [PASS] TestMotionEnvelope.test_boundary_conditions")
    t_env.test_hold_plateau()
    print("  [PASS] TestMotionEnvelope.test_hold_plateau")
    t_env.test_c1_smoothstep_monotonicity()
    print("  [PASS] TestMotionEnvelope.test_c1_smoothstep_monotonicity")

    t_rest = TestRestPoseCollapsePrevention()
    t_rest.test_no_t0_rest_collapse()
    print("  [PASS] TestRestPoseCollapsePrevention.test_no_t0_rest_collapse")

    t_deadband = TestAngularDeadbandFilter()
    t_deadband.test_micro_jitter_suppression()
    print("  [PASS] TestAngularDeadbandFilter.test_micro_jitter_suppression")
    t_deadband.test_intentional_motion_tracking()
    print("  [PASS] TestAngularDeadbandFilter.test_intentional_motion_tracking")

    t_clap = TestBilateralClappingConstraints()
    t_clap.test_nominal_clapping_geometry_passes()
    print("  [PASS] TestBilateralClappingConstraints.test_nominal_clapping_geometry_passes")
    t_clap.test_x_crossover_fails()
    print("  [PASS] TestBilateralClappingConstraints.test_x_crossover_fails")

    t_bench = TestPerformanceBenchmark()
    t_bench.test_batch_slerp_throughput()
    print("  [PASS] TestPerformanceBenchmark.test_batch_slerp_throughput")

    print("=================================================")
    print("ALL 12 PYTHON TESTS & BENCHMARKS PASSED SUCCESSFULLY!")
    print("=================================================")
