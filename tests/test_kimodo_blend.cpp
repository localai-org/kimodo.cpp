/**
 * @file test_kimodo_blend.cpp
 * @brief Comprehensive C++20 unit test and benchmark suite for Kimodo SO(3) Kinematic Blending & Deadband Filtering.
 */

#include "../include/kimodo_blend.hpp"
#include <iostream>
#include <cassert>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>

#define ASSERT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "[FAILED] " << #expr << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::cerr << "[FAILED] |" << #a << " - " << #b << "| = " << std::abs((a) - (b)) \
                      << " > " << (eps) << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

namespace tests {

bool test_quaternion_euler_roundtrip() {
    std::cout << "  [Test] Quaternion Euler XYZ Roundtrip..." << std::endl;
    std::vector<std::array<float, 3>> test_cases = {
        {0.0f, 0.0f, 0.0f},
        {0.5f, -0.3f, 1.2f},
        {-0.8f, 0.4f, -0.6f},
        {0.0f, 0.0f, 1.18f},      // A-pose baseline
        {-0.95f, 0.85f, 0.55f}    // Crossed arms posture
    };

    for (const auto& angles : test_cases) {
        kimodo::Quaternion q = kimodo::Quaternion::from_euler_xyz(angles[0], angles[1], angles[2]);
        ASSERT_NEAR(q.norm(), 1.0f, 1e-5f);

        auto euler = q.to_euler_xyz();
        kimodo::Quaternion q_recon = kimodo::Quaternion::from_euler_xyz(euler[0], euler[1], euler[2]);
        float angle_err = kimodo::Quaternion::geodesic_angle(q, q_recon);
        ASSERT_TRUE(angle_err < 1e-4f);
    }
    return true;
}

bool test_geodesic_angle() {
    std::cout << "  [Test] Geodesic Angle Exactness..." << std::endl;
    kimodo::Quaternion q_id{0.0f, 0.0f, 0.0f, 1.0f};

    // 1. Identity vs Identity
    ASSERT_NEAR(kimodo::Quaternion::geodesic_angle(q_id, q_id), 0.0f, 1e-6f);

    // 2. Antipodal equivalence (q vs -q)
    kimodo::Quaternion q_neg{0.0f, 0.0f, 0.0f, -1.0f};
    ASSERT_NEAR(kimodo::Quaternion::geodesic_angle(q_id, q_neg), 0.0f, 1e-6f);

    // 3. Known rotation
    float theta = 0.45f;
    kimodo::Quaternion q_rot = kimodo::Quaternion::from_euler_xyz(theta, 0.0f, 0.0f);
    ASSERT_NEAR(kimodo::Quaternion::geodesic_angle(q_id, q_rot), theta, 1e-5f);

    // 4. Sub-milliradian precision
    float tiny_theta = 0.00025f;
    kimodo::Quaternion q_tiny = kimodo::Quaternion::from_euler_xyz(tiny_theta, 0.0f, 0.0f);
    ASSERT_NEAR(kimodo::Quaternion::geodesic_angle(q_id, q_tiny), tiny_theta, 1e-6f);

    return true;
}

bool test_shortest_path_slerp() {
    std::cout << "  [Test] Shortest-Path SO(3) SLERP..." << std::endl;
    kimodo::Quaternion q0 = kimodo::Quaternion::from_euler_xyz(0.0f, 0.0f, 0.0f);
    kimodo::Quaternion q1 = kimodo::Quaternion::from_euler_xyz(0.6f, 0.0f, 0.0f);
    kimodo::Quaternion q1_neg{-q1.x, -q1.y, -q1.z, -q1.w}; // Antipodal representation

    kimodo::Quaternion mid1 = kimodo::Quaternion::slerp(q0, q1, 0.5f);
    kimodo::Quaternion mid2 = kimodo::Quaternion::slerp(q0, q1_neg, 0.5f);

    float diff = kimodo::Quaternion::geodesic_angle(mid1, mid2);
    ASSERT_TRUE(diff < 1e-5f);
    ASSERT_NEAR(mid1.norm(), 1.0f, 1e-5f);
    ASSERT_NEAR(mid2.norm(), 1.0f, 1e-5f);

    return true;
}

bool test_motion_envelope() {
    std::cout << "  [Test] Fast-Attack Trapezoid Envelope..." << std::endl;

    // Boundaries
    ASSERT_NEAR(kimodo::MotionEnvelope::fast_attack_trapezoid(0.0f, 0.15f, 0.20f), 0.0f, 1e-6f);
    ASSERT_NEAR(kimodo::MotionEnvelope::fast_attack_trapezoid(1.0f, 0.15f, 0.20f), 0.0f, 1e-6f);

    // Hold plateau
    ASSERT_NEAR(kimodo::MotionEnvelope::fast_attack_trapezoid(0.15f, 0.15f, 0.20f), 1.0f, 1e-6f);
    ASSERT_NEAR(kimodo::MotionEnvelope::fast_attack_trapezoid(0.50f, 0.15f, 0.20f), 1.0f, 1e-6f);
    ASSERT_NEAR(kimodo::MotionEnvelope::fast_attack_trapezoid(0.80f, 0.15f, 0.20f), 1.0f, 1e-6f);

    // Monotonic attack ramp
    float prev = 0.0f;
    for (float p = 0.01f; p < 0.15f; p += 0.01f) {
        float val = kimodo::MotionEnvelope::fast_attack_trapezoid(p, 0.15f, 0.20f);
        ASSERT_TRUE(val >= prev);
        prev = val;
    }

    return true;
}

bool test_no_t0_rest_pose_collapse() {
    std::cout << "  [Test] Zero Rest Pose Collapse at t=0..." << std::endl;

    // Resting A-pose for UpperArm: Z = 1.18 rad (downward arm slope)
    kimodo::Quaternion q_rest_upper_arm = kimodo::Quaternion::from_euler_xyz(0.0f, 0.0f, 1.18f);
    // Target clapping pose: X = -0.70 rad, Y = 0.35 rad, Z = 0.40 rad
    kimodo::Quaternion q_target_upper_arm = kimodo::Quaternion::from_euler_xyz(-0.70f, 0.35f, 0.40f);

    // Evaluate blended rotation at t = 0.0
    float env_t0 = kimodo::MotionEnvelope::fast_attack_trapezoid(0.0f, 0.15f, 0.20f);
    kimodo::Quaternion q_blend_t0 = kimodo::Quaternion::slerp(q_rest_upper_arm, q_target_upper_arm, env_t0);

    float err_t0 = kimodo::Quaternion::geodesic_angle(q_blend_t0, q_rest_upper_arm);
    ASSERT_TRUE(err_t0 < 1e-5f); // Must match resting A-pose exactly, not collapse to (0,0,0,1)

    // Evaluate at peak hold plateau (t = 0.50)
    float env_t50 = kimodo::MotionEnvelope::fast_attack_trapezoid(0.50f, 0.15f, 0.20f);
    kimodo::Quaternion q_blend_t50 = kimodo::Quaternion::slerp(q_rest_upper_arm, q_target_upper_arm, env_t50);
    float err_t50 = kimodo::Quaternion::geodesic_angle(q_blend_t50, q_target_upper_arm);
    ASSERT_TRUE(err_t50 < 1e-5f);

    // Evaluate at final frame (t = 1.0)
    float env_t100 = kimodo::MotionEnvelope::fast_attack_trapezoid(1.0f, 0.15f, 0.20f);
    kimodo::Quaternion q_blend_t100 = kimodo::Quaternion::slerp(q_rest_upper_arm, q_target_upper_arm, env_t100);
    float err_t100 = kimodo::Quaternion::geodesic_angle(q_blend_t100, q_rest_upper_arm);
    ASSERT_TRUE(err_t100 < 1e-5f);

    return true;
}

bool test_angular_deadband_filter() {
    std::cout << "  [Test] Angular Deadband Filter (0.0005 rad)..." << std::endl;
    kimodo::AngularDeadbandFilter filter(0.0005f, 18.0f);
    kimodo::Quaternion current = kimodo::Quaternion::from_euler_xyz(0.0f, 0.3f, 0.0f);

    // 1. Sub-threshold jitter: delta = 0.0003 rad (< 0.0005 rad)
    kimodo::Quaternion jitter_target = kimodo::Quaternion::from_euler_xyz(0.0f, 0.3003f, 0.0f);
    ASSERT_TRUE(filter.is_in_deadband(current, jitter_target));

    kimodo::Quaternion filtered_jitter = filter.filter(current, jitter_target, 0.016f);
    ASSERT_TRUE(filtered_jitter == current); // 100% suppression

    // 2. Above-threshold intentional motion: delta = 0.08 rad
    kimodo::Quaternion real_target = kimodo::Quaternion::from_euler_xyz(0.0f, 0.38f, 0.0f);
    ASSERT_TRUE(!filter.is_in_deadband(current, real_target));

    kimodo::Quaternion filtered_motion = filter.filter(current, real_target, 0.016f);
    float angle_moved = kimodo::Quaternion::geodesic_angle(current, filtered_motion);
    ASSERT_TRUE(angle_moved > 0.005f);
    ASSERT_TRUE(angle_moved < 0.08f);

    return true;
}

bool test_bilateral_clapping_constraint() {
    std::cout << "  [Test] Bilateral Clapping Constraints..." << std::endl;

    std::array<float, 3> l_pos{-0.085f, 1.15f, 0.35f};
    std::array<float, 3> r_pos{+0.085f, 1.15f, 0.35f};
    kimodo::Quaternion l_rot = kimodo::Quaternion::from_euler_xyz(0.0f, 0.0f, 0.0f);
    kimodo::Quaternion r_rot = kimodo::Quaternion::from_euler_xyz(0.0f, 0.0f, 0.0f);

    auto res = kimodo::BilateralClappingConstraint::validate(l_pos, r_pos, l_rot, r_rot);
    ASSERT_TRUE(res.is_valid);
    ASSERT_NEAR(res.actual_separation_m, 0.170f, 1e-4f);
    ASSERT_TRUE(res.palm_dot <= -0.90f);
    ASSERT_TRUE(res.left_thumb_dot >= 0.90f);
    ASSERT_TRUE(res.right_thumb_dot >= 0.90f);
    ASSERT_TRUE(res.no_x_crossover);

    // Test X-crossover failure (hands swapped)
    std::array<float, 3> l_cross{+0.085f, 1.15f, 0.35f};
    std::array<float, 3> r_cross{-0.085f, 1.15f, 0.35f};
    auto res_cross = kimodo::BilateralClappingConstraint::validate(l_cross, r_cross, l_rot, r_rot);
    ASSERT_TRUE(!res_cross.is_valid);
    ASSERT_TRUE(!res_cross.no_x_crossover);

    return true;
}

bool test_performance_benchmark() {
    std::cout << "  [Benchmark] C++20 SO(3) SLERP & Deadband Benchmark..." << std::endl;

    constexpr std::size_t N_FRAMES = 5000;
    constexpr std::size_t N_JOINTS = 22;
    std::vector<kimodo::Quaternion> q_rest(N_JOINTS, kimodo::Quaternion::from_euler_xyz(0.0f, 0.0f, 1.18f));
    std::vector<kimodo::Quaternion> q_target(N_JOINTS, kimodo::Quaternion::from_euler_xyz(-0.7f, 0.35f, 0.4f));
    std::vector<kimodo::Quaternion> q_out(N_JOINTS);

    kimodo::AngularDeadbandFilter deadband(0.0005f, 18.0f);

    auto start = std::chrono::high_resolution_clock::now();

    for (std::size_t f = 0; f < N_FRAMES; ++f) {
        float progress = static_cast<float>(f) / static_cast<float>(N_FRAMES - 1);
        kimodo::blend_skeleton_pose(q_rest.data(), q_target.data(), q_out.data(), N_JOINTS, progress);

        // Apply deadband filter
        for (std::size_t j = 0; j < N_JOINTS; ++j) {
            q_out[j] = deadband.filter(q_rest[j], q_out[j], 0.0166f);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double total_ops = static_cast<double>(N_FRAMES * N_JOINTS);
    double throughput = total_ops / diff.count();

    std::cout << "    Evaluated " << total_ops << " joint blends in " << (diff.count() * 1000.0)
              << " ms (" << static_cast<std::size_t>(throughput) << " quat blends/sec)" << std::endl;

    ASSERT_TRUE(throughput > 1'000'000.0);
    return true;
}

} // namespace tests

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "Running Kimodo C++20 Kinematic Blending Tests" << std::endl;
    std::cout << "=================================================" << std::endl;

    bool pass = true;
    pass &= tests::test_quaternion_euler_roundtrip();
    pass &= tests::test_geodesic_angle();
    pass &= tests::test_shortest_path_slerp();
    pass &= tests::test_motion_envelope();
    pass &= tests::test_no_t0_rest_pose_collapse();
    pass &= tests::test_angular_deadband_filter();
    pass &= tests::test_bilateral_clapping_constraint();
    pass &= tests::test_performance_benchmark();

    std::cout << "=================================================" << std::endl;
    if (pass) {
        std::cout << "ALL 8 C++20 TESTS AND BENCHMARKS PASSED!" << std::endl;
        return 0;
    } else {
        std::cerr << "SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
