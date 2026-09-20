#ifndef KIMODO_BLEND_HPP
#define KIMODO_BLEND_HPP

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <array>
#include <vector>
#include <span>

namespace kimodo {

/**
 * @brief High-precision unit quaternion representing SO(3) rotations.
 * Adheres to Hamilton convention: q = w + x*i + y*j + z*k, with i^2 = j^2 = k^2 = ijk = -1.
 */
struct Quaternion {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float w{1.0f};

    constexpr Quaternion() noexcept = default;
    constexpr Quaternion(float x, float y, float z, float w) noexcept : x(x), y(y), z(z), w(w) {}

    [[nodiscard]] constexpr bool operator==(const Quaternion& other) const noexcept {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }

    [[nodiscard]] constexpr bool operator!=(const Quaternion& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] float norm_sq() const noexcept {
        return x * x + y * y + z * z + w * w;
    }

    [[nodiscard]] float norm() const noexcept {
        return std::sqrt(norm_sq());
    }

    void normalize() noexcept {
        float n_sq = norm_sq();
        if (n_sq > 1e-12f) {
            float inv = 1.0f / std::sqrt(n_sq);
            x *= inv;
            y *= inv;
            z *= inv;
            w *= inv;
        } else {
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
            w = 1.0f;
        }
    }

    [[nodiscard]] Quaternion normalized() const noexcept {
        Quaternion q = *this;
        q.normalize();
        return q;
    }

    [[nodiscard]] constexpr Quaternion conjugate() const noexcept {
        return {-x, -y, -z, w};
    }

    [[nodiscard]] Quaternion inverse() const noexcept {
        float n_sq = norm_sq();
        if (n_sq > 1e-12f) {
            float inv = 1.0f / n_sq;
            return {-x * inv, -y * inv, -z * inv, w * inv};
        }
        return {0.0f, 0.0f, 0.0f, 1.0f};
    }

    [[nodiscard]] static constexpr float dot(const Quaternion& q1, const Quaternion& q2) noexcept {
        return q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    }

    /**
     * @brief Construct quaternion from intrinsic Euler XYZ rotation angles (in radians).
     * R = R_x(rx) * R_y(ry) * R_z(rz)
     */
    [[nodiscard]] static Quaternion from_euler_xyz(float rx, float ry, float rz) noexcept {
        float hx = rx * 0.5f;
        float hy = ry * 0.5f;
        float hz = rz * 0.5f;

        float cx = std::cos(hx), sx = std::sin(hx);
        float cy = std::cos(hy), sy = std::sin(hy);
        float cz = std::cos(hz), sz = std::sin(hz);

        Quaternion q{
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
            cx * cy * cz - sx * sy * sz
        };
        q.normalize();
        return q;
    }

    /**
     * @brief Extract Euler XYZ angles (in radians) from quaternion.
     * R = R_x(rx) * R_y(ry) * R_z(rz)
     */
    [[nodiscard]] std::array<float, 3> to_euler_xyz() const noexcept {
        std::array<float, 3> euler{};

        // R02 = 2*(x*z + w*y) = sin(ry)
        float sin_ry = 2.0f * (x * z + w * y);
        float sin_ry_clipped = std::clamp(sin_ry, -1.0f, 1.0f);
        euler[1] = std::asin(sin_ry_clipped);

        if (std::abs(sin_ry) >= (1.0f - 1e-6f)) {
            // Gimbal lock (ry = +/- pi/2): set rz = 0, rx = atan2(2*(xy + wz), 1 - 2*(xx + zz))
            euler[2] = 0.0f;
            euler[0] = std::atan2(2.0f * (x * y + w * z), 1.0f - 2.0f * (x * x + z * z));
        } else {
            // Standard non-gimbal extraction:
            // -R12 = 2*(w*x - y*z), R22 = 1 - 2*(x*x + y*y)
            euler[0] = std::atan2(2.0f * (w * x - y * z), 1.0f - 2.0f * (x * x + y * y));
            // -R01 = 2*(w*z - x*y), R00 = 1 - 2*(y*y + z*z)
            euler[2] = std::atan2(2.0f * (w * z - x * y), 1.0f - 2.0f * (y * y + z * z));
        }

        return euler;
    }

    /**
     * @brief Exact geodesic distance (angular difference in radians) on SO(3).
     * Uses relative quaternion dq = q1^{-1} * q2 and 2*atan2(||v||, |w|) to avoid acos cancellation.
     */
    [[nodiscard]] static float geodesic_angle(const Quaternion& q1, const Quaternion& q2) noexcept {
        // Compute relative quaternion dq = q1^{-1} * q2
        float xd = q1.w * q2.x - q1.x * q2.w - q1.y * q2.z + q1.z * q2.y;
        float yd = q1.w * q2.y + q1.x * q2.z - q1.y * q2.w - q1.z * q2.x;
        float zd = q1.w * q2.z - q1.x * q2.y + q1.y * q2.x - q1.z * q2.w;
        float wd = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        float v_norm = std::sqrt(xd * xd + yd * yd + zd * zd);
        return 2.0f * std::atan2(v_norm, std::abs(wd));
    }

    /**
     * @brief Spherical Linear Interpolation (SLERP) on SO(3) taking shortest geodesic path.
     * Includes NLERP fallback for near-parallel quaternions (dot > 0.9995f).
     */
    [[nodiscard]] static Quaternion slerp(const Quaternion& q1, Quaternion q2, float t) noexcept {
        float d = dot(q1, q2);

        // Shortest-path sign alignment on 3-sphere
        if (d < 0.0f) {
            d = -d;
            q2.x = -q2.x;
            q2.y = -q2.y;
            q2.z = -q2.z;
            q2.w = -q2.w;
        }

        // Near-parallel threshold: use normalized linear interpolation (NLERP)
        if (d > 0.9995f) {
            Quaternion r{
                q1.x + t * (q2.x - q1.x),
                q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z),
                q1.w + t * (q2.w - q1.w)
            };
            r.normalize();
            return r;
        }

        float theta_0 = std::acos(std::clamp(d, -1.0f, 1.0f));
        float theta = theta_0 * t;
        float sin_theta = std::sin(theta);
        float sin_theta_0 = std::sin(theta_0);

        float s0 = std::cos(theta) - d * sin_theta / sin_theta_0;
        float s1 = sin_theta / sin_theta_0;

        Quaternion r{
            (s0 * q1.x) + (s1 * q2.x),
            (s0 * q1.y) + (s1 * q2.y),
            (s0 * q1.z) + (s1 * q2.z),
            (s0 * q1.w) + (s1 * q2.w)
        };
        r.normalize();
        return r;
    }

    /**
     * @brief Rotate a 3D vector by this quaternion: v' = q * v * q^{-1}
     */
    [[nodiscard]] std::array<float, 3> rotate_vector(const std::array<float, 3>& v) const noexcept {
        float qx = x, qy = y, qz = z, qw = w;
        
        float tx = 2.0f * (qy * v[2] - qz * v[1]);
        float ty = 2.0f * (qz * v[0] - qx * v[2]);
        float tz = 2.0f * (qx * v[1] - qy * v[0]);

        return {
            v[0] + qw * tx + (qy * tz - qz * ty),
            v[1] + qw * ty + (qz * tx - qx * tz),
            v[2] + qw * tz + (qx * ty - qy * tx)
        };
    }

    /**
     * @brief Hamilton quaternion product: q_out = q1 * q2
     */
    [[nodiscard]] constexpr Quaternion operator*(const Quaternion& q2) const noexcept {
        return {
            w * q2.x + x * q2.w + y * q2.z - z * q2.y,
            w * q2.y - x * q2.z + y * q2.w + z * q2.x,
            w * q2.z + x * q2.y - y * q2.x + z * q2.w,
            w * q2.w - x * q2.x - y * q2.y - z * q2.z
        };
    }
};

/**
 * @brief Fast-Attack Trapezoidal Motion Envelope Generator.
 * Implements C1-continuous Hermite cubic smoothstep easing: S(s) = s^2 * (3 - 2s).
 * Default profile: 15% attack ramp, 65% steady hold plateau, 20% release ramp.
 */
class MotionEnvelope {
public:
    /**
     * @brief Compute instantaneous motion envelope factor E(p) in [0, 1].
     * @param progress Normalized time in [0, 1].
     * @param attack_frac Fraction of total duration dedicated to attack (default 0.15 = 15%).
     * @param release_frac Fraction of total duration dedicated to release (default 0.20 = 20%).
     */
    [[nodiscard]] static constexpr float fast_attack_trapezoid(
        float progress,
        float attack_frac = 0.15f,
        float release_frac = 0.20f
    ) noexcept {
        float p = std::clamp(progress, 0.0f, 1.0f);
        float att = std::clamp(attack_frac, 0.001f, 0.999f);
        float rel = std::clamp(release_frac, 0.001f, 1.0f - att);

        if (p < att) {
            float s = p / att;
            return s * s * (3.0f - 2.0f * s);
        }
        if (p <= (1.0f - rel)) {
            return 1.0f;
        }
        float s = (1.0f - p) / rel;
        return s * s * (3.0f - 2.0f * s);
    }

    /**
     * @brief Standard Hermite cubic smoothstep S(s) for s in [0, 1].
     */
    [[nodiscard]] static constexpr float cubic_smoothstep(float s) noexcept {
        float sc = std::clamp(s, 0.0f, 1.0f);
        return sc * sc * (3.0f - 2.0f * sc);
    }

    /**
     * @brief Evaluate trapezoidal envelope at explicit time t with absolute attack and release durations.
     */
    [[nodiscard]] static constexpr float evaluate_timed(
        float t,
        float total_duration,
        float attack_sec = 0.45f,
        float release_sec = 0.60f
    ) noexcept {
        if (total_duration <= 1e-5f) {
            return 0.0f;
        }
        float p = t / total_duration;
        float att_frac = attack_sec / total_duration;
        float rel_frac = release_sec / total_duration;
        return fast_attack_trapezoid(p, att_frac, rel_frac);
    }
};

/**
 * @brief Geodesic Angular Deadband Filter for joint orientation streams.
 * Eliminates 60Hz floating-point micro-jitter below threshold (default epsilon = 0.0005 rad / 0.0286 deg)
 * and applies exponential damping alpha = 1 - exp(-lambda * delta_sec) above threshold.
 */
class AngularDeadbandFilter {
private:
    float epsilon_{0.0005f}; // 0.0286 degrees
    float lambda_{18.0f};    // Damping coefficient (18.0 - 24.0 s^-1)

public:
    constexpr explicit AngularDeadbandFilter(float epsilon = 0.0005f, float lambda = 18.0f) noexcept
        : epsilon_(epsilon), lambda_(lambda) {}

    [[nodiscard]] constexpr float get_epsilon() const noexcept { return epsilon_; }
    constexpr void set_epsilon(float eps) noexcept { epsilon_ = eps; }

    [[nodiscard]] constexpr float get_lambda() const noexcept { return lambda_; }
    constexpr void set_lambda(float lam) noexcept { lambda_ = lam; }

    [[nodiscard]] bool is_in_deadband(const Quaternion& current, const Quaternion& target) const noexcept {
        return Quaternion::geodesic_angle(current, target) < epsilon_;
    }

    /**
     * @brief Filter target quaternion against current accepted state.
     * @param current The last accepted joint quaternion.
     * @param target The incoming candidate quaternion.
     * @param delta_sec Time delta since last update.
     * @return Filtered quaternion state.
     */
    [[nodiscard]] Quaternion filter(
        const Quaternion& current,
        const Quaternion& target,
        float delta_sec
    ) const noexcept {
        float delta_rad = Quaternion::geodesic_angle(current, target);
        if (delta_rad < epsilon_) {
            // Drop update: inside deadband zone, lock to current to eliminate jitter
            return current;
        }
        // Apply continuous-time exponential damping
        float dt_clamped = std::min(std::max(delta_sec, 0.0f), 0.033f);
        float alpha = 1.0f - std::exp(-lambda_ * dt_clamped);
        return Quaternion::slerp(current, target, alpha);
    }
};

/**
 * @brief Bilateral Clapping and Contact Envelopes Constraint Checker.
 * Enforces:
 * 1. Target wrist separation d_wrist = 17.0 cm (0.170 m, half-width 0.085 m).
 * 2. Opposing inward-facing palms (n_left . n_right <= -0.90).
 * 3. Vertical thumbs (t_left . y_up >= +0.90, t_right . y_up >= +0.90).
 * 4. Prevention of face-obstructing "X" forearm crossover.
 */
class BilateralClappingConstraint {
public:
    static constexpr float TARGET_WRIST_SEPARATION_M = 0.170f;
    static constexpr float HALF_WRIST_SEPARATION_M = 0.085f;
    static constexpr float MIN_PALM_OPPOSITION_DOT = -0.90f;
    static constexpr float MIN_THUMB_UPWARD_DOT = 0.90f;

    struct ValidationResult {
        bool is_valid{false};
        float actual_separation_m{0.0f};
        float palm_dot{0.0f};
        float left_thumb_dot{0.0f};
        float right_thumb_dot{0.0f};
        bool no_x_crossover{true};
    };

    /**
     * @brief Validate humanoid bimanual clapping kinematics.
     * @param left_wrist_pos Left wrist position in world space (meters).
     * @param right_wrist_pos Right wrist position in world space (meters).
     * @param left_wrist_rot Left wrist orientation quaternion.
     * @param right_wrist_rot Right wrist orientation quaternion.
     * @param separation_tol_m Maximum allowable wrist separation error (meters).
     */
    [[nodiscard]] static ValidationResult validate(
        const std::array<float, 3>& left_wrist_pos,
        const std::array<float, 3>& right_wrist_pos,
        const Quaternion& left_wrist_rot,
        const Quaternion& right_wrist_rot,
        float separation_tol_m = 0.03f
    ) noexcept {
        ValidationResult res{};

        // 1. Separation distance
        float dx = right_wrist_pos[0] - left_wrist_pos[0];
        float dy = right_wrist_pos[1] - left_wrist_pos[1];
        float dz = right_wrist_pos[2] - left_wrist_pos[2];
        res.actual_separation_m = std::sqrt(dx * dx + dy * dy + dz * dz);

        bool sep_ok = std::abs(res.actual_separation_m - TARGET_WRIST_SEPARATION_M) <= separation_tol_m;

        // 2. Palm normal alignment (assuming local +X is palm outward for left, -X for right)
        std::array<float, 3> left_palm_local{1.0f, 0.0f, 0.0f};
        std::array<float, 3> right_palm_local{-1.0f, 0.0f, 0.0f};

        auto left_palm_world = left_wrist_rot.rotate_vector(left_palm_local);
        auto right_palm_world = right_wrist_rot.rotate_vector(right_palm_local);

        res.palm_dot = left_palm_world[0] * right_palm_world[0] +
                       left_palm_world[1] * right_palm_world[1] +
                       left_palm_world[2] * right_palm_world[2];

        bool palm_ok = res.palm_dot <= MIN_PALM_OPPOSITION_DOT;

        // 3. Thumb alignment (assuming local +Y is thumb upward)
        std::array<float, 3> thumb_local{0.0f, 1.0f, 0.0f};
        auto left_thumb_world = left_wrist_rot.rotate_vector(thumb_local);
        auto right_thumb_world = right_wrist_rot.rotate_vector(thumb_local);

        res.left_thumb_dot = left_thumb_world[1];
        res.right_thumb_dot = right_thumb_world[1];

        bool thumb_ok = (res.left_thumb_dot >= MIN_THUMB_UPWARD_DOT) &&
                        (res.right_thumb_dot >= MIN_THUMB_UPWARD_DOT);

        // 4. X-crossover check: Left wrist must remain to the left of the right wrist in coronal plane
        res.no_x_crossover = (left_wrist_pos[0] < right_wrist_pos[0]);

        res.is_valid = sep_ok && palm_ok && thumb_ok && res.no_x_crossover;
        return res;
    }
};

/**
 * @brief Blend an entire humanoid skeleton pose between resting baseline and target gesture.
 * Guarantees zero resting collapse at t=0 by interpolating on SO(3) from resting quaternions.
 */
inline void blend_skeleton_pose(
    const Quaternion* q_rest,
    const Quaternion* q_target,
    Quaternion* q_out,
    std::size_t num_joints,
    float envelope_progress,
    float attack_frac = 0.15f,
    float release_frac = 0.20f
) noexcept {
    float env = MotionEnvelope::fast_attack_trapezoid(envelope_progress, attack_frac, release_frac);
    for (std::size_t i = 0; i < num_joints; ++i) {
        q_out[i] = Quaternion::slerp(q_rest[i], q_target[i], env);
    }
}

} // namespace kimodo

#endif // KIMODO_BLEND_HPP
