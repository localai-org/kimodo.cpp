#!/usr/bin/env python3
"""Convert Kimodo raw motion float buffers (.f32) to standard BVH animation file for Unreal Engine / Blender."""

import argparse
import math
import struct
import sys
from pathlib import Path

# SOMA-30 skeleton specification
SOMA30_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm",
    "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
    "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"
]

SOMA30_PARENTS = [-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28]

SOMA30_OFFSETS = [
    [0.0, 0.0, 0.0], [-0.00013727, 0.0500376256, -0.00053726669], [-1.86574103e-09, 0.0712530139, -0.000298248546],
    [-5.75188398e-09, 0.0755006305, -0.00815970992], [-0.00181676517, 0.263112953, -0.00553348292],
    [-2.85102231e-08, 0.0770939664, 0.0230258546], [-4.5975437e-08, 0.0612891595, 0.0195370861],
    [2.63687901e-05, 0.0047559225, 0.0309494062], [0.0320638079, 0.0538020513, 0.0758688308],
    [-0.0322244017, 0.05361869, 0.0755823359], [0.0162165175, 0.232371641, 0.0511341324],
    [0.149198457, 2.19397873e-08, -0.0550232576], [0.287393078, 2.50268389e-09, -2.58787737e-05],
    [0.270939812, -7.06625108e-09, 2.60897248e-05], [0.122686267, -0.0322017573, 0.0483306876],
    [0.190119595, -0.00312878387, -0.000339570373], [-0.0138011824, 0.231803086, 0.0521415786],
    [-0.150371962, 1.17387901e-07, -0.0554560437], [-0.287366393, 1.87628082e-08, -2.59709359e-05],
    [-0.271336198, -1.16767401e-09, 2.61269368e-05], [-0.122642483, -0.0321145448, 0.0480403904],
    [-0.190005945, -0.00306615542, -0.0003157343], [0.10043214, -0.0843452671, 0.0259565473],
    [-1e-08, -0.432217537, -0.00802912805], [1e-08, -0.421550959, -0.0348152298],
    [0.0, -0.0505947206, 0.132315294], [-0.10047278, -0.0829525995, 0.0262031695],
    [1e-08, -0.433622059, -0.00805555828], [2e-08, -0.421173943, -0.0347839785],
    [-3.42907669e-09, -0.0507960932, 0.132841956]
]

def quat_to_euler_zxy(x, y, z, w):
    """Convert XYZW quaternion to ZXY Euler angles in degrees."""
    # Matrix elements
    m00 = 1.0 - 2.0 * (y * y + z * z)
    m01 = 2.0 * (x * y - z * w)
    m02 = 2.0 * (x * z + y * w)
    m10 = 2.0 * (x * y + z * w)
    m11 = 1.0 - 2.0 * (x * x + z * z)
    m12 = 2.0 * (y * z - x * w)
    m20 = 2.0 * (x * z - y * w)
    m21 = 2.0 * (y * z + x * w)
    m22 = 1.0 - 2.0 * (x * x + y * y)

    # Extract ZXY
    asin_val = max(-1.0, min(1.0, -m12))
    rot_x = math.asin(asin_val)
    if abs(rot_x) < math.pi / 2 - 1e-4:
        rot_z = math.atan2(m10, m11)
        rot_y = math.atan2(m02, m22)
    else:
        rot_z = math.atan2(-m01, m00)
        rot_y = 0.0

    return math.degrees(rot_z), math.degrees(rot_x), math.degrees(rot_y)

def build_bvh_hierarchy(names, parents, offsets, scale=100.0):
    """Generate BVH HIERARCHY section."""
    children = {i: [] for i in range(len(names))}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)

    lines = ["HIERARCHY"]

    def write_joint(idx, indent=""):
        name = names[idx]
        off = offsets[idx]
        scaled_off = f"{off[0]*scale:.4f} {off[1]*scale:.4f} {off[2]*scale:.4f}"
        
        if parents[idx] == -1:
            lines.append(f"{indent}ROOT {name}")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}\tOFFSET {scaled_off}")
            lines.append(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{indent}JOINT {name}")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}\tOFFSET {scaled_off}")
            lines.append(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation")

        if len(children[idx]) == 0:
            lines.append(f"{indent}\tEnd Site")
            lines.append(f"{indent}\t{{")
            lines.append(f"{indent}\t\tOFFSET 0.0 1.0 0.0")
            lines.append(f"{indent}\t}}")
        else:
            for child in children[idx]:
                write_joint(child, indent + "\t")

        lines.append(f"{indent}}}")

    write_joint(0)
    return "\n".join(lines)

def convert_motion_to_bvh(motion_dir: Path, output_file: Path, fps=30.0, scale=100.0):
    root_pos_file = motion_dir / "root_positions.f32"
    rot_file = motion_dir / "local_rotations_xyzw.f32"

    if not root_pos_file.exists() or not rot_file.exists():
        raise SystemExit(f"Missing root_positions.f32 or local_rotations_xyzw.f32 in {motion_dir}")

    pos_bytes = root_pos_file.read_bytes()
    rot_bytes = rot_file.read_bytes()

    num_pos_floats = len(pos_bytes) // 4
    num_frames = num_pos_floats // 3
    num_joints = len(SOMA30_NAMES)

    positions = struct.unpack(f"{num_frames * 3}f", pos_bytes)
    rotations = struct.unpack(f"{num_frames * num_joints * 4}f", rot_bytes)

    hierarchy = build_bvh_hierarchy(SOMA30_NAMES, SOMA30_PARENTS, SOMA30_OFFSETS, scale=scale)

    motion_lines = [
        "MOTION",
        f"Frames: {num_frames}",
        f"Frame Time: {1.0 / fps:.6f}"
    ]

    for f in range(num_frames):
        frame_tokens = []
        # Root translation (scaled to cm)
        rx = positions[f * 3 + 0] * scale
        ry = positions[f * 3 + 1] * scale
        rz = positions[f * 3 + 2] * scale
        frame_tokens.extend([f"{rx:.4f}", f"{ry:.4f}", f"{rz:.4f}"])

        # Rotations for all joints
        for j in range(num_joints):
            base = (f * num_joints + j) * 4
            qx, qy, qz, qw = rotations[base:base+4]
            rz_deg, rx_deg, ry_deg = quat_to_euler_zxy(qx, qy, qz, qw)
            frame_tokens.extend([f"{rz_deg:.4f}", f"{rx_deg:.4f}", f"{ry_deg:.4f}"])

        motion_lines.append(" ".join(frame_tokens))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    bvh_content = hierarchy + "\n" + "\n".join(motion_lines) + "\n"
    output_file.write_text(bvh_content, encoding="utf-8")
    print(f"Successfully exported {num_frames} frames to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Export Kimodo .f32 motion to BVH for Unreal Engine")
    parser.add_argument("--motion-dir", default="output_motion", help="Directory containing root_positions.f32 and local_rotations_xyzw.f32")
    parser.add_argument("--output", default="output_motion/animation.bvh", help="Output BVH file path")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS (default: 30)")
    parser.add_argument("--scale", type=float, default=100.0, help="Unit scale from meters to cm (default: 100 for Unreal Engine)")
    args = parser.parse_args()

    convert_motion_to_bvh(Path(args.motion_dir), Path(args.output), fps=args.fps, scale=args.scale)

if __name__ == "__main__":
    main()
