#!/usr/bin/env python3
"""Convert Kimodo raw motion float buffers (.f32) to standard binary glTF (.glb) with Skinned Mesh for Unreal Engine / Blender."""

import argparse
import json
import struct
from pathlib import Path

# Skeletons specifications
SKELETONS = {
    "soma30": {
        "names": [
            "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
            "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm",
            "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin", "LeftFoot",
            "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"
        ],
        "parents": [-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28],
        "offsets": [
            [0.0, 0.988, 0.0], [-0.00013727, 0.0500376256, -0.00053726669], [-1.86574103e-09, 0.0712530139, -0.000298248546],
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
    }
}

def compute_global_rest_positions(parents, offsets):
    """Compute rest global position for each joint."""
    global_pos = []
    for i, (parent, offset) in enumerate(zip(parents, offsets)):
        if parent == -1:
            global_pos.append(list(offset))
        else:
            p_pos = global_pos[parent]
            global_pos.append([p_pos[0] + offset[0], p_pos[1] + offset[1], p_pos[2] + offset[2]])
    return global_pos

def create_bone_mesh(global_pos, parents):
    """Create a minimal skinned mesh (a bone pyramid for each bone)."""
    vertices = []
    joints = []
    weights = []
    indices = []

    def add_cube(center, joint_idx, size=0.015):
        base_idx = len(vertices)
        cx, cy, cz = center
        # 8 corners of a cube
        for dx in (-size, size):
            for dy in (-size, size):
                for dz in (-size, size):
                    vertices.append([cx + dx, cy + dy, cz + dz])
                    joints.append([joint_idx, 0, 0, 0])
                    weights.append([1.0, 0.0, 0.0, 0.0])
        # 6 quad faces (12 triangles)
        cube_faces = [
            (0,1,3),(0,3,2), (4,6,7),(4,7,5), # front/back
            (0,4,5),(0,5,1), (2,3,7),(2,7,6), # bottom/top
            (0,2,6),(0,6,4), (1,5,7),(1,7,3)  # left/right
        ]
        for f in cube_faces:
            indices.extend([base_idx + f[0], base_idx + f[1], base_idx + f[2]])

    for i in range(len(global_pos)):
        add_cube(global_pos[i], i)

    return vertices, joints, weights, indices

def convert_motion_to_glb(motion_dir: Path, output_file: Path, skeleton_key="soma30", fps=30.0):
    root_pos_file = motion_dir / "root_positions.f32"
    rot_file = motion_dir / "local_rotations_xyzw.f32"

    if not root_pos_file.exists() or not rot_file.exists():
        raise SystemExit(f"Missing root_positions.f32 or local_rotations_xyzw.f32 in {motion_dir}")

    pos_bytes = root_pos_file.read_bytes()
    rot_bytes = rot_file.read_bytes()

    num_pos_floats = len(pos_bytes) // 4
    num_frames = num_pos_floats // 3

    skel = SKELETONS[skeleton_key]
    names = skel["names"]
    parents = skel["parents"]
    offsets = skel["offsets"]
    num_joints = len(names)

    times = [i / fps for i in range(num_frames)]
    roots = struct.unpack(f"{num_frames * 3}f", pos_bytes)
    rotations = struct.unpack(f"{num_frames * num_joints * 4}f", rot_bytes)

    global_rest = compute_global_rest_positions(parents, offsets)
    mesh_verts, mesh_joints, mesh_weights, mesh_indices = create_bone_mesh(global_rest, parents)

    # Compute Inverse Bind Matrices (Column-major 4x4 matrix: identity with -pos in translation)
    inv_bind_matrices = []
    for pos in global_rest:
        mat = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -pos[0], -pos[1], -pos[2], 1.0
        ]
        inv_bind_matrices.extend(mat)

    bin_data = bytearray()
    views = []

    def add_raw_view(data_bytes):
        while len(bin_data) % 4 != 0:
            bin_data.append(0)
        offset = len(bin_data)
        bin_data.extend(data_bytes)
        views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data_bytes)
        })
        return len(views) - 1

    # Pack mesh data
    idx_bytes = struct.pack(f"<{len(mesh_indices)}H", *mesh_indices)
    v_flat = [coord for pt in mesh_verts for coord in pt]
    pos_mesh_bytes = struct.pack(f"<{len(v_flat)}f", *v_flat)
    j_flat = [j for jnt in mesh_joints for j in jnt]
    jnt_bytes = struct.pack(f"<{len(j_flat)}H", *j_flat)
    w_flat = [w for wt in mesh_weights for w in wt]
    wt_bytes = struct.pack(f"<{len(w_flat)}f", *w_flat)

    # Pack Inverse Bind Matrices
    ibm_bytes = struct.pack(f"<{len(inv_bind_matrices)}f", *inv_bind_matrices)

    # Pack Animation data
    time_bytes = struct.pack(f"<{len(times)}f", *times)
    root_anim_bytes = struct.pack(f"<{len(roots)}f", *roots)

    # Add buffer views
    v_idx = add_raw_view(idx_bytes)
    v_pos = add_raw_view(pos_mesh_bytes)
    v_jnt = add_raw_view(jnt_bytes)
    v_wt  = add_raw_view(wt_bytes)
    v_ibm = add_raw_view(ibm_bytes)
    v_time = add_raw_view(time_bytes)
    v_root = add_raw_view(root_anim_bytes)

    rot_anim_views = []
    for j in range(num_joints):
        track = []
        for f in range(num_frames):
            base = (f * num_joints + j) * 4
            track.extend(rotations[base:base+4])
        track_bytes = struct.pack(f"<{len(track)}f", *track)
        rot_anim_views.append(add_raw_view(track_bytes))

    # Min/max bounds for mesh positions
    min_x = min(p[0] for p in mesh_verts)
    max_x = max(p[0] for p in mesh_verts)
    min_y = min(p[1] for p in mesh_verts)
    max_y = max(p[1] for p in mesh_verts)
    min_z = min(p[2] for p in mesh_verts)
    max_z = max(p[2] for p in mesh_verts)

    accessors = [
        # 0: Indices
        {"bufferView": v_idx, "componentType": 5123, "count": len(mesh_indices), "type": "SCALAR"},
        # 1: Mesh Positions
        {"bufferView": v_pos, "componentType": 5126, "count": len(mesh_verts), "type": "VEC3", "min": [min_x, min_y, min_z], "max": [max_x, max_y, max_z]},
        # 2: Mesh Joints
        {"bufferView": v_jnt, "componentType": 5123, "count": len(mesh_joints), "type": "VEC4"},
        # 3: Mesh Weights
        {"bufferView": v_wt, "componentType": 5126, "count": len(mesh_weights), "type": "VEC4"},
        # 4: Inverse Bind Matrices
        {"bufferView": v_ibm, "componentType": 5126, "count": num_joints, "type": "MAT4"},
        # 5: Time
        {"bufferView": v_time, "componentType": 5126, "count": num_frames, "type": "SCALAR", "min": [times[0]], "max": [times[-1]]},
        # 6: Root Translation
        {"bufferView": v_root, "componentType": 5126, "count": num_frames, "type": "VEC3"},
    ]

    # Add rotation accessors
    for v in rot_anim_views:
        accessors.append({"bufferView": v, "componentType": 5126, "count": num_frames, "type": "VEC4"})

    # Build node hierarchy
    # Node 0..29 are skeleton joints
    # Node 30 is the Skinned Mesh Node
    nodes = []
    for j in range(num_joints):
        node = {"name": names[j], "translation": offsets[j]}
        children = [child for child, parent in enumerate(parents) if parent == j]
        if children:
            node["children"] = children
        nodes.append(node)

    # Add Skinned Mesh Node
    mesh_node_idx = num_joints
    nodes.append({
        "name": "KimodoSkinnedMesh",
        "mesh": 0,
        "skin": 0
    })

    # Animation Samplers & Channels
    samplers = []
    channels = []

    def add_anim_channel(target_node, output_acc, path):
        samplers.append({"input": 5, "output": output_acc, "interpolation": "LINEAR"})
        channels.append({
            "sampler": len(samplers) - 1,
            "target": {"node": target_node, "path": path}
        })

    add_anim_channel(0, 6, "translation")
    for j in range(num_joints):
        add_anim_channel(j, 7 + j, "rotation")

    gltf = {
        "asset": {"version": "2.0", "generator": "kimodo.cpp Skinned GLB Exporter for Unreal Engine"},
        "scene": 0,
        "scenes": [{"nodes": [0, mesh_node_idx]}],
        "nodes": nodes,
        "meshes": [{
            "name": "KimodoMesh",
            "primitives": [{
                "attributes": {
                    "POSITION": 1,
                    "JOINTS_0": 2,
                    "WEIGHTS_0": 3
                },
                "indices": 0
            }]
        }],
        "skins": [{
            "name": "KimodoSkin",
            "inverseBindMatrices": 4,
            "joints": list(range(num_joints)),
            "skeleton": 0
        }],
        "buffers": [{"byteLength": len(bin_data)}],
        "bufferViews": views,
        "accessors": accessors,
        "animations": [{
            "name": "KimodoMotion",
            "samplers": samplers,
            "channels": channels
        }]
    }

    json_str = json.dumps(gltf, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '
    while len(bin_data) % 4 != 0:
        bin_data += b'\x00'

    # GLB Header
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    glb = bytearray()
    glb.extend(struct.pack("<4sII", b"glTF", 2, total_length))

    # JSON Chunk
    glb.extend(struct.pack("<I4s", len(json_bytes), b"JSON"))
    glb.extend(json_bytes)

    # BIN Chunk
    glb.extend(struct.pack("<I4s", len(bin_data), b"BIN\x00"))
    glb.extend(bin_data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(glb)
    print(f"Successfully exported Skinned GLB ({num_frames} frames, {num_joints} joints) to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Export Kimodo motion to UE-compliant Skinned GLB")
    parser.add_argument("--motion-dir", default="output_motion", help="Directory containing root_positions.f32 and local_rotations_xyzw.f32")
    parser.add_argument("--output", default="output_motion/animation.glb", help="Output GLB file path")
    parser.add_argument("--model", default="soma30", choices=["soma30"], help="Skeleton type (default: soma30)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS (default: 30)")
    args = parser.parse_args()

    convert_motion_to_glb(Path(args.motion_dir), Path(args.output), skeleton_key=args.model, fps=args.fps)

if __name__ == "__main__":
    main()
