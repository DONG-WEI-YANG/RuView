#!/usr/bin/env python3
"""Generate a SMPL-style body mesh as a GLB file for the dashboard.

Creates a rigged human body mesh with:
- Smooth body contours from elliptical cross-sections
- 24-bone skeleton matching the project's joint format
- Vertex skin weights for pose-driven deformation
- UV coordinates for texture mapping

Usage:
    pip install numpy trimesh pygltflib
    python tools/generate_body_glb.py

Output:
    dashboard/models/body.glb
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

# ─── 24-joint T-pose positions (meters, Y-up) ────────────────
REST_POSE = np.array([
    [ 0.000, 1.700, 0.000],  #  0 head
    [ 0.000, 1.550, 0.000],  #  1 neck
    [ 0.000, 1.380, 0.000],  #  2 chest
    [ 0.000, 1.120, 0.000],  #  3 spine
    [-0.200, 1.400, 0.000],  #  4 L shoulder
    [-0.480, 1.400, 0.000],  #  5 L elbow
    [-0.700, 1.400, 0.000],  #  6 L wrist
    [ 0.200, 1.400, 0.000],  #  7 R shoulder
    [ 0.480, 1.400, 0.000],  #  8 R elbow
    [ 0.700, 1.400, 0.000],  #  9 R wrist
    [ 0.000, 0.950, 0.000],  # 10 pelvis
    [ 0.000, 0.900, 0.000],  # 11 hip center
    [-0.100, 0.880, 0.000],  # 12 L hip
    [-0.100, 0.500, 0.000],  # 13 L knee
    [-0.100, 0.080, 0.000],  # 14 L ankle
    [ 0.100, 0.880, 0.000],  # 15 R hip
    [ 0.100, 0.500, 0.000],  # 16 R knee
    [ 0.100, 0.080, 0.000],  # 17 R ankle
    [-0.100, 0.030, 0.080],  # 18 L foot
    [ 0.100, 0.030, 0.080],  # 19 R foot
    [-0.780, 1.400, 0.000],  # 20 L hand
    [ 0.780, 1.400, 0.000],  # 21 R hand
    [-0.030, 1.720, 0.060],  # 22 L eye
    [ 0.030, 1.720, 0.060],  # 23 R eye
], dtype=np.float32)

# Bone parent indices (-1 = root)
BONE_PARENTS = [
     1,  # 0: head     <- neck
     2,  # 1: neck     <- chest
     3,  # 2: chest    <- spine
    10,  # 3: spine    <- pelvis
     3,  # 4: L shoulder <- spine
     4,  # 5: L elbow  <- L shoulder
     5,  # 6: L wrist  <- L elbow
     3,  # 7: R shoulder <- spine
     7,  # 8: R elbow  <- R shoulder
     8,  # 9: R wrist  <- R elbow
    11,  # 10: pelvis  <- hip center
    -1,  # 11: hip center (ROOT)
    11,  # 12: L hip   <- hip center
    12,  # 13: L knee  <- L hip
    13,  # 14: L ankle <- L knee
    11,  # 15: R hip   <- hip center
    15,  # 16: R knee  <- R hip
    16,  # 17: R ankle <- R knee
    14,  # 18: L foot  <- L ankle
    17,  # 19: R foot  <- R ankle
     6,  # 20: L hand  <- L wrist
     9,  # 21: R hand  <- R wrist
     0,  # 22: L eye   <- head
     0,  # 23: R eye   <- head
]

# ─── Body chain definitions ──────────────────────────────────
# Each chain: list of (joint_indices, profiles)
# profile: (t_along_chain, radius_x, radius_z)
CHAINS = {
    "torso": {
        "joints": [11, 10, 3, 2, 1],
        "joint_t": [0, 0.18, 0.52, 0.76, 1.0],
        "rings": [
            (0.00, 0.148, 0.105), (0.06, 0.142, 0.100),
            (0.12, 0.132, 0.095), (0.18, 0.120, 0.088),
            (0.25, 0.108, 0.078), (0.32, 0.112, 0.082),
            (0.40, 0.128, 0.090), (0.48, 0.142, 0.098),
            (0.52, 0.150, 0.100), (0.58, 0.158, 0.108),
            (0.64, 0.162, 0.110), (0.70, 0.160, 0.108),
            (0.76, 0.155, 0.100), (0.80, 0.138, 0.088),
            (0.83, 0.110, 0.072), (0.86, 0.078, 0.055),
            (0.90, 0.054, 0.044), (0.95, 0.048, 0.040),
            (1.00, 0.044, 0.038),
        ],
    },
    "l_arm": {
        "joints": [4, 5, 6],
        "joint_t": [0, 0.5, 1.0],
        "rings": [
            (0.00, 0.052, 0.048), (0.15, 0.050, 0.046),
            (0.30, 0.046, 0.042), (0.50, 0.040, 0.036),
            (0.65, 0.035, 0.031), (0.85, 0.030, 0.027),
            (1.00, 0.025, 0.021),
        ],
    },
    "r_arm": {
        "joints": [7, 8, 9],
        "joint_t": [0, 0.5, 1.0],
        "rings": [
            (0.00, 0.052, 0.048), (0.15, 0.050, 0.046),
            (0.30, 0.046, 0.042), (0.50, 0.040, 0.036),
            (0.65, 0.035, 0.031), (0.85, 0.030, 0.027),
            (1.00, 0.025, 0.021),
        ],
    },
    "l_leg": {
        "joints": [12, 13, 14],
        "joint_t": [0, 0.52, 1.0],
        "rings": [
            (0.00, 0.072, 0.068), (0.10, 0.074, 0.070),
            (0.25, 0.070, 0.064), (0.40, 0.060, 0.054),
            (0.52, 0.048, 0.044), (0.65, 0.044, 0.040),
            (0.80, 0.040, 0.036), (1.00, 0.034, 0.030),
        ],
    },
    "r_leg": {
        "joints": [15, 16, 17],
        "joint_t": [0, 0.52, 1.0],
        "rings": [
            (0.00, 0.072, 0.068), (0.10, 0.074, 0.070),
            (0.25, 0.070, 0.064), (0.40, 0.060, 0.054),
            (0.52, 0.048, 0.044), (0.65, 0.044, 0.040),
            (0.80, 0.040, 0.036), (1.00, 0.034, 0.030),
        ],
    },
}

RAD_SEGS = 20  # circumference resolution


def _chain_pos(joints: np.ndarray, joint_t: list[float], t: float) -> np.ndarray:
    """Interpolate position along a joint chain at parameter t in [0,1]."""
    t = np.clip(t, 0, 1)
    for i in range(len(joint_t) - 1):
        if t <= joint_t[i + 1] + 1e-6:
            s = joint_t[i + 1] - joint_t[i]
            lt = (t - joint_t[i]) / s if s > 1e-6 else 0
            lt = np.clip(lt, 0, 1)
            return joints[i] * (1 - lt) + joints[i + 1] * lt
    return joints[-1]


def _chain_tangent(joints: np.ndarray, joint_t: list[float], t: float) -> np.ndarray:
    eps = 0.008
    a = _chain_pos(joints, joint_t, t - eps)
    b = _chain_pos(joints, joint_t, t + eps)
    d = b - a
    n = np.linalg.norm(d)
    return d / n if n > 1e-7 else np.array([0, 1, 0], dtype=np.float32)


def _local_frame(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0, 0, 1], dtype=np.float32)
    bn = np.cross(tangent, up)
    if np.linalg.norm(bn) < 0.01:
        bn = np.cross(tangent, np.array([1, 0, 0], dtype=np.float32))
    bn = bn / (np.linalg.norm(bn) + 1e-8)
    nm = np.cross(bn, tangent)
    nm = nm / (np.linalg.norm(nm) + 1e-8)
    return bn, nm


def _find_nearest_bones(
    pos: np.ndarray, joint_indices: list[int], joint_t: list[float], t: float
) -> tuple[list[int], list[float]]:
    """Find the two nearest bones and their weights for a vertex at parameter t."""
    # Find which segment t falls in
    for i in range(len(joint_t) - 1):
        if t <= joint_t[i + 1] + 1e-6:
            s = joint_t[i + 1] - joint_t[i]
            lt = (t - joint_t[i]) / s if s > 1e-6 else 0
            lt = np.clip(lt, 0, 1)
            b0 = joint_indices[i]
            b1 = joint_indices[i + 1]
            return [b0, b1], [1 - lt, lt]
    last = joint_indices[-1]
    return [last, last], [1.0, 0.0]


def build_chain_mesh(chain_def: dict) -> dict:
    """Build vertices, faces, UVs, and skin data for one body chain."""
    joints_idx = chain_def["joints"]
    joint_t = chain_def["joint_t"]
    rings_def = chain_def["rings"]

    joints = REST_POSE[joints_idx]

    verts = []
    normals = []
    uvs = []
    skin_joints = []  # (v, 4) bone indices
    skin_weights = []  # (v, 4) weights
    faces = []

    n_rings = len(rings_def)
    angles = np.linspace(0, 2 * np.pi, RAD_SEGS + 1)

    for ri, (t, rx, rz) in enumerate(rings_def):
        center = _chain_pos(joints, joint_t, t)
        tangent = _chain_tangent(joints, joint_t, t)
        bn, nm = _local_frame(tangent)

        bones, weights = _find_nearest_bones(center, joints_idx, joint_t, t)

        for i in range(RAD_SEGS + 1):
            c = np.cos(angles[i])
            s = np.sin(angles[i])

            v = center + c * rx * bn + s * rz * nm
            verts.append(v)

            # Ellipse outward normal
            enx = c / (rx if rx > 0.001 else 0.001)
            enz = s / (rz if rz > 0.001 else 0.001)
            el = np.sqrt(enx**2 + enz**2) or 1
            n_vec = (enx * bn + enz * nm) / el
            normals.append(n_vec)

            uvs.append([i / RAD_SEGS, t])

            # Skin (pad to 4 influences)
            sj = bones + [0] * (4 - len(bones))
            sw = weights + [0.0] * (4 - len(weights))
            skin_joints.append(sj[:4])
            skin_weights.append(sw[:4])

    # Faces
    for ri in range(n_rings - 1):
        for i in range(RAD_SEGS):
            a = ri * (RAD_SEGS + 1) + i
            b = a + 1
            c = a + (RAD_SEGS + 1)
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return {
        "vertices": np.array(verts, dtype=np.float32),
        "normals": np.array(normals, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "faces": np.array(faces, dtype=np.uint16),
        "skin_joints": np.array(skin_joints, dtype=np.uint16),
        "skin_weights": np.array(skin_weights, dtype=np.float32),
    }


def build_head_mesh() -> dict:
    """Build an ellipsoidal head mesh."""
    lat_segs, lon_segs = 16, 20
    rx, ry, rz = 0.098, 0.116, 0.090
    center = REST_POSE[0]

    verts, normals, uvs, faces = [], [], [], []
    skin_joints, skin_weights = [], []

    for lat in range(lat_segs + 1):
        theta = np.pi * lat / lat_segs
        for lon in range(lon_segs + 1):
            phi = 2 * np.pi * lon / lon_segs
            x = rx * np.sin(theta) * np.cos(phi)
            y = ry * np.cos(theta)
            z = rz * np.sin(theta) * np.sin(phi)
            verts.append(center + np.array([x, y, z]))

            nx = x / (rx**2) if rx > 0 else 0
            ny = y / (ry**2) if ry > 0 else 0
            nz = z / (rz**2) if rz > 0 else 0
            nl = np.sqrt(nx**2 + ny**2 + nz**2) or 1
            normals.append(np.array([nx / nl, ny / nl, nz / nl]))

            uvs.append([lon / lon_segs, lat / lat_segs])
            skin_joints.append([0, 1, 0, 0])  # head bone + neck
            skin_weights.append([0.9, 0.1, 0.0, 0.0])

    for lat in range(lat_segs):
        for lon in range(lon_segs):
            a = lat * (lon_segs + 1) + lon
            b = a + 1
            c = a + (lon_segs + 1)
            d = c + 1
            faces.append([a, c, b])
            faces.append([b, c, d])

    return {
        "vertices": np.array(verts, dtype=np.float32),
        "normals": np.array(normals, dtype=np.float32),
        "uvs": np.array(uvs, dtype=np.float32),
        "faces": np.array(faces, dtype=np.uint16),
        "skin_joints": np.array(skin_joints, dtype=np.uint16),
        "skin_weights": np.array(skin_weights, dtype=np.float32),
    }


def merge_meshes(meshes: list[dict]) -> dict:
    """Merge multiple mesh dicts into one."""
    offset = 0
    all_v, all_n, all_uv, all_f = [], [], [], []
    all_sj, all_sw = [], []

    for m in meshes:
        all_v.append(m["vertices"])
        all_n.append(m["normals"])
        all_uv.append(m["uvs"])
        all_f.append(m["faces"] + offset)
        all_sj.append(m["skin_joints"])
        all_sw.append(m["skin_weights"])
        offset += len(m["vertices"])

    return {
        "vertices": np.concatenate(all_v),
        "normals": np.concatenate(all_n),
        "uvs": np.concatenate(all_uv),
        "faces": np.concatenate(all_f),
        "skin_joints": np.concatenate(all_sj),
        "skin_weights": np.concatenate(all_sw),
    }


def write_glb(mesh: dict, output_path: str) -> None:
    """Write mesh + skeleton as a GLB (glTF Binary) file."""
    try:
        from pygltflib import (
            GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView,
            Buffer, Skin, Attributes,
            FLOAT, UNSIGNED_SHORT, SCALAR, VEC2, VEC3, VEC4,
            ELEMENT_ARRAY_BUFFER, ARRAY_BUFFER,
        )
    except ImportError:
        print("pygltflib not installed. Install with: pip install pygltflib")
        print("Falling back to JSON export...")
        _write_json_fallback(mesh, output_path)
        return

    # Pack binary data
    verts = mesh["vertices"].astype(np.float32).tobytes()
    norms = mesh["normals"].astype(np.float32).tobytes()
    uvs_b = mesh["uvs"].astype(np.float32).tobytes()
    faces = mesh["faces"].astype(np.uint16).tobytes()
    sj = mesh["skin_joints"].astype(np.uint16).tobytes()
    sw = mesh["skin_weights"].astype(np.float32).tobytes()

    # Inverse bind matrices (identity for each joint — rest pose is reference)
    ibm = np.zeros((24, 16), dtype=np.float32)
    for i in range(24):
        mat = np.eye(4, dtype=np.float32)
        mat[0, 3] = -REST_POSE[i, 0]
        mat[1, 3] = -REST_POSE[i, 1]
        mat[2, 3] = -REST_POSE[i, 2]
        ibm[i] = mat.flatten()
    ibm_bytes = ibm.tobytes()

    # Concatenate all binary data
    bin_data = verts + norms + uvs_b + faces + sj + sw + ibm_bytes

    n_verts = len(mesh["vertices"])
    n_faces = len(mesh["faces"])

    # Compute byte offsets
    off_verts = 0
    off_norms = off_verts + len(verts)
    off_uvs = off_norms + len(norms)
    off_faces = off_uvs + len(uvs_b)
    off_sj = off_faces + len(faces)
    off_sw = off_sj + len(sj)
    off_ibm = off_sw + len(sw)

    # Compute min/max for position accessor
    v_min = mesh["vertices"].min(axis=0).tolist()
    v_max = mesh["vertices"].max(axis=0).tolist()

    gltf = GLTF2(
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=[],
        meshes=[],
        accessors=[],
        bufferViews=[],
        buffers=[Buffer(byteLength=len(bin_data))],
        skins=[],
    )

    # Create joint nodes (24 joints)
    joint_node_indices = list(range(24))
    for i in range(24):
        parent = BONE_PARENTS[i]
        pos = REST_POSE[i].tolist()
        if parent >= 0:
            # Local position = world pos - parent world pos
            pos = (REST_POSE[i] - REST_POSE[parent]).tolist()
        node = Node(
            name=f"joint_{i}",
            translation=pos,
            children=[],
        )
        gltf.nodes.append(node)

    # Set up parent-child relationships
    for i in range(24):
        parent = BONE_PARENTS[i]
        if parent >= 0:
            gltf.nodes[parent].children.append(i)

    # Mesh node (index 24)
    mesh_node_idx = 24
    gltf.nodes.append(Node(name="body_mesh", mesh=0, skin=0))
    gltf.scenes[0].nodes = [11, mesh_node_idx]  # root bone + mesh

    # Buffer views
    bv_idx = 0

    def add_bv(offset, length, target=None):
        nonlocal bv_idx
        bv = BufferView(buffer=0, byteOffset=offset, byteLength=length)
        if target:
            bv.target = target
        gltf.bufferViews.append(bv)
        idx = bv_idx
        bv_idx += 1
        return idx

    bv_verts = add_bv(off_verts, len(verts), ARRAY_BUFFER)
    bv_norms = add_bv(off_norms, len(norms), ARRAY_BUFFER)
    bv_uvs = add_bv(off_uvs, len(uvs_b), ARRAY_BUFFER)
    bv_faces = add_bv(off_faces, len(faces), ELEMENT_ARRAY_BUFFER)
    bv_sj = add_bv(off_sj, len(sj), ARRAY_BUFFER)
    bv_sw = add_bv(off_sw, len(sw), ARRAY_BUFFER)
    bv_ibm = add_bv(off_ibm, len(ibm_bytes))

    # Accessors
    acc_idx = 0

    def add_acc(bv, comp_type, count, acc_type, min_v=None, max_v=None):
        nonlocal acc_idx
        acc = Accessor(
            bufferView=bv,
            componentType=comp_type,
            count=count,
            type=acc_type,
        )
        if min_v is not None:
            acc.min = min_v
        if max_v is not None:
            acc.max = max_v
        gltf.accessors.append(acc)
        idx = acc_idx
        acc_idx += 1
        return idx

    acc_verts = add_acc(bv_verts, FLOAT, n_verts, VEC3, v_min, v_max)
    acc_norms = add_acc(bv_norms, FLOAT, n_verts, VEC3)
    acc_uvs = add_acc(bv_uvs, FLOAT, n_verts, VEC2)
    acc_faces = add_acc(bv_faces, UNSIGNED_SHORT, n_faces * 3, SCALAR)
    acc_sj = add_acc(bv_sj, UNSIGNED_SHORT, n_verts, VEC4)
    acc_sw = add_acc(bv_sw, FLOAT, n_verts, VEC4)
    acc_ibm = add_acc(bv_ibm, FLOAT, 24, "MAT4")

    # Mesh primitive
    prim = Primitive(
        attributes=Attributes(
            POSITION=acc_verts,
            NORMAL=acc_norms,
            TEXCOORD_0=acc_uvs,
            JOINTS_0=acc_sj,
            WEIGHTS_0=acc_sw,
        ),
        indices=acc_faces,
    )
    gltf.meshes.append(Mesh(primitives=[prim]))

    # Skin
    gltf.skins.append(Skin(
        joints=joint_node_indices,
        skeleton=11,  # root joint
        inverseBindMatrices=acc_ibm,
    ))

    # Set binary blob
    gltf.set_binary_blob(bin_data)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    gltf.save(output_path)
    print(f"GLB saved: {output_path} ({len(bin_data)} bytes binary)")


def _write_json_fallback(mesh: dict, output_path: str) -> None:
    """Fallback: write mesh data as JSON for loading in Three.js."""
    json_path = output_path.replace(".glb", ".json")
    data = {
        "vertices": mesh["vertices"].tolist(),
        "normals": mesh["normals"].tolist(),
        "uvs": mesh["uvs"].tolist(),
        "faces": mesh["faces"].tolist(),
        "skinJoints": mesh["skin_joints"].tolist(),
        "skinWeights": mesh["skin_weights"].tolist(),
        "restPose": REST_POSE.tolist(),
        "boneParents": BONE_PARENTS,
    }
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"JSON fallback saved: {json_path}")


def main() -> None:
    print("Building body mesh chains...")
    meshes = []
    for name, chain_def in CHAINS.items():
        m = build_chain_mesh(chain_def)
        print(f"  {name}: {len(m['vertices'])} verts, {len(m['faces'])} faces")
        meshes.append(m)

    print("Building head mesh...")
    head = build_head_mesh()
    print(f"  head: {len(head['vertices'])} verts, {len(head['faces'])} faces")
    meshes.append(head)

    print("Merging meshes...")
    merged = merge_meshes(meshes)
    print(f"  total: {len(merged['vertices'])} verts, {len(merged['faces'])} faces")

    output = str(Path(__file__).parent.parent / "dashboard" / "models" / "body.glb")
    write_glb(merged, output)
    print("Done!")


if __name__ == "__main__":
    main()
