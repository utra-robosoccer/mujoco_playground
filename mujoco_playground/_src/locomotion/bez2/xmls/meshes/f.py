#!/usr/bin/env python3
"""
Approximate an STL mesh with a MuJoCo primitive geom and emit an MJCF file.

Supported primitives:
  - sphere
  - box
  - capsule
  - cylinder
  - plane   (simple: using XY extents, flat in z)
  - mesh    (just wraps the STL as a MuJoCo mesh asset)

MuJoCo notes (from the docs you quoted):
  - PLANE, HFIELD, SPHERE, CAPSULE, BOX, MESH are fully implemented.
  - ELLIPSOID, CYLINDER are implemented but only collide with primitives.
  - BOX is implemented as a mesh internally.

Dependencies:
  pip install numpy trimesh

Usage example:
  python mesh_to_mjcf_primitive.py \
      input.stl \
      --primitive capsule \
      --model-name mymodel \
      --body-name mybody \
      --geom-name mygeom \
      --scale 0.001 \
      --output primitive_from_mesh.xml
"""

import argparse
import pathlib
from textwrap import dedent

import numpy as np
import trimesh


def load_mesh(path, scale=1.0):
    mesh = trimesh.load_mesh(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    if scale != 1.0:
        mesh.apply_scale(scale)
    return mesh


def fit_box(mesh: trimesh.Trimesh):
    """Axis-aligned bounding box -> half-extents."""
    extents = mesh.extents  # [Lx, Ly, Lz]
    half_extents = extents * 0.5
    center = mesh.bounds.mean(axis=0)
    return half_extents, center


def fit_sphere(mesh: trimesh.Trimesh):
    """Bounding sphere centered at vertex mean."""
    center = mesh.vertices.mean(axis=0)
    radius = np.linalg.norm(mesh.vertices - center, axis=1).max()
    return radius, center


def principal_axis(mesh: trimesh.Trimesh):
    """Main principal axis (unit vector) and center."""
    verts = mesh.vertices
    center = verts.mean(axis=0)
    v = verts - center
    cov = np.cov(v.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    return axis, center


def fit_capsule_like(mesh: trimesh.Trimesh):
    """
    Fit a capsule/cylinder:

    - Axis: principal axis
    - From-to: min/max projection along axis
    - Radius: max distance from axis
    """
    axis, center = principal_axis(mesh)
    v = mesh.vertices - center
    proj = v @ axis  # scalar projection along axis

    t_min = proj.min()
    t_max = proj.max()
    mid_t = 0.5 * (t_min + t_max)

    # Endpoints in 3D (body frame; we keep body at origin, so these are relative)
    p0 = (mid_t - (t_max - t_min) * 0.5) * axis
    p1 = (mid_t + (t_max - t_min) * 0.5) * axis

    # Distance from axis for each vertex
    # component orthogonal to axis: v - (proj * axis)
    orth = v - np.outer(proj, axis)
    radius = np.linalg.norm(orth, axis=1).max()

    return p0, p1, radius


def fit_plane(mesh: trimesh.Trimesh):
    """
    Very simple plane fit:
      - Assume plane is horizontal (normal = +z).
      - Use XY extents for size.
      - z position = min z of mesh.
    """
    bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, _ = bounds[1]

    half_x = 0.5 * (xmax - xmin)
    half_y = 0.5 * (ymax - ymin)
    pos = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax), zmin])

    # MuJoCo plane size = [halfwidth_x, halfwidth_y, unused]
    size = np.array([half_x, half_y, 0.0])
    return size, pos


def format_vec(v):
    return " ".join(f"{x:.8g}" for x in v)


def generate_mjcf(
    primitive: str,
    mesh_path: pathlib.Path,
    model_name: str = "mesh_to_primitive",
    body_name: str = "body0",
    geom_name: str = "geom0",
    scale: float = 1.0,
):
    mesh_rel = mesh_path.name  # assume MJCF sits next to STL
    mesh = load_mesh(mesh_path, scale=scale)

    # Compute primitive parameters
    geom_tag = ""
    body_pos = np.zeros(3)

    if primitive == "box":
        half_extents, center = fit_box(mesh)
        body_pos = center
        geom_tag = (
            f'<geom name="{geom_name}" type="box" '
            f'size="{format_vec(half_extents)}"/>'
        )

    elif primitive == "sphere":
        radius, center = fit_sphere(mesh)
        body_pos = center
        geom_tag = (
            f'<geom name="{geom_name}" type="sphere" '
            f'size="{radius:.8g}"/>'
        )

    elif primitive in ("capsule", "cylinder"):
        p0, p1, radius = fit_capsule_like(mesh)
        # Keep body at origin; fromto is expressed in body frame.
        fromto = np.concatenate([p0, p1])
        geom_tag = (
            f'<geom name="{geom_name}" type="{primitive}" '
            f'fromto="{format_vec(fromto)}" '
            f'size="{radius:.8g}"/>'
        )

    elif primitive == "plane":
        size, pos = fit_plane(mesh)
        body_pos = pos
        geom_tag = (
            f'<geom name="{geom_name}" type="plane" '
            f'size="{format_vec(size)}"/>'
        )

    elif primitive == "mesh":
        # Just wrap as mesh geom; we still fit center for nicer default pose.
        _, center = fit_box(mesh)
        body_pos = center
        geom_tag = (
            f'<geom name="{geom_name}" type="mesh" mesh="{geom_name}_mesh"/>'
        )
    else:
        raise ValueError(f"Unsupported primitive: {primitive}")

    # Optional: asset mesh only when needed
    if primitive == "mesh":
        asset_block = (
            f'  <asset>\n'
            f'    <mesh name="{geom_name}_mesh" file="{mesh_rel}" '
            f'scale="{scale:.8g} {scale:.8g} {scale:.8g}"/>\n'
            f'  </asset>\n'
        )
    else:
        asset_block = (
            f'  <asset>\n'
            f'    <!-- Original mesh, if you still want it for visuals -->\n'
            f'    <mesh name="{geom_name}_vismesh" file="{mesh_rel}" '
            f'scale="{scale:.8g} {scale:.8g} {scale:.8g}"/>\n'
            f'  </asset>\n'
        )

    mjcf = f"""\
<mujoco model="{model_name}">
{asset_block.strip()}
  <worldbody>
    <body name="{body_name}" pos="{format_vec(body_pos)}">
      {geom_tag}
    </body>
  </worldbody>
</mujoco>
"""
    return dedent(mjcf)


def main():
    parser = argparse.ArgumentParser(
        description="Approximate an STL mesh with a MuJoCo primitive geom and emit MJCF."
    )
    parser.add_argument("stl", type=pathlib.Path, help="Input STL mesh file")
    parser.add_argument(
        "--primitive",
        "-p",
        type=str.lower,
        choices=["sphere", "box", "capsule", "cylinder", "plane", "mesh"],
        default="box",
        help="Primitive geom type to fit (default: box)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale applied to STL (e.g., 0.001 for mm -> m)",
    )
    parser.add_argument(
        "--model-name", type=str, default="mesh_to_primitive", help="MJCF model name"
    )
    parser.add_argument(
        "--body-name", type=str, default="body0", help="Name of worldbody child body"
    )
    parser.add_argument(
        "--geom-name", type=str, default="geom0", help="Name of the geom"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        default=None,
        help="Output MJCF file (default: <stl_basename>_<primitive>.xml)",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.stl.with_suffix(f".{args.primitive}.xml")

    mjcf_xml = generate_mjcf(
        primitive=args.primitive,
        mesh_path=args.stl,
        model_name=args.model_name,
        body_name=args.body_name,
        geom_name=args.geom_name,
        scale=args.scale,
    )

    args.output.write_text(mjcf_xml)
    print(f"Wrote MJCF to {args.output}")


if __name__ == "__main__":
    main()
