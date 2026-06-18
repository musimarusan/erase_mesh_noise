#!/usr/bin/env python3
"""
erase_mesh_noise.py

Proposed by: GPT-5.5 Thinking

Scaniverse 由来 OBJ メッシュの孤立小成分および下側偽面を除去する実験用スクリプト。

仕様:
    - 入力ファイルはコマンドライン引数で指定する
    - 出力ファイルは入力ファイルと同じ階層に作成する
    - 出力ファイル名は basename に _erase を付ける

例:
    input.obj       -> input_erase.obj
    data/test.obj   -> data/test_erase.obj

主な処理:
    1. 座標許容差つき virtual vertex id を作る
    2. virtual vertex id 上の共有辺で連結成分を作る
    3. 小さい連結成分を除去する
    4. 任意で --top-visible により、XYグリッド上の上方可視面を残す

方針:
    - trimesh.split() は使わない
    - Z 値だけで削除しない
    - 凹部を単純にノイズ扱いしない
    - 本体とつながった下側偽面は --top-visible で実験的に削る

注意:
    - --top-visible は実験モード
    - ピット側壁・急斜面・オーバーハング状の部分を削る可能性がある
    - 最初は必ず --dry-run で削除率を確認すること
    - テクスチャ・MTL の完全保持は保証しない
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class ComponentInfo:
    component_id: int
    face_count: int
    original_vertex_count: int
    virtual_vertex_count: int
    area: float
    z_min: float
    z_max: float
    keep: bool


@dataclass
class TopVisibleSummary:
    enabled: bool
    cell_size: float
    z_tolerance: float
    sample_mode: str
    candidate_faces: int
    visible_faces: int
    removed_faces: int
    removed_ratio_in_candidates: float


class UnionFind:
    """
    Union-Find / Disjoint Set Union.

    面の連結成分を軽量に求めるために使う。
    """

    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent

        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])

        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]

        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def make_output_path(input_path: Path) -> Path:
    """
    入力ファイルと同じ階層に、basename + '_erase' の出力ファイル名を作る。

    例:
        /data/sample.obj -> /data/sample_erase.obj
    """
    return input_path.with_name(f"{input_path.stem}_erase{input_path.suffix}")


def load_obj_as_mesh(path: Path) -> trimesh.Trimesh:
    """
    OBJファイルを trimesh.Trimesh として読み込む。

    Scene として読まれた場合は、形状処理を優先して単一 Trimesh に結合する。
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")

    loaded = trimesh.load_mesh(path, process=False)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("Loaded scene has no geometry.")

        geometries = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]

        if not geometries:
            raise ValueError("Scene contains no Trimesh geometry.")

        return trimesh.util.concatenate(geometries)

    raise TypeError(f"Unsupported loaded object type: {type(loaded)}")


def build_virtual_vertex_ids(
    vertices: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    """
    座標許容差に基づいて virtual vertex id を作る。

    OBJ / trimesh では、同じ空間位置でも UV・法線・マテリアル境界などの都合で
    頂点番号が分裂することがある。

    そのため、raw vertex index ではなく、座標を tolerance で丸めた整数格子上で
    同一座標とみなせる頂点を同じ virtual vertex id にまとめる。
    """
    if tolerance <= 0.0:
        raise ValueError("vertex tolerance must be positive.")

    quantized = np.rint(vertices / tolerance).astype(np.int64)

    _, inverse = np.unique(
        quantized,
        axis=0,
        return_inverse=True,
    )

    virtual_vertex_ids = inverse.astype(np.int32, copy=False)
    virtual_vertex_count = int(virtual_vertex_ids.max()) + 1

    return virtual_vertex_ids, virtual_vertex_count


def find_face_components_by_virtual_edges(
    faces: np.ndarray,
    virtual_vertex_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    三角形面の連結成分を求める。

    連結条件:
        2つの面が virtual vertex id 上で同じ辺を共有していれば同じ成分。
    """
    face_count = len(faces)

    if face_count == 0:
        raise ValueError("Mesh has no faces.")

    union_find = UnionFind(face_count)
    virtual_faces = virtual_vertex_ids[faces]

    edge_owner: dict[tuple[int, int], int] = {}
    degenerate_face_count = 0

    for face_index, face in enumerate(virtual_faces):
        a = int(face[0])
        b = int(face[1])
        c = int(face[2])

        if a == b or b == c or c == a:
            degenerate_face_count += 1

        raw_edges = (
            (a, b),
            (b, c),
            (c, a),
        )

        for v0, v1 in raw_edges:
            if v0 == v1:
                continue

            edge = (v0, v1) if v0 < v1 else (v1, v0)

            previous_face_index = edge_owner.get(edge)

            if previous_face_index is None:
                edge_owner[edge] = face_index
            else:
                union_find.union(face_index, previous_face_index)

    roots = np.empty(face_count, dtype=np.int32)

    for i in range(face_count):
        roots[i] = union_find.find(i)

    _, face_component_ids = np.unique(roots, return_inverse=True)
    face_component_ids = face_component_ids.astype(np.int32, copy=False)

    component_face_counts = np.bincount(face_component_ids)

    if degenerate_face_count > 0:
        print(
            f"warning: degenerate faces after vertex quantization: "
            f"{degenerate_face_count}",
            flush=True,
        )

    return face_component_ids, component_face_counts


def decide_keep_components(
    component_face_counts: np.ndarray,
    min_faces: int,
    min_ratio: float,
    keep_largest_only: bool,
) -> np.ndarray:
    """
    残す連結成分を決める。

    方針:
        - 最大成分は必ず残す
        - keep_largest_only=True なら最大成分だけ残す
        - それ以外は face 数の閾値で判定する

    閾値:
        max(min_faces, 最大成分 face 数 * min_ratio)
    """
    if len(component_face_counts) == 0:
        raise ValueError("No components found.")

    if min_faces < 1:
        raise ValueError("min_faces must be >= 1.")

    if min_ratio < 0.0:
        raise ValueError("min_ratio must be >= 0.0.")

    largest_component_id = int(np.argmax(component_face_counts))
    largest_face_count = int(component_face_counts[largest_component_id])

    keep_flags = np.zeros(len(component_face_counts), dtype=bool)

    if keep_largest_only:
        keep_flags[largest_component_id] = True
        return keep_flags

    threshold = max(min_faces, int(np.ceil(largest_face_count * min_ratio)))

    keep_flags = component_face_counts >= threshold
    keep_flags[largest_component_id] = True

    return keep_flags


def estimate_median_edge_length(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    max_sample_faces: int = 50000,
) -> float:
    """
    active face の代表的な辺長を推定する。

    --top-cell-size や --top-z-tolerance が未指定の場合の自動値に使う。
    """
    face_indices = np.flatnonzero(face_mask)

    if len(face_indices) == 0:
        raise ValueError("No active faces for edge length estimation.")

    if len(face_indices) > max_sample_faces:
        sample_positions = np.linspace(
            0,
            len(face_indices) - 1,
            max_sample_faces,
            dtype=np.int64,
        )
        face_indices = face_indices[sample_positions]

    sampled_faces = faces[face_indices]

    p0 = vertices[sampled_faces[:, 0]]
    p1 = vertices[sampled_faces[:, 1]]
    p2 = vertices[sampled_faces[:, 2]]

    lengths = np.concatenate(
        [
            np.linalg.norm(p1 - p0, axis=1),
            np.linalg.norm(p2 - p1, axis=1),
            np.linalg.norm(p0 - p2, axis=1),
        ]
    )

    positive_lengths = lengths[lengths > 0.0]

    if len(positive_lengths) == 0:
        raise ValueError("Could not estimate positive edge length.")

    return float(np.median(positive_lengths))


def build_top_visible_face_mask(
    mesh: trimesh.Trimesh,
    candidate_face_mask: np.ndarray,
    cell_size: float,
    z_tolerance: float,
    sample_mode: str,
) -> tuple[np.ndarray, TopVisibleSummary]:
    """
    XYグリッド上の上方可視判定で、残す face mask を作る。

    考え方:
        - candidate face からサンプル点を作る
        - XYグリッドセルごとに最大Zを求める
        - 各faceのサンプル点が、そのセルの最大Zから z_tolerance 以内なら残す
        - どのサンプル点も上方可視でないfaceは削除候補にする

    sample_mode:
        centroid:
            face centroid だけを見る。攻撃的。
        vertices:
            face の3頂点を見る。保守的。
        vertices-centroid:
            3頂点 + centroid を見る。初期値。
    """
    if cell_size <= 0.0:
        raise ValueError("top-visible cell_size must be positive.")

    if z_tolerance < 0.0:
        raise ValueError("top-visible z_tolerance must be >= 0.")

    if sample_mode not in {"centroid", "vertices", "vertices-centroid"}:
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")

    candidate_face_indices = np.flatnonzero(candidate_face_mask)
    face_count = len(mesh.faces)

    if len(candidate_face_indices) == 0:
        empty_mask = np.zeros(face_count, dtype=bool)
        summary = TopVisibleSummary(
            enabled=True,
            cell_size=cell_size,
            z_tolerance=z_tolerance,
            sample_mode=sample_mode,
            candidate_faces=0,
            visible_faces=0,
            removed_faces=0,
            removed_ratio_in_candidates=0.0,
        )
        return empty_mask, summary

    faces = mesh.faces[candidate_face_indices]
    vertices = mesh.vertices

    sample_points_list: list[np.ndarray] = []
    sample_face_indices_list: list[np.ndarray] = []

    if sample_mode in {"vertices", "vertices-centroid"}:
        vertex_sample_points = vertices[faces.reshape(-1)]
        vertex_sample_face_indices = np.repeat(candidate_face_indices, 3)

        sample_points_list.append(vertex_sample_points)
        sample_face_indices_list.append(vertex_sample_face_indices)

    if sample_mode in {"centroid", "vertices-centroid"}:
        face_vertices = vertices[faces]
        centroid_sample_points = face_vertices.mean(axis=1)
        centroid_sample_face_indices = candidate_face_indices.copy()

        sample_points_list.append(centroid_sample_points)
        sample_face_indices_list.append(centroid_sample_face_indices)

    sample_points = np.concatenate(sample_points_list, axis=0)
    sample_face_indices = np.concatenate(sample_face_indices_list, axis=0)

    bounds = mesh.bounds
    x0 = float(bounds[0, 0])
    y0 = float(bounds[0, 1])

    ix = np.floor((sample_points[:, 0] - x0) / cell_size).astype(np.int64)
    iy = np.floor((sample_points[:, 1] - y0) / cell_size).astype(np.int64)

    if np.any(ix < 0) or np.any(iy < 0):
        raise ValueError("Negative grid index detected. Check mesh bounds.")

    iy_span = int(iy.max()) + 1
    keys = ix * (iy_span + 1) + iy

    _, inverse = np.unique(keys, return_inverse=True)

    max_z_by_cell = np.full(int(inverse.max()) + 1, -np.inf, dtype=float)
    np.maximum.at(max_z_by_cell, inverse, sample_points[:, 2])

    visible_sample_mask = (
        sample_points[:, 2] >= max_z_by_cell[inverse] - z_tolerance
    )

    top_visible_face_mask = np.zeros(face_count, dtype=bool)
    top_visible_face_mask[sample_face_indices[visible_sample_mask]] = True

    visible_faces = int(np.count_nonzero(top_visible_face_mask & candidate_face_mask))
    candidate_faces = int(np.count_nonzero(candidate_face_mask))
    removed_faces = candidate_faces - visible_faces

    removed_ratio = (
        removed_faces / candidate_faces
        if candidate_faces > 0
        else 0.0
    )

    summary = TopVisibleSummary(
        enabled=True,
        cell_size=cell_size,
        z_tolerance=z_tolerance,
        sample_mode=sample_mode,
        candidate_faces=candidate_faces,
        visible_faces=visible_faces,
        removed_faces=removed_faces,
        removed_ratio_in_candidates=removed_ratio,
    )

    return top_visible_face_mask, summary


def build_component_infos(
    mesh: trimesh.Trimesh,
    face_component_ids: np.ndarray,
    component_face_counts: np.ndarray,
    keep_component_flags: np.ndarray,
    virtual_vertex_ids: np.ndarray,
) -> list[ComponentInfo]:
    """
    レポート用の連結成分情報を作る。
    """
    component_count = len(component_face_counts)
    area_faces = mesh.area_faces
    vertices = mesh.vertices
    faces = mesh.faces

    component_areas = np.bincount(
        face_component_ids,
        weights=area_faces,
        minlength=component_count,
    )

    original_vertex_sets: list[set[int]] = [set() for _ in range(component_count)]
    virtual_vertex_sets: list[set[int]] = [set() for _ in range(component_count)]

    z_min_values = np.full(component_count, np.inf, dtype=float)
    z_max_values = np.full(component_count, -np.inf, dtype=float)

    for face_index, component_id in enumerate(face_component_ids):
        cid = int(component_id)

        face = faces[face_index]
        v0 = int(face[0])
        v1 = int(face[1])
        v2 = int(face[2])

        original_vertex_sets[cid].add(v0)
        original_vertex_sets[cid].add(v1)
        original_vertex_sets[cid].add(v2)

        virtual_vertex_sets[cid].add(int(virtual_vertex_ids[v0]))
        virtual_vertex_sets[cid].add(int(virtual_vertex_ids[v1]))
        virtual_vertex_sets[cid].add(int(virtual_vertex_ids[v2]))

        z_values = vertices[[v0, v1, v2], 2]

        z_min = float(np.min(z_values))
        z_max = float(np.max(z_values))

        if z_min < z_min_values[cid]:
            z_min_values[cid] = z_min

        if z_max > z_max_values[cid]:
            z_max_values[cid] = z_max

    infos: list[ComponentInfo] = []

    for component_id in range(component_count):
        infos.append(
            ComponentInfo(
                component_id=component_id,
                face_count=int(component_face_counts[component_id]),
                original_vertex_count=len(original_vertex_sets[component_id]),
                virtual_vertex_count=len(virtual_vertex_sets[component_id]),
                area=float(component_areas[component_id]),
                z_min=float(z_min_values[component_id]),
                z_max=float(z_max_values[component_id]),
                keep=bool(keep_component_flags[component_id]),
            )
        )

    return infos


def print_report(
    original_vertex_count: int,
    virtual_vertex_count: int,
    original_face_count: int,
    output_vertex_count: int | None,
    output_face_count: int | None,
    component_infos: list[ComponentInfo],
    final_keep_face_mask: np.ndarray,
    output_path: Path,
    report_limit: int,
    top_visible_summary: TopVisibleSummary | None,
) -> None:
    """
    処理結果を標準出力に表示する。
    """
    component_kept_faces = sum(info.face_count for info in component_infos if info.keep)
    component_removed_faces = original_face_count - component_kept_faces

    kept_components = sum(1 for info in component_infos if info.keep)
    removed_components = len(component_infos) - kept_components

    final_kept_faces = int(np.count_nonzero(final_keep_face_mask))
    final_removed_faces = original_face_count - final_kept_faces

    final_removed_face_ratio = (
        final_removed_faces / original_face_count
        if original_face_count > 0
        else 0.0
    )

    print("=== erase_mesh_noise report ===")
    print(f"input vertices                  : {original_vertex_count}")
    print(f"virtual vertices                : {virtual_vertex_count}")
    print(f"input faces                     : {original_face_count}")
    print(f"components                      : {len(component_infos)}")
    print(f"kept components                 : {kept_components}")
    print(f"removed components              : {removed_components}")
    print(f"component-filter kept faces      : {component_kept_faces}")
    print(f"component-filter removed faces   : {component_removed_faces}")

    if top_visible_summary is not None and top_visible_summary.enabled:
        print()
        print("top-visible filter:")
        print(f"  cell size                     : {top_visible_summary.cell_size}")
        print(f"  z tolerance                   : {top_visible_summary.z_tolerance}")
        print(f"  sample mode                   : {top_visible_summary.sample_mode}")
        print(f"  candidate faces               : {top_visible_summary.candidate_faces}")
        print(f"  visible faces                 : {top_visible_summary.visible_faces}")
        print(f"  removed faces                 : {top_visible_summary.removed_faces}")
        print(
            f"  removed ratio in candidates   : "
            f"{top_visible_summary.removed_ratio_in_candidates:.4%}"
        )

    print()
    print(f"final kept faces                : {final_kept_faces}")
    print(f"final removed faces             : {final_removed_faces}")
    print(f"final removed face ratio        : {final_removed_face_ratio:.4%}")
    print(f"output path                     : {output_path}")
    print()

    sorted_infos = sorted(
        component_infos,
        key=lambda x: x.face_count,
        reverse=True,
    )

    if report_limit > 0:
        display_infos = sorted_infos[:report_limit]
    else:
        display_infos = sorted_infos

    print(
        f"{'cid':>6} "
        f"{'faces':>12} "
        f"{'orig_v':>12} "
        f"{'virt_v':>12} "
        f"{'area':>14} "
        f"{'z_min':>12} "
        f"{'z_max':>12} "
        f"{'decision':>10}"
    )

    for info in display_infos:
        decision = "KEEP" if info.keep else "REMOVE"

        print(
            f"{info.component_id:6d} "
            f"{info.face_count:12d} "
            f"{info.original_vertex_count:12d} "
            f"{info.virtual_vertex_count:12d} "
            f"{info.area:14.6f} "
            f"{info.z_min:12.6f} "
            f"{info.z_max:12.6f} "
            f"{decision:>10}"
        )

    if report_limit > 0 and len(sorted_infos) > report_limit:
        print()
        print(
            f"... {len(sorted_infos) - report_limit} components omitted "
            f"from report. Use --report-limit 0 to show all."
        )

    if output_vertex_count is not None and output_face_count is not None:
        print()
        print(f"output vertices                 : {output_vertex_count}")
        print(f"output faces                    : {output_face_count}")
        print(f"removed faces after save        : {original_face_count - output_face_count}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove small disconnected components and optional lower hidden faces "
            "from a Scaniverse OBJ mesh."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input OBJ file path.",
    )

    parser.add_argument(
        "--min-faces",
        type=int,
        default=10,
        help=(
            "Minimum face count for keeping a component. "
            "Default: 10"
        ),
    )

    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.0,
        help=(
            "Minimum face ratio relative to the largest component. "
            "Default: 0.0"
        ),
    )

    parser.add_argument(
        "--vertex-tolerance",
        type=float,
        default=1.0e-5,
        help=(
            "Tolerance for merging vertices by coordinate. "
            "Default: 1.0e-5"
        ),
    )

    parser.add_argument(
        "--top-visible",
        action="store_true",
        help=(
            "Enable experimental top-visible filter. "
            "This attempts to remove lower hidden faces connected to the main mesh."
        ),
    )

    parser.add_argument(
        "--top-cell-size",
        type=float,
        default=0.0,
        help=(
            "XY grid cell size for top-visible filter. "
            "If <= 0, auto value is median_edge_length * 3. Default: 0.0"
        ),
    )

    parser.add_argument(
        "--top-z-tolerance",
        type=float,
        default=0.0,
        help=(
            "Z tolerance for top-visible filter. "
            "If <= 0, auto value is median_edge_length * 5. Default: 0.0"
        ),
    )

    parser.add_argument(
        "--top-sample-mode",
        choices=["centroid", "vertices", "vertices-centroid"],
        default="vertices-centroid",
        help=(
            "Sampling mode for top-visible filter. "
            "centroid is aggressive, vertices is conservative, "
            "vertices-centroid is default."
        ),
    )

    parser.add_argument(
        "--keep-largest-only",
        action="store_true",
        help=(
            "Keep only the largest connected component. "
            "This is aggressive and may remove valid separated parts."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print component report without writing output OBJ.",
    )

    parser.add_argument(
        "--report-limit",
        type=int,
        default=50,
        help=(
            "Maximum number of components shown in report. "
            "Use 0 to show all. Default: 50"
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path = make_output_path(input_path)

    try:
        print("loading mesh...", flush=True)
        mesh = load_obj_as_mesh(input_path)

        original_vertex_count = len(mesh.vertices)
        original_face_count = len(mesh.faces)

        print(f"loaded vertices: {original_vertex_count}", flush=True)
        print(f"loaded faces   : {original_face_count}", flush=True)

        print(
            "building virtual vertex ids "
            f"(tolerance={args.vertex_tolerance})...",
            flush=True,
        )
        virtual_vertex_ids, virtual_vertex_count = build_virtual_vertex_ids(
            vertices=mesh.vertices,
            tolerance=args.vertex_tolerance,
        )

        print(f"virtual vertices: {virtual_vertex_count}", flush=True)

        print(
            "finding connected components by virtual shared edges...",
            flush=True,
        )
        face_component_ids, component_face_counts = (
            find_face_components_by_virtual_edges(
                faces=mesh.faces,
                virtual_vertex_ids=virtual_vertex_ids,
            )
        )

        keep_component_flags = decide_keep_components(
            component_face_counts=component_face_counts,
            min_faces=args.min_faces,
            min_ratio=args.min_ratio,
            keep_largest_only=args.keep_largest_only,
        )

        component_keep_face_mask = keep_component_flags[face_component_ids]
        final_keep_face_mask = component_keep_face_mask.copy()

        component_infos = build_component_infos(
            mesh=mesh,
            face_component_ids=face_component_ids,
            component_face_counts=component_face_counts,
            keep_component_flags=keep_component_flags,
            virtual_vertex_ids=virtual_vertex_ids,
        )

        top_visible_summary: TopVisibleSummary | None = None

        if args.top_visible:
            print("preparing top-visible filter...", flush=True)

            median_edge_length = estimate_median_edge_length(
                vertices=mesh.vertices,
                faces=mesh.faces,
                face_mask=component_keep_face_mask,
            )

            if args.top_cell_size > 0.0:
                top_cell_size = args.top_cell_size
            else:
                top_cell_size = median_edge_length * 3.0

            if args.top_z_tolerance > 0.0:
                top_z_tolerance = args.top_z_tolerance
            else:
                top_z_tolerance = median_edge_length * 5.0

            print(f"median edge length: {median_edge_length}", flush=True)
            print(f"top cell size     : {top_cell_size}", flush=True)
            print(f"top z tolerance   : {top_z_tolerance}", flush=True)

            top_visible_face_mask, top_visible_summary = (
                build_top_visible_face_mask(
                    mesh=mesh,
                    candidate_face_mask=component_keep_face_mask,
                    cell_size=top_cell_size,
                    z_tolerance=top_z_tolerance,
                    sample_mode=args.top_sample_mode,
                )
            )

            final_keep_face_mask = component_keep_face_mask & top_visible_face_mask

        final_kept_faces = int(np.count_nonzero(final_keep_face_mask))
        final_removed_faces = original_face_count - final_kept_faces

        print("decision summary:", flush=True)
        print(f"  components          : {len(component_face_counts)}", flush=True)
        print(f"  component-kept faces: {int(np.count_nonzero(component_keep_face_mask))}", flush=True)
        print(f"  final kept faces    : {final_kept_faces}", flush=True)
        print(f"  final removed faces : {final_removed_faces}", flush=True)
        print(
            f"  final removed ratio : "
            f"{final_removed_faces / original_face_count:.4%}",
            flush=True,
        )

        if args.dry_run:
            print_report(
                original_vertex_count=original_vertex_count,
                virtual_vertex_count=virtual_vertex_count,
                original_face_count=original_face_count,
                output_vertex_count=None,
                output_face_count=None,
                component_infos=component_infos,
                final_keep_face_mask=final_keep_face_mask,
                output_path=output_path,
                report_limit=args.report_limit,
                top_visible_summary=top_visible_summary,
            )
            print()
            print("dry-run: output OBJ was not written.")
            return 0

        print("applying final face mask...", flush=True)
        mesh.update_faces(final_keep_face_mask)
        mesh.remove_unreferenced_vertices()

        print("exporting mesh...", flush=True)
        mesh.export(output_path)

        print_report(
            original_vertex_count=original_vertex_count,
            virtual_vertex_count=virtual_vertex_count,
            original_face_count=original_face_count,
            output_vertex_count=len(mesh.vertices),
            output_face_count=len(mesh.faces),
            component_infos=component_infos,
            final_keep_face_mask=final_keep_face_mask,
            output_path=output_path,
            report_limit=args.report_limit,
            top_visible_summary=top_visible_summary,
        )

        print()
        print(f"wrote: {output_path}")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())