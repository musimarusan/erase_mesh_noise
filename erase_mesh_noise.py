#!/usr/bin/env python3
"""
erase_mesh_noise.py

Proposed by: GPT-5.5 Thinking

Scaniverse 由来 OBJ メッシュの孤立小成分を除去する実験用スクリプト。

仕様:
    - 入力ファイルはコマンドライン引数で指定する
    - 出力ファイルは入力ファイルと同じ階層に作成する
    - 出力ファイル名は basename に _erase を付ける

例:
    input.obj       -> input_erase.obj
    data/test.obj   -> data/test_erase.obj

方針:
    - trimesh.split() は使わない
    - raw vertex index ではなく、座標許容差つきの virtual vertex id で連結判定する
    - 面同士が virtual vertex id 上の辺を共有していれば同じ連結成分とみなす
    - 小さい連結成分を除去する
    - Z 値だけで削除しない
    - 凹部をノイズ扱いしない

注意:
    - 本体と接続している偽面・二重面は、この版でも原理的には除去対象外
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

    戻り値:
        virtual_vertex_ids:
            original vertex index -> virtual vertex id
        virtual_vertex_count:
            virtual vertex の総数
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

    戻り値:
        face_component_ids:
            各 face が属する component ID。
            shape = (face_count,)

        component_face_counts:
            component ごとの face 数。
            shape = (component_count,)
    """
    face_count = len(faces)

    if face_count == 0:
        raise ValueError("Mesh has no faces.")

    union_find = UnionFind(face_count)

    # original face vertex ids -> virtual face vertex ids
    virtual_faces = virtual_vertex_ids[faces]

    # key: sorted virtual edge tuple (vv0, vv1)
    # value: first face index that has this virtual edge
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
            # 座標丸め後に同一点になった辺は連結判定に使わない。
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

    初期値では min_ratio=0.0 なので、基本的には min_faces のみで判定する。
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


def summarize_decision(
    original_face_count: int,
    component_face_counts: np.ndarray,
    keep_component_flags: np.ndarray,
) -> dict[str, int | float]:
    """
    KEEP / REMOVE の集計値を作る。
    """
    kept_faces = int(component_face_counts[keep_component_flags].sum())
    removed_faces = int(original_face_count - kept_faces)

    kept_components = int(keep_component_flags.sum())
    removed_components = int(len(keep_component_flags) - kept_components)

    removed_face_ratio = (
        removed_faces / original_face_count
        if original_face_count > 0
        else 0.0
    )

    return {
        "kept_faces": kept_faces,
        "removed_faces": removed_faces,
        "kept_components": kept_components,
        "removed_components": removed_components,
        "removed_face_ratio": removed_face_ratio,
    }


def print_report(
    original_vertex_count: int,
    virtual_vertex_count: int,
    original_face_count: int,
    output_vertex_count: int | None,
    output_face_count: int | None,
    component_infos: list[ComponentInfo],
    output_path: Path,
    report_limit: int,
) -> None:
    """
    処理結果を標準出力に表示する。
    """
    kept_faces = sum(info.face_count for info in component_infos if info.keep)
    removed_faces = original_face_count - kept_faces

    kept_components = sum(1 for info in component_infos if info.keep)
    removed_components = len(component_infos) - kept_components

    removed_face_ratio = (
        removed_faces / original_face_count
        if original_face_count > 0
        else 0.0
    )

    print("=== erase_mesh_noise report ===")
    print(f"input vertices          : {original_vertex_count}")
    print(f"virtual vertices        : {virtual_vertex_count}")
    print(f"input faces             : {original_face_count}")
    print(f"components              : {len(component_infos)}")
    print(f"kept components         : {kept_components}")
    print(f"removed components      : {removed_components}")
    print(f"kept faces              : {kept_faces}")
    print(f"removed faces           : {removed_faces}")
    print(f"removed face ratio      : {removed_face_ratio:.4%}")
    print(f"output path             : {output_path}")
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
        print(f"output vertices         : {output_vertex_count}")
        print(f"output faces            : {output_face_count}")
        print(f"removed faces after save: {original_face_count - output_face_count}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove small disconnected components from a Scaniverse OBJ mesh."
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

        summary = summarize_decision(
            original_face_count=original_face_count,
            component_face_counts=component_face_counts,
            keep_component_flags=keep_component_flags,
        )

        print("decision summary:", flush=True)
        print(f"  components        : {len(component_face_counts)}", flush=True)
        print(f"  kept components   : {summary['kept_components']}", flush=True)
        print(f"  removed components: {summary['removed_components']}", flush=True)
        print(f"  kept faces        : {summary['kept_faces']}", flush=True)
        print(f"  removed faces     : {summary['removed_faces']}", flush=True)
        print(
            f"  removed face ratio: {summary['removed_face_ratio']:.4%}",
            flush=True,
        )

        component_infos = build_component_infos(
            mesh=mesh,
            face_component_ids=face_component_ids,
            component_face_counts=component_face_counts,
            keep_component_flags=keep_component_flags,
            virtual_vertex_ids=virtual_vertex_ids,
        )

        keep_face_mask = keep_component_flags[face_component_ids]

        if args.dry_run:
            print_report(
                original_vertex_count=original_vertex_count,
                virtual_vertex_count=virtual_vertex_count,
                original_face_count=original_face_count,
                output_vertex_count=None,
                output_face_count=None,
                component_infos=component_infos,
                output_path=output_path,
                report_limit=args.report_limit,
            )
            print()
            print("dry-run: output OBJ was not written.")
            return 0

        print("removing small components...", flush=True)
        mesh.update_faces(keep_face_mask)
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
            output_path=output_path,
            report_limit=args.report_limit,
        )

        print()
        print(f"wrote: {output_path}")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())