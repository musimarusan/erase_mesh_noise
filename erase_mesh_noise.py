#!/usr/bin/env python
"""
erase_mesh_noise.py

Scaniverse 由来 OBJ メッシュの孤立小成分を除去する初期実験用スクリプト。

仕様:
    - 入力ファイルはコマンドライン引数で指定する
    - 出力ファイルは入力ファイルと同じ階層に作成する
    - 出力ファイル名は basename に _erase を付ける

例:
    input.obj        -> input_erase.obj
    sample_mesh.obj  -> sample_mesh_erase.obj

注意:
    - Z 値だけでは判定しない
    - 凹部をノイズ扱いしない
    - 本体と接続している偽面は、この初期版では除去できない
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
    index: int
    face_count: int
    vertex_count: int
    area: float
    z_min: float
    z_max: float
    keep: bool


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

    trimesh が Scene として読む場合は、内部ジオメトリを結合して単一メッシュ化する。
    初期版では、テクスチャやマテリアル保持よりも形状処理を優先する。
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

        try:
            mesh = loaded.dump(concatenate=True)
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
        except Exception:
            pass

        geometries = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh)
        ]

        if not geometries:
            raise ValueError("Scene contains no Trimesh geometry.")

        return trimesh.util.concatenate(geometries)

    raise TypeError(f"Unsupported loaded object type: {type(loaded)}")


def split_connected_components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """
    メッシュを連結成分に分解する。

    only_watertight=False にする。
    Scaniverse の遺構面メッシュは閉じた立体ではない可能性が高いため、
    watertight 成分だけに限定しない。
    """
    components = mesh.split(only_watertight=False)

    if len(components) == 0:
        return [mesh]

    return list(components)


def decide_components_to_keep(
    components: list[trimesh.Trimesh],
    min_faces: int,
    min_ratio: float,
    keep_largest_only: bool,
) -> tuple[list[trimesh.Trimesh], list[ComponentInfo]]:
    """
    残す連結成分を決める。

    基本方針:
        - 最大成分は必ず残す
        - それ以外は face 数で判定する
        - Z 値による削除はしない

    閾値:
        max(min_faces, 最大成分の face 数 * min_ratio)
    """
    if not components:
        raise ValueError("No components to process.")

    face_counts = np.array([len(c.faces) for c in components], dtype=int)

    largest_index = int(np.argmax(face_counts))
    largest_face_count = int(face_counts[largest_index])

    if keep_largest_only:
        keep_flags = np.zeros(len(components), dtype=bool)
        keep_flags[largest_index] = True
    else:
        threshold = max(min_faces, int(np.ceil(largest_face_count * min_ratio)))
        keep_flags = face_counts >= threshold
        keep_flags[largest_index] = True

    infos: list[ComponentInfo] = []

    for i, component in enumerate(components):
        bounds = component.bounds

        if bounds is None or not np.all(np.isfinite(bounds)):
            z_min = float("nan")
            z_max = float("nan")
        else:
            z_min = float(bounds[0, 2])
            z_max = float(bounds[1, 2])

        infos.append(
            ComponentInfo(
                index=i,
                face_count=int(len(component.faces)),
                vertex_count=int(len(component.vertices)),
                area=float(component.area),
                z_min=z_min,
                z_max=z_max,
                keep=bool(keep_flags[i]),
            )
        )

    kept_components = [
        component
        for component, keep in zip(components, keep_flags)
        if keep
    ]

    return kept_components, infos


def merge_components(components: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """
    残す連結成分を単一メッシュに戻す。
    """
    if not components:
        raise ValueError("No components were kept.")

    if len(components) == 1:
        result = components[0].copy()
    else:
        result = trimesh.util.concatenate(components)

    result.remove_unreferenced_vertices()
    return result


def print_report(
    input_mesh: trimesh.Trimesh,
    output_mesh: trimesh.Trimesh | None,
    component_infos: list[ComponentInfo],
    output_path: Path | None,
) -> None:
    """
    処理結果を標準出力に表示する。
    """
    print("=== erase_mesh_noise report ===")
    print(f"input vertices : {len(input_mesh.vertices)}")
    print(f"input faces    : {len(input_mesh.faces)}")
    print(f"components     : {len(component_infos)}")

    if output_path is not None:
        print(f"output path    : {output_path}")

    print()

    print(
        f"{'idx':>5} "
        f"{'faces':>12} "
        f"{'vertices':>12} "
        f"{'area':>14} "
        f"{'z_min':>12} "
        f"{'z_max':>12} "
        f"{'decision':>10}"
    )

    for info in sorted(component_infos, key=lambda x: x.face_count, reverse=True):
        decision = "KEEP" if info.keep else "REMOVE"

        print(
            f"{info.index:5d} "
            f"{info.face_count:12d} "
            f"{info.vertex_count:12d} "
            f"{info.area:14.6f} "
            f"{info.z_min:12.6f} "
            f"{info.z_max:12.6f} "
            f"{decision:>10}"
        )

    if output_mesh is not None:
        print()
        print(f"output vertices: {len(output_mesh.vertices)}")
        print(f"output faces   : {len(output_mesh.faces)}")
        print(f"removed faces  : {len(input_mesh.faces) - len(output_mesh.faces)}")


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
        default=100,
        help=(
            "Minimum face count for keeping a component. "
            "Default: 100"
        ),
    )

    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.001,
        help=(
            "Minimum face ratio relative to the largest component. "
            "Default: 0.001"
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path = make_output_path(input_path)

    try:
        mesh = load_obj_as_mesh(input_path)
        components = split_connected_components(mesh)

        kept_components, component_infos = decide_components_to_keep(
            components=components,
            min_faces=args.min_faces,
            min_ratio=args.min_ratio,
            keep_largest_only=args.keep_largest_only,
        )

        if args.dry_run:
            print_report(
                input_mesh=mesh,
                output_mesh=None,
                component_infos=component_infos,
                output_path=output_path,
            )
            print()
            print("dry-run: output OBJ was not written.")
            return 0

        cleaned_mesh = merge_components(kept_components)
        cleaned_mesh.export(output_path)

        print_report(
            input_mesh=mesh,
            output_mesh=cleaned_mesh,
            component_infos=component_infos,
            output_path=output_path,
        )

        print()
        print(f"wrote: {output_path}")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())