from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 plyfile 读取并显示 PLY 点云")
    parser.add_argument(
        "ply_path",
        nargs="?",
        default="PointCloud_Astra Pro_20260416_112451.ply",
        help="PLY 文件路径（默认读取当前目录示例文件）",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.2,
        help="绘制点大小（matplotlib scatter size，默认: 0.2）",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120000,
        help="最多显示的点数（默认: 120000，0 表示不限制）",
    )
    parser.add_argument(
        "--bg",
        choices=["black", "white"],
        default="black",
        help="背景颜色（默认: black）",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="保留无效点（NaN/Inf/全零点）",
    )
    return parser.parse_args()


def load_point_cloud(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise RuntimeError(f"PLY 中没有 vertex 元素: {ply_path}")

    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise RuntimeError(f"PLY 顶点缺少 x/y/z 字段: {ply_path}")

    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    if len(points) == 0:
        raise RuntimeError(f"点云为空或读取失败: {ply_path}")

    colors: np.ndarray | None = None
    color_candidates = [("red", "green", "blue"), ("r", "g", "b")]
    for r_name, g_name, b_name in color_candidates:
        if {r_name, g_name, b_name}.issubset(names):
            colors = np.column_stack(
                [vertex[r_name], vertex[g_name], vertex[b_name]]
            ).astype(np.float32)
            break

    return points, colors


def filter_invalid_points(
    points: np.ndarray,
    colors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0)
    removed = int((~valid).sum())
    if removed == 0:
        return points, colors, 0

    points = points[valid]
    if colors is not None and len(colors) == len(valid):
        colors = colors[valid]
    return points, colors, removed


def downsample_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors, 0

    rng = np.random.default_rng(42)
    indices = rng.choice(len(points), size=max_points, replace=False)
    points_ds = points[indices]
    colors_ds = colors[indices] if colors is not None else None
    return points_ds, colors_ds, len(points) - max_points


def _set_axes_equal(ax: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def show_point_cloud(
    points: np.ndarray,
    colors: np.ndarray | None,
    title: str,
    point_size: float,
    bg: str,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if colors is not None and len(colors) == len(points):
        max_val = float(np.max(colors))
        if max_val > 1.0:
            rgb = np.clip(colors[:, :3] / 255.0, 0.0, 1.0)
        else:
            rgb = np.clip(colors[:, :3], 0.0, 1.0)
    else:
        rgb = np.array([[0.1, 0.8, 0.9]], dtype=np.float32)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=point_size,
        c=rgb,
        marker=".",
        linewidths=0,
    )

    bg_color = "black" if bg == "black" else "white"
    fg_color = "white" if bg == "black" else "black"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=fg_color)
    ax.xaxis.label.set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)
    ax.zaxis.label.set_color(fg_color)
    ax.title.set_color(fg_color)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax, points)
    plt.tight_layout()
    plt.show()


def main() -> None:
    args = parse_args()
    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"找不到文件: {ply_path}")

    points, colors = load_point_cloud(ply_path)
    before = len(points)

    removed_invalid = 0
    if not args.keep_invalid:
        points, colors, removed_invalid = filter_invalid_points(points, colors)

    points, colors, removed_sample = downsample_points(points, colors, args.max_points)
    after = len(points)

    print(f"Loaded: {ply_path}")
    print(
        f"Points: {before} -> {after} "
        f"(removed invalid {removed_invalid}, sampled out {removed_sample})"
    )
    print("matplotlib 3D 窗口中：左键旋转，滚轮缩放。")

    show_point_cloud(
        points=points,
        colors=colors,
        title=f"PLY Viewer - {ply_path.name}",
        point_size=args.point_size,
        bg=args.bg,
    )


if __name__ == "__main__":
    main()
