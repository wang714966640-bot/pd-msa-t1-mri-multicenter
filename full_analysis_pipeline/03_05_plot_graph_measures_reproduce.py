# -*- coding: utf-8 -*-
"""复现脚本（Brainspace 路线）：

目标：
- 不覆盖旧结果，输出到全新文件夹
- 复现/生成两类脑图（带色条/刻度尺）：
  1) FC/SC 的 8 个局部拓扑指标（clustering/efficiency/degree/betweenness）
  2) 结构指标的 Area / Thickness（来自 surface_metrics.npy）

输入（默认自动从 PD_analyse 目录寻找）：
- PD_analyse/FC_SC_MatPlots/*.csv
- PD_analyse/surface_metrics.npy

输出（默认）：
- PD_analyse/demo_1126_orig/MSN Result (Figure)/reproduce_<timestamp>/brainspace/

用法：
python 6.graph_measures_plot_reproduce.py
python 6.graph_measures_plot_reproduce.py --out_dir "/your/output/dir"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from brainspace.datasets import load_conte69, load_parcellation
from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres


@dataclass(frozen=True)
class LocalMeasureSpec:
    key: str
    title: str
    color_range: tuple[float, float]


LOCAL_SPECS = [
    LocalMeasureSpec("clustering_fc", "Clustering FC", (0.2, 0.45)),
    LocalMeasureSpec("clustering_sc", "Clustering SC", (0.0, 0.7)),
    LocalMeasureSpec("local_efficiency_fc", "Efficiency FC", (0.0, 0.15)),
    LocalMeasureSpec("local_efficiency_sc", "Efficiency SC", (0.5, 1.0)),
    LocalMeasureSpec("degree_fc", "Degree FC", (20, 80)),
    LocalMeasureSpec("degree_sc", "Degree SC", (12300, 80000)),
    LocalMeasureSpec("betweenness_fc", "Betweenness FC", (46, 800)),
    LocalMeasureSpec("betweenness_sc", "Betweenness SC", (100, 2000)),
]


def _infer_pd_analyse_dir() -> Path:
    # 本文件位于：PD_analyse/最终文章使用代码及原始文件/1.code/
    code_dir = Path(__file__).resolve().parent
    pd_analyse_dir = code_dir.parent.parent
    return pd_analyse_dir


def _default_out_dir(pd_analyse_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return pd_analyse_dir / "demo_1126_orig" / "MSN Result (Figure)" / f"reproduce_{ts}" / "brainspace"


def _read_measure_csv(fc_sc_dir: Path, key: str) -> np.ndarray:
    fp = fc_sc_dir / f"{key}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"缺少输入文件: {fp}")
    df = pd.read_csv(fp, header=None)
    # 期望 shape: (n_subject, 400)
    vec = df.mean(axis=0).to_numpy(dtype=float)
    return vec


def _map_parcel_to_vertex(vec_400: np.ndarray, labeling: np.ndarray) -> np.ndarray:
    if vec_400.ndim != 1:
        vec_400 = vec_400.reshape(-1)
    if vec_400.shape[0] != 400:
        raise ValueError(f"期望 400 个 parcel 值，实际: {vec_400.shape[0]}")
    mask = labeling != 0
    return map_to_labels(vec_400, labeling, mask=mask, fill=np.nan)


def plot_local_measures(fc_sc_dir: Path, out_dir: Path) -> Path:
    labeling = load_parcellation("schaefer", scale=400, join=True)
    surf_lh, surf_rh = load_conte69()

    measures_vertex = []
    titles = []
    color_ranges = []

    for spec in LOCAL_SPECS:
        vec = _read_measure_csv(fc_sc_dir, spec.key)
        measures_vertex.append(_map_parcel_to_vertex(vec, labeling))
        titles.append(spec.title)
        color_ranges.append(spec.color_range)

    out_png = out_dir / "local_measures_brainspace_with_colorbar.png"
    plot_hemispheres(
        surf_lh,
        surf_rh,
        array_name=measures_vertex,
        size=(1400, 1900),
        color_range=color_ranges,
        cmap="PiYG_r",
        color_bar=True,
        filename=str(out_png),
        transparent_bg=False,
        screenshot=True,
        embed_nb=False,
        interactive=False,
        nan_color=(0.5, 0.5, 0.5, 1),
        label_text=titles,
        zoom=1.0,
    )

    # 额外：每个指标单独出图（更适合“每张图一个色条”）
    single_dir = out_dir / "local_measures_single"
    single_dir.mkdir(parents=True, exist_ok=True)
    for (spec, vtx) in zip(LOCAL_SPECS, measures_vertex):
        one_png = single_dir / f"{spec.key}_brainspace.png"
        plot_hemispheres(
            surf_lh,
            surf_rh,
            array_name=vtx,
            size=(1100, 320),
            color_range=spec.color_range,
            cmap="PiYG_r",
            color_bar="right",
            filename=str(one_png),
            transparent_bg=False,
            screenshot=True,
            embed_nb=False,
            interactive=False,
            share="both",
            nan_color=(0.5, 0.5, 0.5, 1),
            label_text=[spec.title],
            zoom=1.0,
        )

    return out_png


def plot_area_thickness(surface_metrics_npy: Path, out_dir: Path) -> Path:
    if not surface_metrics_npy.exists():
        raise FileNotFoundError(f"缺少输入文件: {surface_metrics_npy}")

    struc = np.load(surface_metrics_npy)
    # 兼容你现有代码：np.mean(struc_data, axis=1)[:, i]
    mean_by_subject = np.mean(struc, axis=1)
    if mean_by_subject.shape[0] != 400:
        raise ValueError(f"surface_metrics.npy 期望 400 parcels，实际 mean 后: {mean_by_subject.shape}")

    area = mean_by_subject[:, 1].astype(float)
    thickness = mean_by_subject[:, 2].astype(float)

    labeling = load_parcellation("schaefer", scale=400, join=True)
    surf_lh, surf_rh = load_conte69()

    area_vtx = _map_parcel_to_vertex(area, labeling)
    thick_vtx = _map_parcel_to_vertex(thickness, labeling)

    out_png = out_dir / "area_thickness_brainspace_with_colorbar.png"
    plot_hemispheres(
        surf_lh,
        surf_rh,
        array_name=[area_vtx, thick_vtx],
        size=(1200, 700),
        color_range=[(0.6, 0.85), (1.5, 3.5)],
        cmap="PiYG_r",
        color_bar=True,
        filename=str(out_png),
        transparent_bg=False,
        screenshot=True,
        embed_nb=False,
        interactive=False,
        nan_color=(0.5, 0.5, 0.5, 1),
        label_text=["Area", "Thickness"],
        zoom=1.0,
    )

    return out_png


def export_ggseg_inputs(fc_sc_dir: Path, surface_metrics_npy: Path, out_dir: Path) -> tuple[Path, Path]:
    # 1) local measures: 每个 metric 400 行（按 index 顺序）
    rows = []
    for spec in LOCAL_SPECS:
        vec = _read_measure_csv(fc_sc_dir, spec.key)
        for i, v in enumerate(vec, start=1):
            rows.append({"metric": spec.title, "idx": i, "value": float(v)})
    df_local = pd.DataFrame(rows)
    local_csv = out_dir / "ggseg_input_local_measures.csv"
    df_local.to_csv(local_csv, index=False)

    # 2) area/thickness
    struc = np.load(surface_metrics_npy)
    mean_by_subject = np.mean(struc, axis=1)
    area = mean_by_subject[:, 1].astype(float)
    thickness = mean_by_subject[:, 2].astype(float)
    rows2 = []
    for name, vec in [("Area", area), ("Thickness", thickness)]:
        for i, v in enumerate(vec, start=1):
            rows2.append({"metric": name, "idx": i, "value": float(v)})
    df_at = pd.DataFrame(rows2)
    at_csv = out_dir / "ggseg_input_area_thickness.csv"
    df_at.to_csv(at_csv, index=False)

    return local_csv, at_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pd_analyse_dir",
        type=str,
        default=None,
        help="PD_analyse 目录（默认自动推断）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录（默认自动创建到 MSN Result (Figure)/reproduce_<timestamp>/brainspace）",
    )
    args = parser.parse_args()

    pd_analyse_dir = Path(args.pd_analyse_dir).resolve() if args.pd_analyse_dir else _infer_pd_analyse_dir()
    fc_sc_dir = pd_analyse_dir / "FC_SC_MatPlots"
    surface_metrics_npy = pd_analyse_dir / "surface_metrics.npy"

    out_dir = Path(args.out_dir).resolve() if args.out_dir else _default_out_dir(pd_analyse_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[brainspace reproduce] pd_analyse_dir: {pd_analyse_dir}")
    print(f"[brainspace reproduce] out_dir: {out_dir}")

    p1 = plot_local_measures(fc_sc_dir, out_dir)
    print(f"[brainspace reproduce] saved: {p1}")

    p2 = plot_area_thickness(surface_metrics_npy, out_dir)
    print(f"[brainspace reproduce] saved: {p2}")

    local_csv, at_csv = export_ggseg_inputs(fc_sc_dir, surface_metrics_npy, out_dir)
    print(f"[brainspace reproduce] ggseg inputs: {local_csv}")
    print(f"[brainspace reproduce] ggseg inputs: {at_csv}")


if __name__ == "__main__":
    main()







