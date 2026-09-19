"""
Flicker KPI 批次 / 命令列版（無需 GUI，可掛進 regression 流程）

用法:
  單一序列:
    python flicker_kpi_cli.py -i path/to/seq -o out_dir
  多序列（root 下每個子資料夾一組 6 張 frame）:
    python flicker_kpi_cli.py -i path/to/root --batch -o out_dir
  threshold 掃描（決定門檻該訂多少）:
    python flicker_kpi_cli.py -i path/to/seq --sweep pixel:R_variance
  量測噪聲底線（拿一組確定無 flicker 的序列跑，看 dither 本身貢獻多少）:
    python flicker_kpi_cli.py -i path/to/clean_seq --noise-floor
"""

import os
import csv
import glob
import argparse

import numpy as np
import cv2

from flicker_kpi_core import (
    DEFAULT_THRESHOLDS, METRIC_NAMES,
    analyze_sequence, format_full_report, combined_fail_map,
    render_heatmap, upscale,
)


def export_sequence(result, out_dir, modes=("ratio", "binary")):
    os.makedirs(out_dir, exist_ok=True)
    H = result["shape"][0]
    for level in ("pixel", "block"):
        rep, maps = result[level]["report"], result[level]["maps"]
        for n in METRIC_NAMES:
            for mode in modes:
                vis = render_heatmap(maps[n], rep[n]["threshold"], mode=mode)
                vis = upscale(vis, max(1, H // vis.shape[0]))
                cv2.imwrite(os.path.join(out_dir, f"{level}_{n}_{mode}.png"), vis)
        acc = combined_fail_map(rep)
        mx = max(int(acc.max()), 1)
        vis = cv2.applyColorMap((acc / mx * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        vis = upscale(vis, max(1, H // vis.shape[0]))
        cv2.imwrite(os.path.join(out_dir, f"{level}_combined_max{mx}.png"), vis)


def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["sequence", "level", "metric", "threshold", "fail_ratio_%",
                     "fail_count", "total", "max", "p99.9", "p99", "mean",
                     "max/threshold", "verdict"])
        for r in results:
            seq = os.path.basename(r["folder"].rstrip("/\\"))
            for level in ("pixel", "block"):
                rep = r[level]["report"]
                for n in METRIC_NAMES:
                    s = rep[n]
                    wr.writerow([seq, level, n, s["threshold"],
                                 f"{s['fail_ratio']*100:.6f}", s["fail_count"],
                                 s["total"], f"{s['max']:.3f}", f"{s['p999']:.3f}",
                                 f"{s['p99']:.3f}", f"{s['mean']:.4f}",
                                 f"{s['margin']:.3f}",
                                 "FAIL" if s["fail_count"] else "PASS"])


def sweep(result, spec, points=25):
    """spec 格式 'pixel:R_variance'，輸出 threshold -> 超標比例曲線"""
    level, metric = spec.split(":")
    m = result[level]["maps"][metric]
    default = DEFAULT_THRESHOLDS[level][metric]
    hi = max(float(m.max()) * 1.05, default * 1.5)
    print(f"\n[{level}] {metric}  (預設 threshold = {default})")
    print(f"{'threshold':>12}{'fail%':>12}{'fail_cnt':>10}")
    for t in np.linspace(0, hi, points):
        f = m > t
        print(f"{t:>12.2f}{f.mean()*100:>11.4f}%{int(f.sum()):>10d}")


def noise_floor(result):
    """
    量測 metric 的分布上緣。若輸入序列確定無 flicker，這就是 dither 造成的
    噪聲底線 —— threshold 必須高於此值，否則正常 dither 會被判為 flicker。
    """
    for level in ("pixel", "block"):
        print(f"\n--- {level} noise floor ---")
        print(f"{'metric':<15}{'p99':>12}{'p99.9':>12}{'max':>12}"
              f"{'default_thr':>14}{'headroom':>11}")
        maps = result[level]["maps"]
        for n in METRIC_NAMES:
            m = maps[n]
            mx = float(m.max())
            thr = DEFAULT_THRESHOLDS[level][n]
            print(f"{n:<15}{np.percentile(m,99):>12.3f}"
                  f"{np.percentile(m,99.9):>12.3f}{mx:>12.3f}"
                  f"{thr:>14.1f}{(thr/mx if mx>0 else float('inf')):>10.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="序列資料夾（或 --batch 時的 root）")
    ap.add_argument("-o", "--output", default=None, help="輸出資料夾")
    ap.add_argument("-p", "--pattern", default="*.bmp")
    ap.add_argument("--batch", action="store_true", help="root 下每個子資料夾為一組序列")
    ap.add_argument("--block", type=int, default=2, help="block-based 尺寸，預設 2")
    ap.add_argument("--diff-mode", default="peak_to_peak",
                    choices=["peak_to_peak", "max_adjacent"])
    ap.add_argument("--sweep", default=None, help="例如 pixel:R_variance")
    ap.add_argument("--noise-floor", action="store_true")
    ap.add_argument("--no-images", action="store_true", help="只出 CSV，不出 heatmap")
    args = ap.parse_args()

    folders = (sorted(f for f in glob.glob(os.path.join(args.input, "*"))
                      if os.path.isdir(f))
               if args.batch else [args.input])

    results = []
    for folder in folders:
        try:
            r = analyze_sequence(folder, args.pattern, DEFAULT_THRESHOLDS,
                                 args.diff_mode, args.block)
        except (IOError, ValueError) as e:
            print(f"[跳過] {folder}: {e}")
            continue
        results.append(r)
        print(format_full_report(r))
        print()

        if args.output and not args.no_images:
            name = os.path.basename(folder.rstrip("/\\"))
            export_sequence(r, os.path.join(args.output, name))

    if args.output and results:
        os.makedirs(args.output, exist_ok=True)
        write_csv(results, os.path.join(args.output, "flicker_kpi_report.csv"))
        print(f"CSV 已輸出: {os.path.join(args.output, 'flicker_kpi_report.csv')}")

    if results and args.sweep:
        sweep(results[0], args.sweep)
    if results and args.noise_floor:
        noise_floor(results[0])


if __name__ == "__main__":
    main()
