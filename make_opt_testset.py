"""
產生 Threshold 最佳化功能的測試資料集（480x270，每組 6 frames，8-bit dither +1）

  opt_clean/  6 組 (c0~c5)       全部不閃
  opt_mixed/  6 組 + flicker_labels.csv
              m_f1  30 個 1~3px 區塊，G ±25 逐幀交錯   -> 閃
              m_f2   5 個 1~3px 區塊，G ±40 逐幀交錯   -> 閃
              m_f3  60 個 1~3px 區塊，G ±14 逐幀交錯   -> 閃
              m_c1~m_c3                                -> 不閃

所有序列（含不閃）都有 40 個隨機單點擾動：每幀 RGB 加 N(0, a)，a ~ U(3, 12)，
模擬壓縮 + SR 造成的非週期小擾動，讓「不閃」序列也有非零的 metric 值。

用法: python make_opt_testset.py [輸出根目錄]   (預設 ./opt_testset)
"""
import os
import sys
import csv

import numpy as np
import cv2

H, W, N = 270, 480, 6
ROOT = sys.argv[1] if len(sys.argv) > 1 else "opt_testset"
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)


def base(seed):
    r = np.random.default_rng(seed)
    return np.stack([60 + xx / W * 100 * r.random(),
                     80 + yy / H * 90 * r.random(),
                     100 + (xx + yy) / (W + H) * 80], -1)


def write_bmp(path, rgb):
    ok, buf = cv2.imencode(".bmp", np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1])
    buf.tofile(path)


def seq(d, name, seed, n_flicker=0, amp=0.0):
    os.makedirs(d, exist_ok=True)
    B = base(seed)
    r = np.random.default_rng(seed + 100)
    pts = [(r.integers(0, H - 4), r.integers(0, W - 4)) for _ in range(40)]
    amps = r.uniform(3, 12, len(pts))
    fl = [(r.integers(0, H - 4), r.integers(0, W - 4), r.integers(1, 4))
          for _ in range(n_flicker)]
    for i in range(N):
        f = B + r.integers(0, 2, B.shape)                       # dither +1
        for (y, x), a in zip(pts, amps):
            f[y, x, :] += r.normal(0, a)                        # 非週期擾動
        for (y, x, s) in fl:
            f[y:y + s, x:x + s, 1] += amp * (1 if i % 2 else -1)  # G 逐幀交錯
        write_bmp(os.path.join(d, f"{name}_{i+1:04d}.bmp"), f)


def main():
    clean = os.path.join(ROOT, "opt_clean")
    mixed = os.path.join(ROOT, "opt_mixed")
    for k in range(6):
        seq(clean, f"c{k}", k)
    spec = {"m_f1": (30, 25), "m_f2": (5, 40), "m_f3": (60, 14),
            "m_c1": (0, 0), "m_c2": (0, 0), "m_c3": (0, 0)}
    for k, (name, (n, a)) in enumerate(spec.items()):
        seq(mixed, name, 50 + k, n, a)
    with open(os.path.join(mixed, "flicker_labels.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["sequence", "label"])
        for name in sorted(spec):
            wr.writerow([name, 1 if name.startswith("m_f") else 0])
    print(f"已輸出到 {os.path.abspath(ROOT)}")


if __name__ == "__main__":
    main()
