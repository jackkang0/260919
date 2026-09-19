"""
產生 2 組 1920x1080、各 6 frames 的「小區塊閃爍」測試序列，並輸出 ground truth CSV。
閃爍區塊大小 1x1 ~ 4x4，位置隨機（x/y 奇偶亦隨機 -> 與 2x2 block 網格對齊與否隨機）。
全圖皆有 8-bit dither +1。

  S1_small_alt_    全部為 ±a 逐幀交錯；通道隨機 R/G/B/Y(RGB 同步)
  S2_small_mixed_  時間 pattern 隨機:
                     alt    : ±a 逐幀交錯
                     spike  : 只有 1 個隨機 frame +a
                     random : 每幀隨機 ±a
                     antiRB : R +a / B -a 反相交錯 (L1 抵銷)

ground truth: <prefix>groundtruth.csv
  id, x, y, size, channel, pattern, amp, spike_frame, grid_aligned
  (x,y 為左上角；grid_aligned = x,y 皆為偶數且 size 為偶數，即完全落在 2x2 網格上)

用法: python make_kpi_test_small.py [輸出資料夾] [每組區塊數]
      預設 ./test_1080p_small, 600
"""
import os
import sys
import csv
import numpy as np
import cv2

W, H, N = 1920, 1080, 6
OUT = sys.argv[1] if len(sys.argv) > 1 else "test_1080p_small"
N_BLOCKS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
CELL = 40                         # 每個 cell 最多放 1 個區塊 -> 區塊之間不重疊、不相鄰
CH_IDX = {"R": [0], "G": [1], "B": [2], "Y": [0, 1, 2]}


def base_image():
    """與 make_kpi_test_1080p 相同構圖，數值壓到 75~180，避免 ±a 被 clip"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([40 + xx / W * 150,
                    60 + yy / H * 120,
                    170 - (xx / W + yy / H) * 60], axis=-1)
    img[700:1000, 100:500] = (180, 70, 60)
    img[750:950, 1500:1850] = (50, 150, 190)
    cv2.circle(img, (960, 780), 150, (190, 185, 90), -1)
    lo, hi = img.min(), img.max()
    return 75 + (img - lo) / (hi - lo) * 105          # 75~180: ±70 (+dither) 不會 clip


def place_blocks(rng, n):
    cells = [(cy, cx) for cy in range(H // CELL) for cx in range(W // CELL)]
    pick = rng.choice(len(cells), size=n, replace=False)
    blocks = []
    for k, ci in enumerate(sorted(pick)):
        cy, cx = cells[ci]
        size = int(rng.integers(1, 5))                          # 1..4
        x = cx * CELL + int(rng.integers(0, CELL - size - 4))   # 右/下留 >=4 px 間隔
        y = cy * CELL + int(rng.integers(0, CELL - size - 4))
        blocks.append({"id": k, "x": x, "y": y, "size": size,
                       "grid_aligned": int(x % 2 == 0 and y % 2 == 0 and size % 2 == 0)})
    return blocks


def temporal(pattern, a, rng):
    """回傳長度 N 的 offset 序列"""
    if pattern == "alt":
        return np.array([a if i % 2 == 0 else -a for i in range(N)], np.float32), -1
    if pattern == "spike":
        f = int(rng.integers(0, N))
        t = np.zeros(N, np.float32)
        t[f] = a
        return t, f
    if pattern == "random":
        return rng.choice([-a, a], size=N).astype(np.float32), -1
    raise ValueError(pattern)


def build(prefix, patterns, seed):
    rng = np.random.default_rng(seed)
    B = base_image()
    frames = np.repeat(B[None], N, axis=0)                      # (N,H,W,3)
    blocks = place_blocks(rng, N_BLOCKS)

    for b in blocks:
        pat = str(rng.choice(patterns))
        a = float(rng.integers(8, 71))                          # 8 ~ 70
        ys, xs = slice(b["y"], b["y"] + b["size"]), slice(b["x"], b["x"] + b["size"])
        if pat == "antiRB":
            t, sf = temporal("alt", a, rng)
            frames[:, ys, xs, 0] += t[:, None, None]
            frames[:, ys, xs, 2] -= t[:, None, None]
            ch = "R+/B-"
        else:
            ch = str(rng.choice(list(CH_IDX)))
            t, sf = temporal(pat, a, rng)
            for c in CH_IDX[ch]:
                frames[:, ys, xs, c] += t[:, None, None]
        b.update(channel=ch, pattern=pat, amp=a, spike_frame=sf + 1 if sf >= 0 else "")

    for i in range(N):
        f = frames[i] + rng.integers(0, 2, frames[i].shape)    # dither +1
        bgr = np.clip(np.floor(f), 0, 255).astype(np.uint8)[:, :, ::-1]
        ok, buf = cv2.imencode(".bmp", bgr)
        buf.tofile(os.path.join(OUT, f"{prefix}{i+1:04d}.bmp"))

    cols = ["id", "x", "y", "size", "channel", "pattern", "amp", "spike_frame",
            "grid_aligned"]
    with open(os.path.join(OUT, f"{prefix}groundtruth.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for b in blocks:
            wr.writerow({k: b[k] for k in cols})
    return blocks


def main():
    os.makedirs(OUT, exist_ok=True)
    build("S1_small_alt_", ["alt"], seed=11)
    build("S2_small_mixed_", ["alt", "spike", "random", "antiRB"], seed=22)
    print(f"已輸出 2 組 x {N} frames ({W}x{H}), 每組 {N_BLOCKS} 個 1x1~4x4 閃爍區塊 -> "
          f"{os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
