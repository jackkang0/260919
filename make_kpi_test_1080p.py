"""
產生 4 組 1920x1080、各 6 frames 的 flicker KPI 測試序列（同一資料夾，依 prefix 分組）
全圖皆有 8-bit dither +1（每 frame 每 pixel 每通道隨機 0/+1）。

  T1_clean_    只有 dither                         -> 應全 SAFE / PASS
  T2_strong_   R/B ±60 反相 (L1 抵銷)、G ±20       -> FAIL / SEVERE
  T3_graded_   4 塊 G ±a 交錯, a 依 block G_variance(thr 200) 設計:
               a=11 -> WATCH, a=13.4 -> NEAR, a=17.3 -> FAIL, a=24.5 -> SEVERE
  T4_temporal_ (a) 只有 frame 4 +90 單次突跳
               (b) 1-pixel 棋盤格相位 ±40 (pixel-based 會抓到, 2x2 平均後抵銷)
               (c) 奇數座標起點區塊 G ±20 (測 2x2 block 未對齊)
               (d) 全圖亮度 0..+5 緩慢漂移 (整體 pumping, 應不致 FAIL)

用法: python make_kpi_test_1080p.py [輸出資料夾]   (預設 ./test_1080p)
"""
import os
import sys
import numpy as np
import cv2

W, H, N = 1920, 1080, 6
OUT = sys.argv[1] if len(sys.argv) > 1 else "test_1080p"
rng = np.random.default_rng(2026)


def base_image():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.stack([40 + xx / W * 150,
                    60 + yy / H * 120,
                    170 - (xx / W + yy / H) * 60], axis=-1)
    # 幾個色塊 / 圓, 讓背景不是單純漸層
    img[700:1000, 100:500] = (200, 60, 50)
    img[750:950, 1500:1850] = (40, 160, 200)
    cv2.circle(img, (960, 780), 150, (230, 220, 90), -1)
    return img


def write(prefix, frames):
    for i, f in enumerate(frames, 1):
        f = f + rng.integers(0, 2, f.shape)                     # dither +1
        bgr = np.clip(np.floor(f), 0, 255).astype(np.uint8)[:, :, ::-1]
        ok, buf = cv2.imencode(".bmp", bgr)
        buf.tofile(os.path.join(OUT, f"{prefix}{i:04d}.bmp"))  # 支援中文路徑


def alt(i, a):
    return a if i % 2 == 0 else -a


def main():
    os.makedirs(OUT, exist_ok=True)
    B = base_image()

    # T1 clean
    write("T1_clean_", [B.copy() for _ in range(N)])

    # T2 strong
    fr = []
    for i in range(N):
        f = B.copy()
        f[100:400, 200:700, 0] += alt(i, 60)
        f[100:400, 200:700, 2] -= alt(i, 60)
        f[500:700, 1200:1600, 1] += alt(i, 20)
        fr.append(f)
    write("T2_strong_", fr)

    # T3 graded (G 通道, 4 塊由左到右 WATCH / NEAR / FAIL / SEVERE)
    amps = [11.0, 13.4, 17.3, 24.5]
    fr = []
    for i in range(N):
        f = B.copy()
        for k, a in enumerate(amps):
            x0 = 160 + k * 420
            f[300:600, x0:x0 + 320, 1] += alt(i, a)
        fr.append(f)
    write("T3_graded_", fr)

    # T4 temporal / structural
    yy, xx = np.mgrid[0:H, 0:W]
    checker = ((yy + xx) % 2 * 2 - 1).astype(np.float32)        # ±1 棋盤
    fr = []
    for i in range(N):
        f = B.copy()
        if i == 3:
            f[100:300, 150:550] += 90                          # (a) 單次突跳
        f[100:400, 800:1200, 1] += alt(i, 40) * checker[100:400, 800:1200]  # (b)
        f[101:301, 1401:1701, 1] += alt(i, 20)                 # (c) 奇數起點
        f += i                                                 # (d) 0..5 漂移
        fr.append(f)
    write("T4_temporal_", fr)

    print(f"已輸出 4 組 x {N} frames ({W}x{H}) 到 {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
